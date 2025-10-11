"""
Credits to Karpathy's nanoGPT repo for much of this code.
"""

# saves a dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import sys
import numpy as np
import tiktoken
import re

from datasets import DatasetDict
from tqdm import tqdm
from string import punctuation


enc = tiktoken.get_encoding("gpt2")
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc


def clean_text(text):
    """Clean input text by removing unwanted characters, HTML tags, and extra formatting."""
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Replace newlines and tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")
    # Normalize whitespace
    text = " ".join(text.split())
    # Lowercase
    text = text.lower()
    # Remove unwanted punctuation (keep common ones)
    allowed_punct = {"'", ".", ",", "!", "?"}
    text = text.translate(
        str.maketrans(
            "", "", "".join([c for c in punctuation if c not in allowed_punct])
        )
    )
    # Remove extra spaces again (in case punctuation removal added some)
    text = " ".join(text.split())
    return text


def process(example):
    """
    Take a text example and encode to ids using tiktoken GPT-2 BPE.
    """
    text = example["text"]
    text = clean_text(text)
    # encode
    ids = enc.encode_ordinary(text)  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    return {"ids": ids, "len": len(ids)}


def process_sft(examples):
    """
    Take examples with 'Question' and 'Answer' fields and encode to ids using tiktoken GPT-2 BPE.
    Works with batched processing to avoid serialization issues.
    Filters out examples with None or non-string Question/Answer fields.
    """
    # Handle both single example and batch
    if isinstance(examples["Question"], str):
        # Single example
        questions = [examples["Question"]]
        answers = [examples["Answer"]]
    else:
        # Batch
        questions = examples["Question"]
        answers = examples["Answer"]

    all_ids = []
    all_masks = []
    all_lens = []

    for q, a in zip(questions, answers):
        # Skip examples with None or non-string values
        if q is None or a is None or not isinstance(q, str) or not isinstance(a, str):
            continue

        # Skip examples that become empty after cleaning
        q_clean = clean_text(q)
        a_clean = clean_text(a)
        if not q_clean or not a_clean:
            continue

        q_ids = enc.encode_ordinary(q_clean + "\n")  # include the separator
        a_ids = enc.encode_ordinary(a_clean)
        ids = q_ids + a_ids
        ids.append(enc.eot_token)

        # 1 for assistant tokens, 0 for user tokens
        mask = [0] * len(q_ids) + [1] * (
            len(a_ids) + 1
        )  # +1 for eot if you want loss on eot

        all_ids.append(ids)
        all_masks.append(mask)
        all_lens.append(len(ids))

    # Return in the format expected by datasets
    if isinstance(examples["Question"], str):
        # Single example - return single values (or None if filtered out)
        if len(all_ids) == 0:
            return None
        return {"ids": all_ids[0], "mask": all_masks[0], "len": all_lens[0]}
    else:
        # Batch - return lists
        return {"ids": all_ids, "mask": all_masks, "len": all_lens}


def memmap(split, dset, dtype, suffix=""):
    """
    Take a tokenized dataset and save to binary files.
    """
    data_directory = os.path.join(os.path.dirname(__file__), "../data")
    data_directory = os.path.abspath(data_directory)
    arr_len = np.sum(dset["len"], dtype=np.uint64)  # type: ignore
    if suffix == "":
        filename = os.path.join(data_directory, f"{split}.bin")
    else:
        filename = os.path.join(data_directory, f"{split}_{suffix}.bin")
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))  # type: ignore
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])  # type: ignore
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    assert idx == arr_len
    return arr


def memmap_sft(split, dset, dtype, suffix="sft"):
    """
    Take a tokenized dataset and save to binary files for tokens and masks.
    """
    data_directory = os.path.join(os.path.dirname(__file__), "../data")
    data_directory = os.path.abspath(data_directory)
    arr_len = np.sum(dset["len"], dtype=np.uint64)
    token_file = os.path.join(
        data_directory, f'{split}{"" if suffix=="" else "_"+suffix}.bin'
    )
    mask_file = os.path.join(
        data_directory, f'{split}{"" if suffix=="" else "_"+suffix}_mask.bin'
    )

    tokens = np.memmap(token_file, dtype=dtype, mode="w+", shape=(arr_len,))  # type: ignore
    masks = np.memmap(mask_file, dtype=np.uint8, mode="w+", shape=(arr_len,))  # type: ignore

    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {token_file}"):
        batch = dset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        ids_batch = np.concatenate(batch["ids"])
        mask_batch = np.concatenate(batch["mask"])
        tokens[idx : idx + len(ids_batch)] = ids_batch
        masks[idx : idx + len(mask_batch)] = mask_batch
        idx += len(ids_batch)

    assert idx == arr_len
    return tokens, masks


def to_bins(tokenized, suffix="", is_sft=False):
    """
    Take a tokenized dataset and save to binary files.
    """
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

    # Auto-detect SFT format if not explicitly specified
    if not is_sft and suffix == "sft":
        is_sft = True

    for split, dset in tokenized.items():  # type: ignore
        if is_sft:
            tokens, masks = memmap_sft(split, dset, dtype, suffix)
            masks.flush()
        else:
            tokens = memmap(split, dset, dtype, suffix)
        # flush to disk
        tokens.flush()


def split_dataset_in_memory(ds, test_size=0.001, seed=42):
    n = len(ds)
    n_test = max(1, int(n * test_size))

    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)

    test_idxs = idxs[:n_test]
    train_idxs = idxs[n_test:]

    train_ds = ds.select(train_idxs)
    val_ds = ds.select(test_idxs)

    return DatasetDict({"train": train_ds, "val": val_ds})


def clear_console():
    # Check if running inside Jupyter
    if 'ipykernel' in sys.modules:
        from IPython.display import clear_output
        clear_output(wait=True)
    else:
        os.system('cls' if os.name == 'nt' else 'clear')


def stream_to_bin(iterable_dataset, n_items, suffix="pretrain", test_size=0.001, seed=42, process_fn=None):
    """
    Stream dataset directly to binary files without caching.

    This saves disk space by not storing intermediate data.

    Args:
        iterable_dataset: HuggingFace IterableDataset
        n_items: Total number of items to process
        suffix: Suffix for output files
        test_size: Proportion for validation split
        seed: Random seed
        process_fn: Function to process each example (defaults to process())
    """
    if process_fn is None:
        process_fn = process

    data_directory = os.path.join(os.path.dirname(__file__), "../data")
    data_directory = os.path.abspath(data_directory)
    os.makedirs(data_directory, exist_ok=True)

    dtype = np.uint16

    # Determine split indices
    n_val = max(1, int(n_items * test_size))
    n_train = n_items - n_val

    # Create random indices for train/val split
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    val_indices = set(indices[:n_val].tolist())

    # Open binary files for writing
    train_file = os.path.join(data_directory, f"train_{suffix}.bin")
    val_file = os.path.join(data_directory, f"val_{suffix}.bin")

    print(f"Streaming to {train_file} and {val_file}...")
    print(f"Train examples: {n_train}, Val examples: {n_val}")

    # First pass: collect tokens to write
    train_tokens = []
    val_tokens = []

    for idx, example in enumerate(tqdm(iterable_dataset, total=n_items, desc="Streaming and tokenizing")):
        if idx >= n_items:
            break

        # Process the example
        processed = process_fn(example)
        ids = processed["ids"]

        # Add to appropriate split
        if idx in val_indices:
            val_tokens.extend(ids)
        else:
            train_tokens.extend(ids)

    clear_console()

    # Write train file
    print(f"Writing {len(train_tokens)} tokens to train file...")
    train_arr = np.array(train_tokens, dtype=dtype)
    with open(train_file, 'wb') as f:
        train_arr.tofile(f)

    # Write val file
    print(f"Writing {len(val_tokens)} tokens to val file...")
    val_arr = np.array(val_tokens, dtype=dtype)
    with open(val_file, 'wb') as f:
        val_arr.tofile(f)

    print(f"Done! Train: {len(train_tokens)} tokens, Val: {len(val_tokens)} tokens")
    return train_file, val_file


def stream_to_bin_sft(iterable_dataset, n_items, suffix="sft", test_size=0.1, seed=42):
    """
    Stream SFT dataset directly to binary files without caching.

    This saves disk space by not storing intermediate data.

    Args:
        iterable_dataset: HuggingFace IterableDataset
        n_items: Total number of items to process
        suffix: Suffix for output files
        test_size: Proportion for validation split
        seed: Random seed
    """
    data_directory = os.path.join(os.path.dirname(__file__), "../data")
    data_directory = os.path.abspath(data_directory)
    os.makedirs(data_directory, exist_ok=True)

    dtype = np.uint16

    # Determine split indices
    n_val = max(100, int(n_items * test_size))  # At least 100 val examples
    n_train = n_items - n_val

    # Create random indices for train/val split
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    val_indices = set(indices[:n_val].tolist())

    # Open binary files for writing
    train_token_file = os.path.join(data_directory, f"train_{suffix}.bin")
    train_mask_file = os.path.join(data_directory, f"train_{suffix}_mask.bin")
    val_token_file = os.path.join(data_directory, f"val_{suffix}.bin")
    val_mask_file = os.path.join(data_directory, f"val_{suffix}_mask.bin")

    print(f"Streaming to {train_token_file} and {val_token_file}...")
    print(f"Train examples: {n_train}, Val examples: {n_val}")

    # First pass: collect tokens to write
    train_tokens = []
    train_masks = []
    val_tokens = []
    val_masks = []

    valid_count = 0

    for example in tqdm(iterable_dataset, total=n_items, desc="Streaming and tokenizing"):
        if valid_count >= n_items:
            break

        # Process the example
        processed = process_sft({"Question": example["Question"], "Answer": example["Answer"]})

        # Skip invalid examples
        if processed is None:
            continue

        ids = processed["ids"]
        mask = processed["mask"]

        # Add to appropriate split
        if valid_count in val_indices:
            val_tokens.extend(ids)
            val_masks.extend(mask)
        else:
            train_tokens.extend(ids)
            train_masks.extend(mask)

        valid_count += 1

    clear_console()

    # Write train files
    print(f"Writing {len(train_tokens)} tokens to train files...")
    train_token_arr = np.array(train_tokens, dtype=dtype)
    train_mask_arr = np.array(train_masks, dtype=np.uint8)
    with open(train_token_file, 'wb') as f:
        train_token_arr.tofile(f)
    with open(train_mask_file, 'wb') as f:
        train_mask_arr.tofile(f)

    # Write val files
    print(f"Writing {len(val_tokens)} tokens to val files...")
    val_token_arr = np.array(val_tokens, dtype=dtype)
    val_mask_arr = np.array(val_masks, dtype=np.uint8)
    with open(val_token_file, 'wb') as f:
        val_token_arr.tofile(f)
    with open(val_mask_file, 'wb') as f:
        val_mask_arr.tofile(f)

    print(f"Done! Train: {len(train_tokens)} tokens, Val: {len(val_tokens)} tokens")
    return train_token_file, val_token_file
