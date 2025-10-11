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
