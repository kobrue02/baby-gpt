"""
Credits to Karpathy's nanoGPT repo for much of this code.
"""

# saves a dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import sys
import numpy as np
import tiktoken

from tqdm import tqdm
from string import punctuation
from data.load_datasets import load_finepdfs

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

# we want to tokenize the dataset. first define the encoding function (gpt2 bpe)

def clean_text(text):
    text = text.replace('\n', ' ') # make all newlines spaces
    text = ' '.join(text.split()) # make all whitespace single spaces
    text = text.strip() # remove leading/trailing whitespace
    text = text.lower() # make all lowercase
    text = text.translate(
        str.maketrans('', '', ''.join(
            [c for c in punctuation if c not in ["'", ".", ",", "!", "?"]])
    )) # remove most punctuation except some common ones
    return text

def process(example):
    """
    Take a text example and encode to ids using tiktoken GPT-2 BPE.
    """
    text = example['text']
    text = clean_text(text)

    # encode
    ids = enc.encode_ordinary(text) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

def memmap(split, dset, dtype):
    arr_len = np.sum(dset['len'], dtype=np.uint64) # type: ignore
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,)) # type: ignore
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids']) # type: ignore
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    assert idx == arr_len
    return arr

def to_bins(tokenized):
    """
    Take a tokenized dataset and save to binary files.
    """
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    for split, dset in tokenized.items(): # type: ignore
        arr = memmap(split, dset, dtype)
        arr.flush()

if __name__ == '__main__':
    n_rows = int(sys.argv[1] if len(sys.argv) > 1 else 1000000)
    split_dataset = load_finepdfs(n_rows)
    tokenized = split_dataset.map(process, remove_columns=['text'])
    to_bins(tokenized)
    print("Done. Now you can run train.py to train a model on the dataset.")

    