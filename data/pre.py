"""
Credits to Karpathy's nanoGPT repo for much of this code.
"""

# saves a dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import sys
from data.load_datasets import load_finepdfs
from data.utils import process, to_bins


def create_pretraining_dataset(n_rows=1000000):
    """
    Create and process pretraining dataset from finepdfs.
    """
    print(f"Loading {n_rows} rows from finepdfs dataset...")
    split_dataset = load_finepdfs(n_rows)

    print("Tokenizing dataset for pretraining...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Processing pretraining examples" # type: ignore
    )

    print("Saving tokenized dataset to binary files...")
    to_bins(tokenized, suffix="pretrain")

    return tokenized


if __name__ == '__main__':
    n_rows = int(sys.argv[1] if len(sys.argv) > 1 else 1000000)
    create_pretraining_dataset(n_rows)
    print("Done. Now you can run train.py to train a model on the dataset.")