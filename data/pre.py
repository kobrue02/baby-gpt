"""
Credits to Karpathy's nanoGPT repo for much of this code.
"""

# saves a dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import sys
from datasets import load_dataset, DatasetDict
from data.utils import process, to_bins


def create_pretraining_dataset(n_shards=10):
    """
    Create and process pretraining dataset from EssentialAI by downloading specific parquet shards.

    Args:
        n_shards: Number of parquet shards to download from the dataset
    """
    dataset_key = "EssentialAI/eai-taxonomy-stem-w-dclm"
    print(f"Loading first {n_shards} shards from {dataset_key} dataset...")

    # Load only the first n_shards by using data_files parameter
    # The dataset uses parquet files named like "data/train-00000-of-09987.parquet"
    shard_files = [f"data/train-{i:05d}-of-09987.parquet" for i in range(n_shards)]

    ds = load_dataset(dataset_key, data_files=shard_files, split="train")

    # Remove all columns except 'text'
    ds = ds.remove_columns([col for col in ds.column_names if col != 'text'])

    # Create train/val split
    splits = ds.train_test_split(test_size=0.001, seed=42)
    split_dataset = DatasetDict({
        "train": splits["train"],
        "val": splits["test"]
    })

    print(split_dataset)

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
    n_shards = int(sys.argv[1] if len(sys.argv) > 1 else 10)
    create_pretraining_dataset(n_shards)
    print("Done. Now you can run train.py to train a model on the dataset.")