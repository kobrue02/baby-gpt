"""
Credits to Karpathy's nanoGPT repo for much of this code.
"""

# saves a dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import sys
from datasets import load_dataset, DatasetDict, Dataset
from data.utils import process, to_bins
from tqdm import tqdm



def get_dataset_splits(dataset_key, n_items, test_size=0.001, seed=42):
    print(f"Loading first {n_shards} shards from {dataset_key} dataset...")
    # Load dataset in streaming mode to get first n_shards worth of data
    # Stream the dataset and take only what we need
    ds = load_dataset(dataset_key, split="train", streaming=True)
    # Calculate approximate number of examples to take based on shard count
    # Each shard has roughly similar number of examples
    # We'll take the first portion of the dataset
    ds_list = list(tqdm(ds.take(n_items), total=n_items, desc="Loading examples")) # type: ignore
    ds = Dataset.from_list(ds_list)
    # Remove all columns except 'text'
    ds = ds.remove_columns([col for col in ds.column_names if col != 'text'])
    # Create train/val split
    splits = ds.train_test_split(test_size=0.001, seed=42)
    split_dataset = DatasetDict({
        "train": splits["train"],
        "val": splits["test"]
    })
    return split_dataset

def tokenize(dataset):
    print("Tokenizing dataset for pretraining...")
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="Processing pretraining examples" # type: ignore
    )
    return tokenized
    

def create_pretraining_dataset(n_shards=10, dataset_key="facebook/recycling_the_web"):
    """
    Create and process pretraining dataset by downloading specific parquet shards.

    Args:
        n_shards: Number of parquet shards to download from the dataset
    """
    split_dataset = get_dataset_splits(dataset_key, n_shards, test_size=0.001, seed=42)
    tokenized = tokenize(split_dataset)
    to_bins(tokenized, suffix="pretrain")


if __name__ == '__main__':
    n_shards = int(sys.argv[1] if len(sys.argv) > 1 else 10)
    create_pretraining_dataset(n_shards)
    print("Done. Now you can run train.py to train a model on the dataset.")