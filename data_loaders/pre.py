"""
Credits to Karpathy's nanoGPT repo for much of this code.
"""

# saves a dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import sys
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset, IterableDatasetDict
from data_loaders.utils import process, to_bins, clear_console
from tqdm import tqdm


def get_dataset_splits(dataset_key, subset=None, n_items=None, test_size=0.001, seed=42):
    if n_items:
        print(f"Loading first {n_items} shards from {dataset_key} dataset...")
    else:
        print(f"Loading full {dataset_key} dataset...")
    if subset:
        print(f"Using subset: {subset}")
    # Load dataset in streaming mode to get first n_shards worth of data
    # Stream the dataset and take only what we need
    ds: Dataset | DatasetDict | IterableDataset | IterableDatasetDict
    if n_items:
        ds = load_dataset(dataset_key, name=subset, split="train", streaming=True)
        ds_list = list(tqdm(ds.take(n_items), total=n_items, desc="Loading examples"))  # type: ignore
        ds = Dataset.from_list(ds_list)
    else:
        ds = load_dataset(dataset_key, name=subset, split="train")

    # Remove all columns except 'text'
    ds = ds.remove_columns([col for col in ds.column_names if col != "text"]) # type: ignore
    clear_console() # its cluttered with tqdm bars otherwise
    # Create train/val split
    splits = ds.train_test_split(test_size=0.001, seed=42) # type: ignore
    split_dataset = DatasetDict({"train": splits["train"], "val": splits["test"]})
    
    return split_dataset


def tokenize(dataset):
    print("Tokenizing dataset for pretraining...")
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="Processing pretraining examples",  # type: ignore
    )
    return tokenized


def create_pretraining_dataset(n_items=None, dataset_key="facebook/recycling_the_web", subset=None):
    """
    Create and process pretraining dataset by downloading specific parquet shards.

    Args:
        n_shards: Number of parquet shards to download from the dataset
    """
    split_dataset = get_dataset_splits(dataset_key, subset=subset, n_items=n_items, test_size=0.001, seed=42)
    tokenized = tokenize(split_dataset)
    to_bins(tokenized, suffix="pretrain")


if __name__ == "__main__":
    n_items = int(sys.argv[1]) if len(sys.argv) > 1 else None
    create_pretraining_dataset(n_items)
    print(
        "Done. Now you can run `python -m training.pretraining` to train a model on the dataset."
    )
