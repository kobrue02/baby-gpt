import sys
from datasets import load_dataset, DatasetDict, Dataset
from data.utils import to_bins, process_sft
from tqdm import tqdm

def load_general_knowledge(n_rows=1000000, test_size=0.001, seed=42):
    """
    Load the General-Knowledge dataset and return train/val splits.

    Args:
        n_rows: Number of rows to load from the dataset
        test_size: Proportion of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with 'train' and 'val' splits containing 'Question' and 'Answer' columns
    """
    ds_key = "MuskumPillerum/General-Knowledge"

    # Load dataset in streaming mode
    ds = load_dataset(ds_key, split="train", streaming=True)

    # Take first n_rows examples
    ds_list = list(tqdm(ds.take(n_rows), total=n_rows, desc="Loading examples")) # type: ignore
    ds = Dataset.from_list(ds_list)

    # Keep only Question and Answer columns
    columns_to_keep = ['Question', 'Answer']
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    if columns_to_remove:
        ds = ds.remove_columns(columns_to_remove)

    # Create train/val split
    splits = ds.train_test_split(test_size=test_size, seed=seed)
    split_dataset = DatasetDict({
        "train": splits["train"],
        "val": splits["test"]
    })

    return split_dataset


def create_sft_dataset(n_rows=1000000):
    """
    Create and process SFT dataset from general knowledge Q&A pairs.
    """
    print(f"Loading {n_rows} rows from general knowledge dataset...")
    split_dataset = load_general_knowledge(n_rows=n_rows)

    print("Tokenizing dataset with SFT format (including masks)...")
    tokenized = split_dataset.map(
        process_sft,
        remove_columns=['Question', 'Answer'],
        desc="Processing SFT examples",
        batched=True,  # Use batched processing
        batch_size=1000,
        num_proc=1
    )

    print("Saving tokenized dataset to binary files...")
    to_bins(tokenized, suffix="sft", is_sft=True)

    return tokenized


if __name__ == '__main__':
    n_rows = int(sys.argv[1] if len(sys.argv) > 1 else 1000000)
    create_sft_dataset(n_rows)
    print("Done. Now you can fine tune a model on the dataset using train_sft.py.")