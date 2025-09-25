import sys
from data.load_datasets import load_general_knowledge
from data.utils import to_bins, process_sft


def create_sft_dataset(n_rows=1000000):
    """
    Create and process SFT dataset from general knowledge Q&A pairs.
    """
    print(f"Loading {n_rows} rows from general knowledge dataset...")
    split_dataset = load_general_knowledge()

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