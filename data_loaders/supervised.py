"""
Supervised Fine-Tuning (SFT) dataset loader.
"""

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from data_loaders.base import BaseDatasetLoader, DatasetConfig
from data_loaders.utils import process_sft, stream_to_bin_sft


class SFTLoader(BaseDatasetLoader):
    """Loader for supervised fine-tuning datasets."""

    def __init__(self, config: DatasetConfig, min_val_examples: int = 100, streaming: bool = True):
        """
        Initialize the loader.

        Args:
            config: Dataset configuration
            min_val_examples: Minimum number of validation examples required
            streaming: If True, stream directly to bins without caching (saves disk space)
        """
        super().__init__(config)
        self.min_val_examples = min_val_examples
        self.streaming = streaming

    def load_dataset(self) -> DatasetDict:
        """Load general knowledge Q&A dataset."""
        # Stream the dataset
        ds = load_dataset(
            self.config.dataset_key,
            name=self.config.subset,
            split="train",
            streaming=True,
        )

        # Take first n_items examples
        n_items = self.config.n_items or 10000
        ds_list = list(tqdm(ds.take(n_items), total=n_items, desc="Loading examples"))  # type: ignore
        ds = Dataset.from_list(ds_list)

        # Keep only Question and Answer columns (adjust for other datasets)
        columns_to_keep = ["Question", "Answer"]
        columns_to_remove = [
            col for col in ds.column_names if col not in columns_to_keep
        ]
        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)

        # Create train/val split
        splits = ds.train_test_split(
            test_size=self.config.test_size, seed=self.config.seed
        )
        split_dataset = DatasetDict({"train": splits["train"], "val": splits["test"]})

        return split_dataset

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Process and tokenize the SFT dataset."""
        print("Filtering out invalid examples...")

        def is_valid(example):
            q = example["Question"]
            a = example["Answer"]
            return (
                q is not None
                and a is not None
                and isinstance(q, str)
                and isinstance(a, str)
                and len(q.strip()) > 0
                and len(a.strip()) > 0
            )

        dataset = dataset.filter(is_valid, desc="Filtering invalid examples")
        print(
            f"Kept {len(dataset['train'])} train and {len(dataset['val'])} val examples"
        )

        # Ensure we have enough validation examples
        if len(dataset["val"]) < self.min_val_examples:
            raise ValueError(
                f"Not enough validation examples: got {len(dataset['val'])}, "
                f"need at least {self.min_val_examples}. Try loading more rows or increasing test_size."
            )

        print("Tokenizing dataset with SFT format (including masks)...")
        tokenized = dataset.map(
            process_sft,
            remove_columns=["Question", "Answer"],
            desc="Processing SFT examples",
            batched=True,
            batch_size=1000,
            num_proc=1,
        )

        return tokenized

    def create_dataset(self, output_suffix: str):
        """
        Main method to create and save the dataset.

        Supports streaming mode to save disk space.
        """
        from data_loaders.utils import to_bins

        print(f"Loading dataset: {self.config.dataset_key}")
        if self.config.subset:
            print(f"Using subset: {self.config.subset}")

        n_items = self.config.n_items or 10000
        print(f"Loading {n_items} items")

        # Use streaming mode if enabled
        if self.streaming:
            print("Using streaming mode (saves disk space by not caching dataset)")

            # Load as streaming dataset
            ds = load_dataset(
                self.config.dataset_key,
                name=self.config.subset,
                split="train",
                streaming=True,
            )

            # Stream directly to bins
            stream_to_bin_sft(
                ds,
                n_items=n_items,
                suffix=output_suffix,
                test_size=self.config.test_size,
                seed=self.config.seed,
            )

            print("Dataset created successfully!")
        else:
            # Use the original method (loads full dataset into memory/cache)
            print("Using non-streaming mode (dataset will be cached to disk)")

            dataset = self.load_dataset()
            processed = self.process_dataset(dataset)

            print(f"Saving to binary files with suffix: {output_suffix}")
            to_bins(processed, suffix=output_suffix, is_sft=True)

            print("Dataset created successfully!")

        return None
