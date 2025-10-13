"""
Pretraining dataset loader.
"""

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from typing import Optional

from data_loaders.base import BaseDatasetLoader, DatasetConfig
from data_loaders.utils import process, clear_console, split_dataset_in_memory, stream_to_bin


class PretrainingLoader(BaseDatasetLoader):
    """Loader for pretraining datasets."""

    def __init__(self, config: DatasetConfig, streaming: bool = True):
        """
        Initialize the loader.

        Args:
            config: Dataset configuration
            streaming: If True, stream directly to bins without caching (saves disk space)
        """
        super().__init__(config)
        self.streaming = streaming

    def load_dataset(self) -> DatasetDict:
        """Load dataset from HuggingFace with optional streaming."""
        if self.config.n_items:
            # Stream the dataset and take only what we need
            ds = load_dataset(
                self.config.dataset_key,
                name=self.config.subset,
                split="train",
                streaming=True,
            )
            # Shuffle to read from multiple files in parallel
            ds = ds.shuffle(seed=self.config.seed, buffer_size=10000) # type: ignore
            ds_list = list(
                tqdm(ds.take(self.config.n_items), total=self.config.n_items, desc="Loading examples")  # type: ignore
            )
            ds = Dataset.from_list(ds_list)
        else:
            # Load the full dataset
            ds = load_dataset(
                self.config.dataset_key,
                name=self.config.subset,
                split="train",
            )

        # Remove all columns except 'text'
        ds = ds.remove_columns([col for col in ds.column_names if col != "text"])  # type: ignore

        clear_console()  # Clear tqdm bars

        # Create train/val split
        splits = split_dataset_in_memory(
            ds, test_size=self.config.test_size, seed=self.config.seed
        )
        return splits

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenize the dataset for pretraining."""
        print("Tokenizing dataset for pretraining...")
        tokenized = dataset.map(
            process,
            remove_columns=["text"],
            desc="Processing pretraining examples",  # type: ignore
        )
        return tokenized

    def create_dataset(self, output_suffix: str, stage: Optional[str] = None):
        """
        Main method to create and save the dataset.

        Supports streaming mode to save disk space.

        Args:
            output_suffix: suffix for the dataset (e.g., 'pretrain')
            stage: optional curriculum stage name (e.g., 'warmup', 'foundation')
        """
        from data_loaders.utils import to_bins

        print(f"Loading dataset: {self.config.dataset_key}")
        if self.config.subset:
            print(f"Using subset: {self.config.subset}")
        if self.config.n_items:
            print(f"Loading {self.config.n_items} items")
        else:
            print("Loading full dataset")
        if stage:
            print(f"Curriculum stage: {stage}")

        # Use streaming mode by default unless explicitly disabled
        if self.streaming:
            print("Using streaming mode (saves disk space by not caching dataset)")

            # Load as streaming dataset
            ds = load_dataset(
                self.config.dataset_key,
                name=self.config.subset,
                split="train",
                streaming=True,
            )

            # Enable faster streaming by using multiple shards in parallel
            # This prevents blocking on single file downloads
            ds = ds.shuffle(seed=self.config.seed, buffer_size=10000) # type: ignore

            # Stream directly to bins
            stream_to_bin(
                ds,
                n_items=self.config.n_items,
                suffix=output_suffix,
                test_size=self.config.test_size,
                seed=self.config.seed,
                stage=stage,
            )

            print("Dataset created successfully!")
        else:
            # Use the original method (loads full dataset into memory/cache)
            print("Using non-streaming mode (dataset will be cached to disk)")

            dataset = self.load_dataset()
            processed = self.process_dataset(dataset)

            print(f"Saving to binary files with suffix: {output_suffix}" + (f" and stage: {stage}" if stage else ""))
            to_bins(processed, suffix=output_suffix, is_sft=(output_suffix == "sft"), stage=stage)

            print("Dataset created successfully!")

        return None
