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
        columns_to_keep = []
        if (
            "Question" in ds.column_names and "Answer" in ds.column_names
        ):
            # Keep only Question and Answer columns (adjust for other datasets)
            columns_to_keep = ["Question", "Answer"]
            columns_to_remove = [
                col for col in ds.column_names if col not in columns_to_keep
            ]
            ds = ds.remove_columns(columns_to_remove)
            ds = ds.rename_column("Question", "instruction")
            ds = ds.rename_column("Answer", "output")
            # Add empty input field for consistency
            ds = ds.map(lambda _: {"input": ""}, batched=False)
        elif ( # prime intellect stack exchange
            "prompt" in ds.column_names and "gold_standard_solution" in ds.column_names
        ):
            columns_to_keep = ["prompt", "gold_standard_solution"]
            columns_to_remove = [
                col for col in ds.column_names if col not in columns_to_keep
            ]
            ds = ds.remove_columns(columns_to_remove)
            # rename to standard names
            ds = ds.rename_column("prompt", "instruction")
            ds = ds.rename_column("gold_standard_solution", "output")
            # Add empty input field for consistency
            ds = ds.map(lambda _: {"input": ""}, batched=False)
        elif (
            "instruction" in ds.column_names and "context" in ds.column_names and "response" in ds.column_names
        ):
            columns_to_keep = ["instruction", "context", "response"]
            columns_to_remove = [
                col for col in ds.column_names if col not in columns_to_keep
            ]
            ds = ds.remove_columns(columns_to_remove)
            # rename to standard names
            ds = ds.rename_column("context", "input")
            ds = ds.rename_column("response", "output")

        # create train/val split
        splits = ds.train_test_split(
            test_size=self.config.test_size, seed=self.config.seed
        )
        split_dataset = DatasetDict({"train": splits["train"], "val": splits["test"]})

        return split_dataset

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Process and tokenize the SFT dataset."""
        print("Filtering out invalid examples...")

        def is_valid(example):
            q = example["instruction"]
            a = example["output"]
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

        # we should have enough validation examples
        if len(dataset["val"]) < self.min_val_examples:
            raise ValueError(
                f"Not enough validation examples: got {len(dataset['val'])}, "
                f"need at least {self.min_val_examples}. Try loading more rows or increasing test_size."
            )

        print("Tokenizing dataset with SFT format (including masks)...")
        tokenized = dataset.map(
            process_sft,
            remove_columns=["instruction", "input", "output"],
            desc="Processing SFT examples",
            batched=True,
            batch_size=1000,
            num_proc=1,
        )

        # Filter out sequences that are too short (minimum 10 tokens)
        min_length = 10
        print(f"Filtering out sequences shorter than {min_length} tokens...")

        def has_min_length(example):
            return example["len"] >= min_length

        tokenized = tokenized.filter(has_min_length, desc="Filtering short sequences")
        print(
            f"After length filtering: {len(tokenized['train'])} train and {len(tokenized['val'])} val examples"
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

        if self.streaming:
            print("Using streaming mode (saves disk space by not caching dataset)")

            # streaming dataset
            ds = load_dataset(
                self.config.dataset_key,
                name=self.config.subset,
                split="train",
                streaming=True,
            )

            stream_to_bin_sft(
                ds,
                n_items=n_items,
                suffix=output_suffix,
                test_size=self.config.test_size,
                seed=self.config.seed,
            )

            print("Dataset created successfully!")
        else:
            # loads full dataset into memory/cache
            print("Using non-streaming mode (dataset will be cached to disk)")

            dataset = self.load_dataset()
            processed = self.process_dataset(dataset)

            print(f"Saving to binary files with suffix: {output_suffix}")
            to_bins(processed, suffix=output_suffix, is_sft=True)

            print("Dataset created successfully!")

        return None
