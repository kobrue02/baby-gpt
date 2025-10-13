"""
Loader for local text files (for foundation stage in curriculum).
"""

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from typing import Optional

from data_loaders.base import BaseDatasetLoader, DatasetConfig
from data_loaders.utils import process, split_dataset_in_memory
from data_loaders.db import LOCAL_TXT_FILES


class LocalFilesLoader(BaseDatasetLoader):
    """Loader for local .txt files from data_loaders/files/."""

    def __init__(self, config: DatasetConfig):
        """
        Initialize the loader.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)
        self.local_txt_files = LOCAL_TXT_FILES

    def _load_local_txt_file(self, file_path: str) -> str:
        """Load text content from a local .txt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_dataset(self) -> DatasetDict:
        """Load all local .txt files and create a dataset."""
        print(f"Loading {len(self.local_txt_files)} local text files...")

        full_text = ""
        for file_path in tqdm(self.local_txt_files, desc="Loading local files"):
            try:
                text = self._load_local_txt_file(file_path)
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {file_path}: {e}")

        # Split into chunks by paragraphs
        chunks = [text for text in full_text.split('\n\n') if text.strip()]

        if not chunks:
            raise ValueError("No text chunks found in local files")

        ds = Dataset.from_dict({'text': chunks})

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
            desc="Processing local file examples",  # type: ignore
        )
        return tokenized
