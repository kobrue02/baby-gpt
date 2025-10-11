"""
Base classes and utilities for data loaders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datasets import DatasetDict


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_key: str
    subset: Optional[str] = None
    n_items: Optional[int] = None
    test_size: float = 0.001
    seed: int = 42


class BaseDatasetLoader(ABC):
    """Base class for all dataset loaders."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def load_dataset(self) -> DatasetDict:
        """Load and return the dataset."""
        pass

    @abstractmethod
    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Process and tokenize the dataset."""
        pass

    def create_dataset(self, output_suffix: str):
        """Main method to create and save the dataset."""
        from data_loaders.utils import to_bins

        print(f"Loading dataset: {self.config.dataset_key}")
        if self.config.subset:
            print(f"Using subset: {self.config.subset}")
        if self.config.n_items:
            print(f"Loading {self.config.n_items} items")

        dataset = self.load_dataset()
        processed = self.process_dataset(dataset)

        print(f"Saving to binary files with suffix: {output_suffix}")
        to_bins(processed, suffix=output_suffix, is_sft=(output_suffix == "sft"))

        print("Dataset created successfully!")
        return processed
