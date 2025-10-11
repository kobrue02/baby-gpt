"""
Data loaders for pretraining and supervised fine-tuning.

This module provides a clean, extensible interface for loading and processing datasets.
"""

from data_loaders.base import BaseDatasetLoader, DatasetConfig
from data_loaders.pretraining import PretrainingLoader
from data_loaders.supervised import SFTLoader
from data_loaders.registry import (
    get_pretraining_loader,
    get_sft_loader,
    list_datasets,
    PRETRAINING_DATASETS,
    SFT_DATASETS,
)

__all__ = [
    "BaseDatasetLoader",
    "DatasetConfig",
    "PretrainingLoader",
    "SFTLoader",
    "get_pretraining_loader",
    "get_sft_loader",
    "list_datasets",
    "PRETRAINING_DATASETS",
    "SFT_DATASETS",
]
