"""
Registry for dataset loaders with predefined dataset configurations.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from data_loaders.base import DatasetConfig
from data_loaders.pretraining import PretrainingLoader
from data_loaders.supervised import SFTLoader
from data_loaders.scraper import ScrapedDataLoader


@dataclass
class DatasetInfo:
    """
    Information about a dataset.
    Fields:
    - dataset_key: HuggingFace dataset key
    - subset: Optional subset name
    - description: Short description of the dataset
    """
    dataset_key: str
    subset: Optional[str] = None
    description: str = ""


# preset configurations for pretraining datasets
PRETRAINING_DATASETS: Dict[str, DatasetInfo] = {
    "fineweb": DatasetInfo(
        dataset_key="HuggingFaceFW/fineweb",
        subset="default",
        description="High-quality web text from Common Crawl",
    ),
    "fineweb-edu": DatasetInfo(
        dataset_key="HuggingFaceFW/fineweb-edu",
        subset="default",
        description="Educational subset of FineWeb",
    ),
    "smollm-corpus": DatasetInfo(
        dataset_key="HuggingFaceTB/smollm-corpus",
        subset="cosmopedia-v2",
        description="SmolLM pretraining corpus",
    ),
    "cosmopedia": DatasetInfo(
        dataset_key="HuggingFaceTB/smollm-corpus",
        subset="cosmopedia-v2",
        description="Synthetic textbooks and educational content",
    ),
    "python-edu": DatasetInfo(
        dataset_key="HuggingFaceTB/smollm-corpus",
        subset="python-edu",
        description="Educational Python code",
    ),
    "web-samples": DatasetInfo(
        dataset_key="HuggingFaceTB/smollm-corpus",
        subset="fineweb-edu-dedup",
        description="Deduplicated web samples",
    ),
    "custom-scrape": DatasetInfo(
        dataset_key="scraped-dataset",
        description="Custom dataset scraped from predefined URLs",
    ),
}

SFT_DATASETS: Dict[str, DatasetInfo] = {
    "general-knowledge": DatasetInfo(
        dataset_key="MuskumPillerum/General-Knowledge",
        description="General knowledge Q&A pairs",
    ),
    "prime-stack-exchange": DatasetInfo(
        dataset_key="PrimeIntellect/stackexchange-question-answering",
        description="Stack Exchange Q&A pairs by PrimeIntellect",
    ),
    "prime-math": DatasetInfo(
        dataset_key="PrimeIntellect/INTELLECT-2-only-math-filtered-2k",
        description="Mathematics Q&A pairs by PrimeIntellect",
    ),
}


def get_pretraining_loader(
    dataset_name: str,
    n_items: Optional[int] = None,
    dataset_key: Optional[str] = None,
    subset: Optional[str] = None,
    test_size: float = 0.001,
    seed: int = 42,
    streaming: bool = True,
) -> PretrainingLoader | ScrapedDataLoader:
    """
    Get a pretraining data loader.

    Args:
        dataset_name: Name of the dataset from the registry (e.g., 'fineweb')
        n_items: Number of items to load (None for full dataset)
        dataset_key: Override dataset key (e.g., 'HuggingFaceFW/fineweb')
        subset: Override subset name
        test_size: Proportion for validation split
        seed: Random seed
        streaming: If True, stream directly to bins without caching (saves disk space)

    Returns:
        PretrainingLoader instance
    """
    # If custom dataset_key provided, use it
    if dataset_key:
        config = DatasetConfig(
            dataset_key=dataset_key,
            subset=subset,
            n_items=n_items,
            test_size=test_size,
            seed=seed,
        )
    # Otherwise, look up in registry
    elif dataset_name in PRETRAINING_DATASETS:
        dataset_info = PRETRAINING_DATASETS[dataset_name]
        config = DatasetConfig(
            dataset_key=dataset_info.dataset_key,
            subset=subset or dataset_info.subset,
            n_items=n_items,
            test_size=test_size,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(PRETRAINING_DATASETS.keys())}"
        )

    if config.dataset_key == "scraped-dataset":
        config.test_size = 0.1  # override for scraped dataset
        return ScrapedDataLoader(config)
    else:
        return PretrainingLoader(config, streaming=streaming)


def get_sft_loader(
    dataset_name: str,
    n_items: Optional[int] = None,
    dataset_key: Optional[str] = None,
    subset: Optional[str] = None,
    test_size: float = 0.1,
    seed: int = 42,
    streaming: bool = True,
) -> SFTLoader:
    """
    Get an SFT data loader.

    Args:
        dataset_name: Name of the dataset from the registry
        n_items: Number of items to load
        dataset_key: Override dataset key
        subset: Override subset name
        test_size: Proportion for validation split
        seed: Random seed
        streaming: If True, stream directly to bins without caching (saves disk space)

    Returns:
        SFTLoader instance
    """
    # if custom dataset_key provided, use it
    if dataset_key:
        config = DatasetConfig(
            dataset_key=dataset_key,
            subset=subset,
            n_items=n_items,
            test_size=test_size,
            seed=seed,
        )
    # else it should a preset, we look up in registry
    elif dataset_name in SFT_DATASETS:
        dataset_info = SFT_DATASETS[dataset_name]
        config = DatasetConfig(
            dataset_key=dataset_info.dataset_key,
            subset=subset or dataset_info.subset,
            n_items=n_items,
            test_size=test_size,
            seed=seed,
        )
    else: # or it is unsupported
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(SFT_DATASETS.keys())}"
        )

    return SFTLoader(config, streaming=streaming)


def list_datasets(dataset_type: str = "pretraining") -> None:
    """Print available datasets."""
    if dataset_type == "pretraining":
        datasets = PRETRAINING_DATASETS
    elif dataset_type == "sft":
        datasets = SFT_DATASETS
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"\nAvailable {dataset_type} datasets:")
    print("=" * 60)
    for name, info in datasets.items():
        print(f"\n{name}")
        print(f"  Dataset: {info.dataset_key}")
        if info.subset:
            print(f"  Subset: {info.subset}")
        if info.description:
            print(f"  Description: {info.description}")
