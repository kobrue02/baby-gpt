from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CurriculumStage:
    """
    Represents a single stage in curriculum learning.

    Attributes:
        name: name of the stage (e.g., 'warmup', 'foundation', 'pretrain', 'sft')
        dataset_key: dataset identifier from HuggingFace or 'local'
        block_size: context length for this stage
        n_epochs: number of epochs to train for this stage
        dataset_suffix: suffix for the binary files (e.g., 'pretrain', 'sft')
    """
    name: str
    dataset_key: str
    block_size: int
    n_epochs: int
    dataset_suffix: str = "pretrain"
    n_items: Optional[int] = None  # Optional: limit number of examples


@dataclass
class Curriculum:
    """
    Defines a full curriculum for pretraining.

    Each stage uses progressively:
    - More complex data
    - Longer context (block_size)
    - More training iterations (epochs)
    """
    stages: List[CurriculumStage]

    @classmethod
    def create_default(cls) -> "Curriculum":
        """
        Create a default 3-stage curriculum.

        Stage 1 (warmup): TinyStories - simple children's stories
        Stage 2 (foundation): Local .txt files from data_loaders/files/
        Stage 3 (pretrain): Web-scraped content (URLs, PDFs, blogs from db.py)
        """
        return cls(stages=[
            CurriculumStage(
                name="warmup",
                dataset_key="roneneldan/TinyStories",
                block_size=128,
                n_epochs=1,
                dataset_suffix="pretrain",
                n_items=1000000  # Small warmup dataset
            ),
            CurriculumStage(
                name="foundation",
                dataset_key="local-files",  # Local .txt files
                block_size=256,
                n_epochs=5,
                dataset_suffix="pretrain",
            ),
            CurriculumStage(
                name="pretrain",
                dataset_key="custom-scrape",  # Web-scraped content
                block_size=512,
                n_epochs=13,
                dataset_suffix="pretrain",
            ),
        ])

    @classmethod
    def create_simple(cls) -> "Curriculum":
        """Create a simple 2-stage curriculum for testing."""
        return cls(stages=[
            CurriculumStage(
                name="warmup",
                dataset_key="roneneldan/TinyStories",
                block_size=128,
                n_epochs=1,
                dataset_suffix="pretrain",
                n_items=10000
            ),
            CurriculumStage(
                name="pretrain",
                dataset_key="custom-scrape",  # Web-scraped content
                block_size=256,
                n_epochs=2,
                dataset_suffix="pretrain",
            ),
        ])
