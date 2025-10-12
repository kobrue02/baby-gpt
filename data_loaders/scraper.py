"""
Pretraining dataset loader.
"""

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from typing import List, Callable, Optional

from data_loaders.base import BaseDatasetLoader, DatasetConfig
from data_loaders.utils import process, clear_console, split_dataset_in_memory, stream_to_bin

URL_LIST = [
    "https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf",
    "https://i.4pcdn.org/tg/1439541465764.pdf", # dune
    "https://dn790002.ca.archive.org/0/items/OneHundredYearsOfSolitude_201710/One_Hundred_Years_of_Solitude.pdf", # one hundred years of solitude
    "https://www.liberalstudies.ca/wp-content/uploads/2014/11/Economics-in-One-Lesson_2.pdf", # economics in one lesson
    "https://avalonlibrary.net/ebooks/David%20Deutsch%20-%20The%20Beginning%20of%20Infinity%20-%20Explanations%20that%20Transform%20the%20World.pdf", # the beginning of infinity
    "https://ia600702.us.archive.org/33/items/poor-charlies-almanack-the-wit-and-wisdom-of-charles-t.-munger-pdfdrive/Poor%20Charlie%E2%80%99s%20Almanack_%20The%20Wit%20and%20Wisdom%20of%20Charles%20T.%20Munger%20%28%20PDFDrive%20%29.pdf", # poor charlie's almanack
    "https://morfene.com/021.pdf", # zero to one
    "https://csbible.com/wp-content/uploads/2018/03/CSB_Pew_Bible_2nd_Printing.pdf", # christian standard bible
    "https://www.eriesd.org/site/handlers/filedownload.ashx?moduleinstanceid=35845&dataid=53662&FileName=The%20Hero%20with%20a%20Thousand%20Faces.pdf", # the hero with a thousand faces
    "https://ia800306.us.archive.org/31/items/durant-will-the-lessons-of-history_202012/Durant%20Will%20-%20The%20Lessons%20of%20History.pdf", # the lessons of history
    "https://irp.cdn-website.com/6b820530/files/uploaded/Ayn%20Rand-%20Atlas%20Shrugged.pdf", # atlas shrugged
]


class ScrapedDataLoader(BaseDatasetLoader):
    """Loader for web-scraped datasets."""

    def __init__(self, config: DatasetConfig):
        """
        Initialize the loader.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)
        self.url_list = URL_LIST

    def _get_content_from_pdf_url(self, url: str) -> str:
        import requests
        from io import BytesIO
        from PyPDF2 import PdfReader

        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _load_urls(self) -> str:
        full_text = ""
        for url in tqdm(self.url_list):
            try:
                text = self._get_content_from_pdf_url(url)
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        return full_text
        
    def load_dataset(self) -> DatasetDict:
        print("Loading and scraping data from URLs...")
        full_text = self._load_urls()
        # Split into chunks by paragraphs
        chunks = [text for text in full_text.split('\n\n') if text.strip()]
        ds = Dataset.from_dict({'text': chunks})

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

    def create_dataset(self, output_suffix: str):
        """
        Main method to create and save the dataset.
        """
        from data_loaders.utils import to_bins

        print("Loading scraped dataset from URLs")
        dataset = self.load_dataset()
        processed = self.process_dataset(dataset)

        print(f"Saving to binary files with suffix: {output_suffix}")
        to_bins(processed, suffix=output_suffix, is_sft=False)

        print("Dataset created successfully!")
        return None