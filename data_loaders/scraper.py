"""
Pretraining dataset loader.
"""

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from typing import List, Callable, Optional
from io import BytesIO
from PyPDF2 import PdfReader

from data_loaders.base import BaseDatasetLoader, DatasetConfig
from data_loaders.utils import process, split_dataset_in_memory
from data_loaders.db import URL_LIST, BLOG_POSTS, ILYA_RECS, MATH_BOOKS

import requests
import trafilatura


headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "DNT": "1",  # Do Not Track
}


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
        self.blog_posts = BLOG_POSTS
        self.ilya_recs = ILYA_RECS
        self.math_books = MATH_BOOKS
        self.session = requests.Session()
        self.session.headers.update(headers)

    def _get_content_from_pdf_url(self, url: str) -> str:
        """ Extract text content from a PDF URL """
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _get_content_from_text_url(self, url: str) -> str:
        """ Extract text content from a plain text URL """
        downloaded = trafilatura.fetch_url(url)
        text = ""
        # Extract the main content
        if downloaded:
            text = trafilatura.extract(downloaded)
        return text or ""
    
    def _load_urls(self) -> str:
        full_text = ""
        for url in tqdm(self.url_list):
            try:
                text = self._get_content_from_pdf_url(url)
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        for url in tqdm(self.blog_posts):
            try:
                text = self._get_content_from_text_url(url)
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        for url in tqdm(self.ilya_recs):
            try:
                text = self._get_content_from_pdf_url(url)
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        for url in tqdm(self.math_books):
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

        # clear_console()  # Clear tqdm bars

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