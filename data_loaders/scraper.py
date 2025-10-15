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
from data_loaders.db import URL_LIST, BLOG_POSTS, ILYA_RECS, MATH_BOOKS, LOCAL_TXT_FILES

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

def clean_metadata_from_text(text: str) -> str:
    """ Remove common metadata lines from text """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line_strip = line.strip()
        if (
            line_strip.startswith("http")
            or line_strip.startswith("www.")
            or line_strip.lower().startswith("published by")
            or line_strip.lower().startswith("published in")
            or "copyright" in line_strip.lower()
            or "all rights reserved" in line_strip.lower()
            or "terms of service" in line_strip.lower()
            or "privacy policy" in line_strip.lower()
            or "press preface" in line_strip.lower()
            or "table of contents" in line_strip.lower()
            or "acknowledgments" in line_strip.lower()
            or len(line_strip) < 10  # remove very short lines
            or line_strip.isupper()  # remove lines that are all uppercase
            or line_strip.isdigit()  # remove lines that are just numbers
            or line_strip.lower().startswith("page ") # remove page numbers
            or line_strip.lower().startswith("chapter ") # remove chapter headings
            or line_strip.lower().startswith("section ") # remove section headings
            or line_strip.count(' ') < 2  # remove lines with less than 2 spaces (likely not meaningful)
            or not any(c.isalnum() for c in line_strip)  # remove lines without alphanumeric characters
            or line_strip in ["\n", "", "\r"]  # remove empty lines
        ):
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines).replace("  ", " ").strip()


class ScrapedDataLoader(BaseDatasetLoader):
    """Loader for web-scraped datasets."""

    def __init__(self, config: DatasetConfig, include_local_files: bool = False):
        """
        Initialize the loader.

        Args:
            config: Dataset configuration
            include_local_files: Whether to include local .txt files (for foundation stage)
        """
        super().__init__(config)
        # sources of data
        self.url_list = URL_LIST
        self.blog_posts = BLOG_POSTS
        self.ilya_recs = ILYA_RECS
        self.math_books = MATH_BOOKS
        self.local_txt_files = LOCAL_TXT_FILES if include_local_files else []
        # requests session for connection pooling
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
                text += page.extract_text() + "\n\n"
        return clean_metadata_from_text(text)
    
    def _load_local_txt_file(self, file_path: str) -> str:
        """ Load text content from a local .txt file """
        with open(file_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        return clean_metadata_from_text(txt)
    
    def _get_content_from_text_url(self, url: str) -> str:
        """ Extract text content from a plain text URL """
        downloaded = trafilatura.fetch_url(url)
        text = ""
        # Extract the main content
        if downloaded:
            text += str(
                trafilatura.extract(downloaded, include_comments=False, include_tables=False, target_language='en')
                ) + "\n\n"
        return clean_metadata_from_text(text)
    
    def _load_urls(self) -> str:
        full_text = ""
        pbar = tqdm(total=len(self.url_list) + len(self.blog_posts) + len(self.ilya_recs) + len(self.math_books) + len(self.local_txt_files), desc="Scraping URLs")
        
        pbar.set_description_str(f"Online PDF files")
        for url in self.url_list:
            pbar.update(1)
            try:
                text = self._get_content_from_pdf_url(url)
                pbar.write(text[:100])  # print first 100 characters for debugging
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        
        pbar.set_description_str(f"Blog posts")
        for url in tqdm(self.blog_posts):
            pbar.update(1)
            try:
                text = self._get_content_from_text_url(url)
                pbar.write(text[:100])  # print first 100 characters for debugging
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        
        pbar.set_description_str(f"Ilya's recommendations")
        for url in tqdm(self.ilya_recs):
            pbar.update(1)
            try:
                text = self._get_content_from_pdf_url(url)
                pbar.write(text[:100])  # print first 100 characters for debugging
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        
        pbar.set_description_str(f"Math books")
        for url in tqdm(self.math_books):
            pbar.update(1)
            try:
                text = self._get_content_from_pdf_url(url)
                pbar.write(text[:100])  # print first 100 characters for debugging
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {url}: {e}")
        
        pbar.set_description_str(f"Local .txt files")
        for file_path in tqdm(self.local_txt_files):
            pbar.update(1)
            try:
                text = self._load_local_txt_file(file_path)
                pbar.write(text[:100])  # print first 100 characters for debugging
                full_text += text + "\n"
            except Exception as e:
                tqdm.write(f"Failed to load {file_path}: {e}")
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

    def create_dataset(self, output_suffix: str, stage: Optional[str] = None):
        """
        Main method to create and save the dataset.

        Args:
            output_suffix: suffix for the dataset (e.g., 'pretrain')
            stage: optional curriculum stage name (e.g., 'foundation', 'pretrain')
        """
        from data_loaders.utils import to_bins

        print("Loading scraped dataset from URLs")
        if stage:
            print(f"Curriculum stage: {stage}")
        dataset = self.load_dataset()
        processed = self.process_dataset(dataset)

        print(f"Saving to binary files with suffix: {output_suffix}" + (f" and stage: {stage}" if stage else ""))
        to_bins(processed, suffix=output_suffix, is_sft=False, stage=stage)

        print("Dataset created successfully!")
        return None