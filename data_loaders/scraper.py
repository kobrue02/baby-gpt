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
    "https://www.readdiary.com/wp-content/uploads/2022/05/Steve-Jobs-PDFDrive-1.pdf", # steve jobs
    "https://www.cia.gov/library/abbottabad-compound/36/36669B7894E857AC4F3445EA646BFFE1_Zbigniew_Brzezinski_-_The_Grand_ChessBoard.doc.pdf", # the grand chessboard
    "https://mo.tnu.tj/wp-content/uploads/2020/11/strategic_vision__america_and_the_crisis_of_global_power.pdf", # strategic vision
    "https://web.cs.ucdavis.edu/~rogaway/classes/188/materials/Industrial%20Society%20and%20Its%20Future.pdf", # kaczynski manifesto
    "https://wiki.chadnet.org/files/generative-energy-restoring-the-wholeness-of-life.pdf", # ray peat generative energy
    "https://wiki.chadnet.org/files/thus-spake-zarathustra.pdf", # thus spoke zarathustra
    "https://wiki.chadnet.org/files/through-the-brazilian-wilderness.pdf", # through the brazilian wilderness
    "https://wiki.chadnet.org/files/the-complete-book-of-self-sufficiency.pdf", # the complete book of self-sufficiency
    "https://wiki.chadnet.org/anton-johanson-the-christian-seer-from-the-norwegian-finnmark.pdf", # the christian seer
    "https://wiki.chadnet.org/files/the-foundations-of-arithmetic-a-logico-mathematical-enquiry-into-the-concept-of-number-by-gottlob-frege.pdf", # the foundations of arithmetic
    "https://wiki.chadnet.org/files/the-milner-fabian-conspiracy.pdf", # the milner fabian conspiracy
    "https://wiki.chadnet.org/files/the-anglo-american-establishment.pdf", # the anglo american establishment
    "https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf", # machine learning yearning
    "https://dn721903.ca.archive.org/0/items/american-psycho-BEE/American%20Psycho%20%28Bret%20Easton%20Ellis%29%20%28z-lib.org%29.pdf", # american psycho
    "https://ia801208.us.archive.org/24/items/TheFabricOfReality/The_Fabric_of_Reality.pdf", # the fabric of reality
    "https://assets.stripeassets.com/fzn2n1nzq965/5j0dFbeGgGbohTE3a2jrVA/ebd35e791ca5fa926c6a0b076860c71c/ZINE-Scaling_Era-singles.pdf", # the scaling era
    "https://dn710000.ca.archive.org/0/items/dli.pahar.3637/1990%20The%20Great%20Game--On%20Secret%20Service%20in%20High%20Asia%20by%20Hopkirk%20s.pdf", # the great game
    "https://www.rrojasdatabank.info/Wealth-Nations.pdf", # the wealth of nations
    "https://faculty.econ.ucdavis.edu/faculty/bonanno/PDF/GT_book.pdf", # game theory
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