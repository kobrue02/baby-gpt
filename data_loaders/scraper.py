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
    "https://irp-cdn.multiscreensite.com/cb9165b2/files/uploaded/The%20Intelligent%20Investor%20-%20BENJAMIN%20GRAHAM.pdf", # the intelligent investor
    "https://files.romanroadsstatic.com/materials/romans/Aeneid-RRM-etext_v1.0.pdf", # the aeneid
    "https://icrrd.com/public/media/16-05-2021-070111The-Richest-Man-in-Babylon.pdf", # the richest man in babylon
    "https://sites.ualberta.ca/~enoch/Readings/The_Art_Of_War.pdf", # the art of war
    "https://antilogicalism.com/wp-content/uploads/2017/07/history-pelo-war.pdf", # the history of the peloponnesian war
    "https://ia801406.us.archive.org/24/items/gorwell1984de/1984.pdf", # 1984
    "http://www.daviddfriedman.com/Machinery%203rd%20Edn.pdf", # machinery of freedom
    "http://pombo.free.fr/friedman2002.pdf", # capitalism and freedom
    "https://dn790000.ca.archive.org/0/items/animalfarm00orwe_0/animalfarm00orwe_0.pdf", # animal farm
    "https://www.lopp.net/pdf/The%20Sovereign%20Individual.pdf", # the sovereign individual
    "https://ia903107.us.archive.org/35/items/j-r-r-tolkien-lord-of-the-rings-01-the-fellowship-of-the-ring-retail-pdf/j-r-r-tolkien-lord-of-the-rings-01-the-fellowship-of-the-ring-retail-pdf.pdf", # lotr fellowship
    "https://wrenchinthegears.com/wp-content/uploads/2023/03/The-Diamond-Age-Novel.pdf", # the diamond age
    "https://identityhunters.org/wp-content/uploads/2017/07/niccolo-machiavelli-discourses-of-livy.pdf", # discourses on livy
    "https://antilogicalism.com/wp-content/uploads/2018/04/reasonableness-christianity.pdf", # reasonableness of christianity
    "https://ia601305.us.archive.org/29/items/cu31924007365467/cu31924007365467.pdf", # the great illusion
    "https://www.thomasmorestudies.org/wp-content/uploads/2020/09/Bacon-New-Atlantis-2020-Edition-7-6-2020.pdf", # new atlantis
    "https://rickbulow.com/Library/Books/Non-Fiction/Political/IntellectualsAndSocietyByThomasSowell.pdf", # intellectuals and society
    "https://theworthyhouse.com/wp-content/uploads/2020/03/Decadent-Society-Douthat-PDF.pdf", # decadent society
    "https://ia601503.us.archive.org/23/items/AsimovTheFoundation/Asimov_the_foundation.pdf", # asimov foundation
    "https://myaccount.inspiruseducation.com/wp-content/uploads/2022/02/The-Hitchhikers-Guide-to-the-Galaxy-Douglas-Adams.pdf", # hitchhiker's guide
    "https://dn790001.ca.archive.org/0/items/autobiobenfran00miffrich/autobiobenfran00miffrich.pdf", # autobiography of ben franklin
    "https://www.mondotheque.be/wiki/images/c/cc/Dave_Eggers_The_Circle.pdf", # the circle
    "https://books-library.website/files/books-library.online-01120755Hy9M4.pdf", # stranger in a strange land
    "https://ia902904.us.archive.org/26/items/ErnstJngerTheStormOfSteel/Ernst_J%C3%BCnger_The_Storm_of_Steel.pdf", # the storm of steel
    "https://ntrs.nasa.gov/api/citations/19710019929/downloads/19710019929.pdf", # rocket engines
    "https://virtualmmx.ddns.net/gbooks/OurFinalInvention.pdf", # our final invention
    "https://www.friendsofsabbath.org/Further_Research/e-books/Durant%20story%20of%20civilization.pdf", # the story of civilization
    "https://library.sciencemadness.org/library/books/ignition.pdf", # ignition
    "https://uberty.org/wp-content/uploads/2015/12/herman-melville-moby-dick.pdf", # moby dick
    "https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/king-lear_PDF_FolgerShakespeare.pdf", # king leare
    "https://staibabussalamsula.ac.id/wp-content/uploads/2023/11/yuval_noah_harari-sapiens_a_brief_histor.pdf", # sapiens
    "https://ross.aoe.vt.edu/books/Ross_3BodyProblem_Book_2022.pdf", # three body problem
    "https://images.pcmac.org/SiSFiles/Schools/AL/LeedsCity/LeedsMiddle/Uploads/DocumentsCategories/Documents/Enders_Game_Full_Book.pdf", # ender's game

]

ILYA_RECS = [
    "https://arxiv.org/pdf/1409.2329", # rnn regularization
    "https://www.cs.toronto.edu/~fritz/absps/colt93.pdf", # keeping neural nets simple
    "https://arxiv.org/pdf/1506.03134", # pointer networks
    "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf", # imagenet
    "https://arxiv.org/pdf/1511.06391", # seq2seq
    "https://arxiv.org/pdf/1811.06965", # gpipe
    "https://arxiv.org/pdf/1512.03385", # deep residual learning
    "https://arxiv.org/pdf/1511.07122", # context aggregation
    "https://arxiv.org/pdf/1704.01212", # neural quantum chemistry
    "https://arxiv.org/pdf/1706.03762", # attention is all you need
    "https://arxiv.org/pdf/1409.0473", # neural MT
    "https://arxiv.org/pdf/1603.05027", # identity mappings in deep residual networks
    "https://arxiv.org/pdf/1706.01427", # relational reasoning
    "https://arxiv.org/pdf/1611.02731", # variational lossy autoencoder
    "https://arxiv.org/pdf/1806.01822", # relational rnns
    "https://arxiv.org/pdf/1405.6903", # the coffee automaton
    "https://arxiv.org/pdf/1410.5401", # neural turing machines
    "https://arxiv.org/pdf/1512.02595", # deep speech 2
    "https://arxiv.org/pdf/2001.08361", # scaling laws
    "https://arxiv.org/pdf/math/0406077", # minimum description length
    "https://www.vetta.org/documents/Machine_Super_Intelligence.pdf", # superintelligence
    "https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf", # kolmogorov complexity

]

BLOG_POSTS = [
    "https://transformer-circuits.pub/2025/attribution-graphs/biology.html", # anthropic on llm biology
    "https://nlp.seas.harvard.edu/annotated-transformer/", # annotated transformer
    "https://scottaaronson.blog/?p=762", # complexodynamics
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/", # karpathy rnn
    "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", # colah lstm
    "https://www.gutenberg.org/files/4061/4061-h/4061-h.htm", # fifteen battles
    "https://terebess.hu/zen/mesterek/ZMBM-Lectures.html", # zen mind, beginner's mind

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
        self.blog_posts = BLOG_POSTS
        self.ilya_recs = ILYA_RECS
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