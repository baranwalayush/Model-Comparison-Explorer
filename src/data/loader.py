"""
loader.py
---------
DataLoader: responsible for fetching the raw Titanic dataset
from seaborn's built-in loader or from a local CSV file.
"""

import pandas as pd
import seaborn as sns
from pathlib import Path
from src.utils import get_logger


class DataLoader:
    """
    Loads raw Titanic data from either seaborn (default) or a CSV file.

    Parameters
    ----------
    source : str
        'seaborn' to use the built-in dataset, 'csv' to load from disk.
    csv_path : str
        Path to the CSV file (used only when source='csv').
    """

    SUPPORTED_SOURCES = ("seaborn", "csv")

    def __init__(self, source: str = "seaborn", csv_path: str = "data/raw/titanic.csv"):
        if source not in self.SUPPORTED_SOURCES:
            raise ValueError(f"source must be one of {self.SUPPORTED_SOURCES}, got '{source}'")
        self.source = source
        self.csv_path = Path(csv_path)
        self.logger = get_logger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        """Load and return the raw DataFrame."""
        if self.source == "seaborn":
            return self._load_from_seaborn()
        return self._load_from_csv()

    def _load_from_seaborn(self) -> pd.DataFrame:
        self.logger.info("Loading Titanic dataset from seaborn...")
        df = sns.load_dataset("titanic")
        self.logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
        return df

    def _load_from_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {self.csv_path}")
        self.logger.info(f"Loading dataset from CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        self.logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
        return df