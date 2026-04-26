"""
preprocessor.py
---------------
DataPreprocessor: cleans, engineers features, encodes, and
splits the Titanic DataFrame into train/test sets ready for modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from src.utils import get_logger


class DataPreprocessor:
    """
    Handles all data preparation steps for the Titanic dataset.

    Steps performed (in order):
        1. Drop high-cardinality / leaking columns
        2. Feature engineering (family size, title extraction, is_alone)
        3. Impute missing values
        4. Encode categorical columns
        5. Train/test split

    Parameters
    ----------
    target_column : str
        Name of the label column.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Seed for reproducibility.
    """

    DROP_COLUMNS = ["embark_town", "who", "deck", "alive", "adult_male", "class", "Name", "PassengerId"]

    def __init__(
        self,
        target_column: str = "survived",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: list[str] = []
        self._encoders: dict[str, LabelEncoder] = {}
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Full pipeline: clean → engineer → impute → encode → split.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        self.logger.info("Starting preprocessing pipeline...")
        df = df.copy()
        df = self._drop_columns(df)
        df = self._engineer_features(df)
        df = self._impute(df)
        df = self._encode(df)
        X, y = self._separate_target(df)
        self.feature_names = X.columns.tolist()
        self.logger.info(f"Features retained: {self.feature_names}")
        return self._split(X, y)

    def get_feature_names(self) -> list[str]:
        """Return the list of features after preprocessing."""
        return self.feature_names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.DROP_COLUMNS if c in df.columns]
        self.logger.info(f"Dropping columns: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful derived features."""
        # Family size
        if "sibsp" in df.columns and "parch" in df.columns:
            df["family_size"] = df["sibsp"] + df["parch"] + 1
            df["is_alone"] = (df["family_size"] == 1).astype(int)

        # Age bands
        if "age" in df.columns:
            df["age_band"] = pd.cut(
                df["age"],
                bins=[0, 12, 18, 35, 60, 120],
                labels=["child", "teen", "young_adult", "adult", "senior"],
            )

        # Fare bands
        if "fare" in df.columns:
            df["fare_band"] = pd.qcut(df["fare"], q=4, labels=["low", "mid", "high", "very_high"])

        self.logger.info("Feature engineering complete.")
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with median / mode."""
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        self.logger.info("Imputation complete.")
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode all remaining categorical/object/category columns."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if col == self.target_column:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._encoders[col] = le
        self.logger.info(f"Encoded columns: {cat_cols}")
        return df

    def _separate_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found.")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].astype(int)
        return X, y

    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self.logger.info(
            f"Split complete — train: {len(X_train)}, test: {len(X_test)}"
        )
        return X_train, X_test, y_train, y_test