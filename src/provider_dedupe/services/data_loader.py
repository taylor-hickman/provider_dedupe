"""Data loading service with support for multiple file formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from provider_dedupe.core.exceptions import DataLoadError
from provider_dedupe.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Service for loading data from various file formats.

    Supports CSV, Excel, JSON, and Parquet formats with automatic
    format detection based on file extension.
    """

    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}

    def __init__(self) -> None:
        """Initialize the data loader."""
        self._loaders = {
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json,
            ".parquet": self._load_parquet,
            ".pq": self._load_parquet,
        }

    def load(
        self,
        file_path: Union[str, Path],
        **kwargs: Dict,
    ) -> pd.DataFrame:
        """Load data from file.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to the loader

        Returns:
            Loaded DataFrame

        Raises:
            DataLoadError: If file cannot be loaded
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        # Validate file format
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise DataLoadError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Load file
        loader = self._loaders[file_path.suffix.lower()]
        try:
            logger.info("Loading file", path=str(file_path), format=file_path.suffix)
            df = loader(file_path, **kwargs)
            logger.info(
                "File loaded successfully", rows=len(df), columns=len(df.columns)
            )
            return df
        except Exception as e:
            raise DataLoadError(f"Failed to load file {file_path}: {e}")

    def _load_csv(self, file_path: Path, **kwargs: Dict) -> pd.DataFrame:
        """Load CSV file.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv arguments

        Returns:
            Loaded DataFrame
        """
        # Set default parameters
        params = {
            "dtype": str,  # Load all columns as strings initially
            "na_values": ["null", "NULL", "None", "NONE", ""],
            "keep_default_na": True,
        }
        params.update(kwargs)

        return pd.read_csv(file_path, **params)

    def _load_excel(self, file_path: Path, **kwargs: Dict) -> pd.DataFrame:
        """Load Excel file.

        Args:
            file_path: Path to Excel file
            **kwargs: Additional pandas read_excel arguments

        Returns:
            Loaded DataFrame
        """
        params = {
            "dtype": str,
            "na_values": ["null", "NULL", "None", "NONE", ""],
        }
        params.update(kwargs)

        return pd.read_excel(file_path, **params)

    def _load_json(self, file_path: Path, **kwargs: Dict) -> pd.DataFrame:
        """Load JSON file.

        Args:
            file_path: Path to JSON file
            **kwargs: Additional pandas read_json arguments

        Returns:
            Loaded DataFrame
        """
        params: Dict[str, Any] = {"dtype": str}
        params.update(kwargs)

        return pd.read_json(file_path, **params)

    def _load_parquet(self, file_path: Path, **kwargs: Dict) -> pd.DataFrame:
        """Load Parquet file.

        Args:
            file_path: Path to Parquet file
            **kwargs: Additional pandas read_parquet arguments

        Returns:
            Loaded DataFrame
        """
        return pd.read_parquet(file_path, **kwargs)

    def load_multiple(
        self,
        file_paths: List[Union[str, Path]],
        concat: bool = True,
        **kwargs: Dict,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Load multiple files.

        Args:
            file_paths: List of file paths to load
            concat: Whether to concatenate into single DataFrame
            **kwargs: Additional arguments passed to loaders

        Returns:
            Single concatenated DataFrame or list of DataFrames

        Raises:
            DataLoadError: If any file cannot be loaded
        """
        dataframes = []

        for file_path in file_paths:
            df = self.load(file_path, **kwargs)
            dataframes.append(df)

        if concat:
            logger.info("Concatenating dataframes", count=len(dataframes))
            return pd.concat(dataframes, ignore_index=True)

        return dataframes

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, type]] = None,
    ) -> List[str]:
        """Validate DataFrame schema.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            column_types: Dictionary of column names to expected types

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")

        # Check column types
        if column_types:
            for column, expected_type in column_types.items():
                if column in df.columns:
                    actual_type = df[column].dtype
                    if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                        errors.append(
                            f"Column '{column}' has type {actual_type}, "
                            f"expected {expected_type}"
                        )

        return errors
