"""Unit tests for data loader service."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from provider_dedupe.core.exceptions import DataLoadError
from provider_dedupe.services.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_load_csv_file(self, temp_csv_file: Path):
        """Test loading CSV file."""
        loader = DataLoader()
        df = loader.load(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "npi" in df.columns

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        loader = DataLoader()
        
        with pytest.raises(DataLoadError, match="File not found"):
            loader.load("nonexistent.csv")

    def test_load_unsupported_format(self, tmp_path: Path):
        """Test error handling for unsupported file format."""
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("some content")
        
        loader = DataLoader()
        
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            loader.load(unsupported_file)

    def test_load_excel_file(self, tmp_path: Path, sample_dataframe: pd.DataFrame):
        """Test loading Excel file."""
        excel_file = tmp_path / "test.xlsx"
        sample_dataframe.to_excel(excel_file, index=False)
        
        loader = DataLoader()
        df = loader.load(excel_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)

    def test_load_json_file(self, tmp_path: Path, sample_dataframe: pd.DataFrame):
        """Test loading JSON file."""
        json_file = tmp_path / "test.json"
        sample_dataframe.to_json(json_file, orient="records")
        
        loader = DataLoader()
        df = loader.load(json_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)

    def test_load_parquet_file(self, tmp_path: Path, sample_dataframe: pd.DataFrame):
        """Test loading Parquet file."""
        parquet_file = tmp_path / "test.parquet"
        sample_dataframe.to_parquet(parquet_file, index=False)
        
        loader = DataLoader()
        df = loader.load(parquet_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)

    def test_load_csv_with_custom_params(self, temp_csv_file: Path):
        """Test loading CSV with custom parameters."""
        loader = DataLoader()
        df = loader.load(temp_csv_file, sep=",", header=0)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_multiple_files_concatenated(self, tmp_path: Path, sample_dataframe: pd.DataFrame):
        """Test loading multiple files with concatenation."""
        # Create multiple CSV files
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        
        sample_dataframe.iloc[:2].to_csv(file1, index=False)
        sample_dataframe.iloc[2:].to_csv(file2, index=False)
        
        loader = DataLoader()
        df = loader.load_multiple([file1, file2], concat=True)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)

    def test_load_multiple_files_separate(self, tmp_path: Path, sample_dataframe: pd.DataFrame):
        """Test loading multiple files without concatenation."""
        # Create multiple CSV files
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        
        sample_dataframe.iloc[:2].to_csv(file1, index=False)
        sample_dataframe.iloc[2:].to_csv(file2, index=False)
        
        loader = DataLoader()
        dataframes = loader.load_multiple([file1, file2], concat=False)
        
        assert isinstance(dataframes, list)
        assert len(dataframes) == 2
        assert all(isinstance(df, pd.DataFrame) for df in dataframes)

    def test_validate_schema_success(self, sample_dataframe: pd.DataFrame):
        """Test successful schema validation."""
        loader = DataLoader()
        required_columns = ["npi", "firstname", "lastname"]
        
        errors = loader.validate_schema(sample_dataframe, required_columns=required_columns)
        
        assert errors == []

    def test_validate_schema_missing_columns(self, sample_dataframe: pd.DataFrame):
        """Test schema validation with missing columns."""
        loader = DataLoader()
        required_columns = ["npi", "missing_column"]
        
        errors = loader.validate_schema(sample_dataframe, required_columns=required_columns)
        
        assert len(errors) == 1
        assert "missing_column" in errors[0]

    def test_validate_schema_column_types(self, sample_dataframe: pd.DataFrame):
        """Test schema validation with column types."""
        loader = DataLoader()
        # All columns are loaded as strings by default
        column_types = {"npi": "object"}
        
        errors = loader.validate_schema(sample_dataframe, column_types=column_types)
        
        assert errors == []

    @patch("pandas.read_csv")
    def test_load_csv_exception_handling(self, mock_read_csv, temp_csv_file: Path):
        """Test exception handling during CSV loading."""
        mock_read_csv.side_effect = Exception("Pandas error")
        
        loader = DataLoader()
        
        with pytest.raises(DataLoadError, match="Failed to load file"):
            loader.load(temp_csv_file)

    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        loader = DataLoader()
        expected_formats = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}
        
        assert loader.SUPPORTED_FORMATS == expected_formats

    def test_csv_loader_default_params(self, temp_csv_file: Path):
        """Test CSV loader uses correct default parameters."""
        loader = DataLoader()
        
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame()
            loader._load_csv(temp_csv_file)
            
            # Check that default parameters are used
            call_args = mock_read_csv.call_args
            assert call_args[1]["dtype"] == str
            assert "null" in call_args[1]["na_values"]

    def test_excel_loader_default_params(self, tmp_path: Path):
        """Test Excel loader uses correct default parameters."""
        excel_file = tmp_path / "test.xlsx"
        
        loader = DataLoader()
        
        with patch("pandas.read_excel") as mock_read_excel:
            mock_read_excel.return_value = pd.DataFrame()
            loader._load_excel(excel_file)
            
            # Check that default parameters are used
            call_args = mock_read_excel.call_args
            assert call_args[1]["dtype"] == str
            assert "null" in call_args[1]["na_values"]

    def test_json_loader_default_params(self, tmp_path: Path):
        """Test JSON loader uses correct default parameters."""
        json_file = tmp_path / "test.json"
        
        loader = DataLoader()
        
        with patch("pandas.read_json") as mock_read_json:
            mock_read_json.return_value = pd.DataFrame()
            loader._load_json(json_file)
            
            # Check that default parameters are used
            call_args = mock_read_json.call_args
            assert call_args[1]["dtype"] == str

    def test_parquet_loader(self, tmp_path: Path):
        """Test Parquet loader."""
        parquet_file = tmp_path / "test.parquet"
        
        loader = DataLoader()
        
        with patch("pandas.read_parquet") as mock_read_parquet:
            mock_read_parquet.return_value = pd.DataFrame()
            loader._load_parquet(parquet_file)
            
            # Parquet loader should be called
            mock_read_parquet.assert_called_once_with(parquet_file)