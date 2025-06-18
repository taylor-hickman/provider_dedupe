"""Pytest configuration and fixtures for provider deduplication tests."""

from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import pandas as pd
import pytest

from provider_dedupe.core.config import DeduplicationConfig
from provider_dedupe.models.provider import Provider, ProviderRecord
from provider_dedupe.utils.normalization import TextNormalizer


@pytest.fixture
def sample_provider_data() -> Dict[str, str]:
    """Sample provider data for testing."""
    return {
        "npi": "1234567893",  # Valid NPI with correct checksum
        "group_npi": "0987654320",  # Valid group NPI
        "group_name": "Medical Group LLC",
        "specialty": "Internal Medicine",
        "firstname": "John",
        "lastname": "Smith",
        "address1": "123 Main St",
        "city": "Springfield",
        "state": "IL",
        "zipcode": "62701",
        "phone": "2175551234",
        "address_status": "verified",
        "phone_status": "verified",
    }


@pytest.fixture
def sample_provider() -> Provider:
    """Sample Provider instance for testing."""
    return Provider(
        npi="1234567893",  # Valid NPI
        group_npi="0987654320",  # Valid group NPI
        group_name="Medical Group LLC",
        specialty="Internal Medicine",
        first_name="John",
        last_name="Smith",
        address_line_1="123 Main St",
        city="Springfield",
        state="IL",
        postal_code="62701",
        phone="2175551234",
    )


@pytest.fixture
def sample_provider_record(sample_provider: Provider) -> ProviderRecord:
    """Sample ProviderRecord instance for testing."""
    return ProviderRecord(provider=sample_provider)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame with provider data for testing."""
    data = [
        {
            "npi": "1234567893",
            "firstname": "John",
            "lastname": "Smith",
            "address1": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "2175551234",
        },
        {
            "npi": "1234567901",
            "firstname": "Jane",
            "lastname": "Doe",
            "address1": "456 Oak Ave",
            "city": "Chicago",
            "state": "IL",
            "zipcode": "60601",
            "phone": "3125559876",
        },
        {
            "npi": "1234567893",  # Duplicate NPI
            "firstname": "Jon",  # Slight variation
            "lastname": "Smith",
            "address1": "123 Main Street",  # Slight variation
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "2175551234",
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_duplicate_dataframe() -> pd.DataFrame:
    """Sample DataFrame with obvious duplicates for testing."""
    data = [
        {
            "unique_id": "1",
            "cluster_id": "cluster_1",
            "npi": "1234567893",
            "given_name": "john",
            "family_name": "smith",
            "cluster_size": 2,
        },
        {
            "unique_id": "2",
            "cluster_id": "cluster_1",
            "npi": "1234567893",
            "given_name": "jon",
            "family_name": "smith",
            "cluster_size": 2,
        },
        {
            "unique_id": "3",
            "cluster_id": "cluster_2",
            "npi": "1234567901",
            "given_name": "jane",
            "family_name": "doe",
            "cluster_size": 1,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def text_normalizer() -> TextNormalizer:
    """TextNormalizer instance for testing."""
    return TextNormalizer()


@pytest.fixture
def deduplication_config() -> DeduplicationConfig:
    """Sample deduplication configuration for testing."""
    return DeduplicationConfig(
        match_threshold=0.95,
        max_iterations=10,
        em_convergence=0.01,
    )


@pytest.fixture
def mock_data_loader() -> Mock:
    """Mock data loader for testing."""
    mock = Mock()
    mock.load.return_value = pd.DataFrame({
        "npi": ["1234567893", "1234567901"],
        "firstname": ["John", "Jane"],
        "lastname": ["Smith", "Doe"],
    })
    return mock


@pytest.fixture
def mock_quality_analyzer() -> Mock:
    """Mock quality analyzer for testing."""
    from provider_dedupe.services.data_quality import QualityMetrics
    
    mock = Mock()
    mock.analyze.return_value = QualityMetrics(
        total_records=100,
        duplicate_records=10,
        missing_value_summary={},
        field_statistics={},
        data_quality_score=85.0,
        recommendations=["Sample recommendation"],
    )
    return mock


@pytest.fixture
def temp_csv_file(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_config_file(tmp_path: Path, deduplication_config: DeduplicationConfig) -> Path:
    """Create a temporary configuration file for testing."""
    import json
    
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(deduplication_config.model_dump(), f)
    return config_path


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup minimal logging for tests."""
    import logging
    
    logging.basicConfig(level=logging.CRITICAL)  # Suppress most log output
    yield
    # Cleanup handled automatically


@pytest.fixture
def provider_variations() -> List[Dict[str, str]]:
    """List of provider record variations for testing normalization."""
    return [
        {
            "npi": "1234567893",
            "firstname": "John",
            "lastname": "Smith",
            "address1": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "(217) 555-1234",
        },
        {
            "npi": "1234567893",
            "firstname": "Jon",
            "lastname": "Smith",
            "address1": "123 Main Street",
            "city": "Springfield",
            "state": "Illinois",
            "zipcode": "62701-1234",
            "phone": "217-555-1234",
        },
        {
            "npi": "1234567893",
            "firstname": "J.",
            "lastname": "Smith",
            "address1": "123 Main St.",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "2175551234",
        },
    ]