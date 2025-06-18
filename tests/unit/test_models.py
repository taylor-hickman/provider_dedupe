"""Unit tests for provider data models."""

import pytest
from pydantic import ValidationError

from provider_dedupe.models.provider import (
    AddressStatus,
    PhoneStatus,
    Provider,
    ProviderRecord,
)


class TestProvider:
    """Test cases for Provider model."""

    def test_valid_provider_creation(self, sample_provider: Provider):
        """Test creating a valid provider."""
        assert sample_provider.npi == "1234567893"
        assert sample_provider.first_name == "John"
        assert sample_provider.last_name == "Smith"
        assert sample_provider.state == "IL"

    def test_npi_validation(self):
        """Test NPI validation including Luhn algorithm."""
        # Valid NPI
        provider = Provider(
            npi="1234567893",  # Valid Luhn checksum
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="IL",
            postal_code="62701",
        )
        assert provider.npi == "1234567893"

        # Invalid NPI format
        with pytest.raises(ValidationError) as exc_info:
            Provider(
                npi="123456789",  # Too short
                first_name="John",
                last_name="Smith",
                address_line_1="123 Main St",
                city="Springfield",
                state="IL",
                postal_code="62701",
            )
        assert "at least 10 characters" in str(exc_info.value)

        # Invalid NPI with letters
        with pytest.raises(ValidationError):
            Provider(
                npi="123456789A",
                first_name="John",
                last_name="Smith",
                address_line_1="123 Main St",
                city="Springfield",
                state="IL",
                postal_code="62701",
            )

    def test_state_validation(self):
        """Test state code validation and normalization."""
        provider = Provider(
            npi="1234567893",
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="il",  # Lowercase input
            postal_code="62701",
        )
        assert provider.state == "IL"  # Should be uppercase

    def test_postal_code_validation(self):
        """Test postal code validation."""
        # Valid 5-digit ZIP
        provider = Provider(
            npi="1234567893",
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="IL",
            postal_code="62701",
        )
        assert provider.postal_code == "62701"

        # Valid ZIP+4
        provider = Provider(
            npi="1234567893",
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="IL",
            postal_code="62701-1234",
        )
        assert provider.postal_code == "62701-1234"

        # Invalid postal code
        with pytest.raises(ValidationError):
            Provider(
                npi="1234567893",
                first_name="John",
                last_name="Smith",
                address_line_1="123 Main St",
                city="Springfield",
                state="IL",
                postal_code="ABC123",
            )

    def test_phone_validation(self):
        """Test phone number validation."""
        provider = Provider(
            npi="1234567893",
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="IL",
            postal_code="62701",
            phone="2175551234",
        )
        assert provider.phone == "2175551234"

        # Invalid phone (too short)
        with pytest.raises(ValidationError):
            Provider(
                npi="1234567893",
                first_name="John",
                last_name="Smith",
                address_line_1="123 Main St",
                city="Springfield",
                state="IL",
                postal_code="62701",
                phone="123",
            )

    def test_optional_fields(self):
        """Test optional field handling."""
        provider = Provider(
            npi="1234567893",
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="IL",
            postal_code="62701",
        )
        assert provider.middle_name is None
        assert provider.group_npi is None
        assert provider.phone is None

    def test_enum_fields(self):
        """Test enum field validation."""
        provider = Provider(
            npi="1234567893",
            first_name="John",
            last_name="Smith",
            address_line_1="123 Main St",
            city="Springfield",
            state="IL",
            postal_code="62701",
            address_status=AddressStatus.VERIFIED,
            phone_status=PhoneStatus.INCONCLUSIVE,
        )
        assert provider.address_status == AddressStatus.VERIFIED
        assert provider.phone_status == PhoneStatus.INCONCLUSIVE


class TestProviderRecord:
    """Test cases for ProviderRecord model."""

    def test_provider_record_creation(self, sample_provider_record: ProviderRecord):
        """Test creating a provider record."""
        assert sample_provider_record.unique_id is not None
        assert sample_provider_record.provider is not None
        assert sample_provider_record.cluster_id is None
        assert sample_provider_record.match_probability is None
        assert sample_provider_record.is_primary is False

    def test_from_dict_legacy_mapping(self, sample_provider_data):
        """Test creating provider record from legacy data format."""
        record = ProviderRecord.from_dict(sample_provider_data)

        assert record.provider.npi == "1234567893"
        assert record.provider.first_name == "John"
        assert record.provider.last_name == "Smith"
        assert record.provider.address_line_1 == "123 Main St"

    def test_from_dict_null_handling(self):
        """Test null value handling in from_dict."""
        data = {
            "npi": "1234567893",
            "firstname": "John",
            "lastname": "Smith",
            "address1": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "null",  # String null
            "group_npi": None,  # Actual None
        }

        record = ProviderRecord.from_dict(data)
        assert record.provider.phone is None
        assert record.provider.group_npi is None

    def test_match_probability_validation(self, sample_provider: Provider):
        """Test match probability validation."""
        # Valid probability
        record = ProviderRecord(
            provider=sample_provider,
            match_probability=0.95,
        )
        assert record.match_probability == 0.95

        # Invalid probability (too high)
        with pytest.raises(ValidationError):
            ProviderRecord(
                provider=sample_provider,
                match_probability=1.5,
            )

        # Invalid probability (negative)
        with pytest.raises(ValidationError):
            ProviderRecord(
                provider=sample_provider,
                match_probability=-0.1,
            )

    def test_cluster_assignment(self, sample_provider: Provider):
        """Test cluster assignment functionality."""
        record = ProviderRecord(
            provider=sample_provider,
            cluster_id="cluster_123",
            match_probability=0.98,
            is_primary=True,
        )

        assert record.cluster_id == "cluster_123"
        assert record.match_probability == 0.98
        assert record.is_primary is True


class TestEnums:
    """Test cases for enum types."""

    def test_address_status_enum(self):
        """Test AddressStatus enum values."""
        assert AddressStatus.VERIFIED == "verified"
        assert AddressStatus.INCONCLUSIVE == "inconclusive"
        assert AddressStatus.INVALID == "invalid"
        assert AddressStatus.NO == "no"

    def test_phone_status_enum(self):
        """Test PhoneStatus enum values."""
        assert PhoneStatus.VERIFIED == "verified"
        assert PhoneStatus.INCONCLUSIVE == "inconclusive"
        assert PhoneStatus.INVALID == "invalid"
        assert PhoneStatus.NO == "no"
