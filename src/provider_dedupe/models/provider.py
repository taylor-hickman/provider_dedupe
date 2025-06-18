"""Provider data models with validation."""

from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class AddressStatus(str, Enum):
    """Address quality status enumeration."""

    VERIFIED = "verified"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"
    NO = "no"


class PhoneStatus(str, Enum):
    """Phone quality status enumeration."""

    VERIFIED = "verified"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"
    NO = "no"


class AddressType(str, Enum):
    """Address type enumeration."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    BILLING = "billing"
    MAILING = "mailing"


class Provider(BaseModel):
    """Provider information model with validation.

    Attributes:
        npi: National Provider Identifier
        group_npi: Group practice NPI
        group_name: Name of the provider group/organization
        specialty: Provider's primary specialty description
        first_name: Provider's first name
        middle_name: Provider's middle name (optional)
        last_name: Provider's last name
        address_type: Type of address (primary, secondary, etc.)
        address_line_1: Primary street address
        address_line_2: Secondary address information (optional)
        city: City name
        state: State code (2 characters)
        postal_code: ZIP or postal code
        phone: Phone number
        address_status: Quality indicator for address
        phone_status: Quality indicator for phone number
    """

    npi: str = Field(..., min_length=10, max_length=10, pattern=r"^\d{10}$")
    group_npi: Optional[str] = Field(
        None, min_length=10, max_length=10, pattern=r"^\d{10}$"
    )
    group_name: Optional[str] = Field(None, max_length=255)
    specialty: Optional[str] = Field(None, max_length=100)
    first_name: str = Field(..., min_length=1, max_length=50)
    middle_name: Optional[str] = Field(None, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    address_type: AddressType = Field(default=AddressType.PRIMARY)
    address_line_1: str = Field(..., min_length=1, max_length=100)
    address_line_2: Optional[str] = Field(None, max_length=100)
    city: str = Field(..., min_length=1, max_length=50)
    state: str = Field(..., min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")
    postal_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    phone: Optional[str] = Field(None, pattern=r"^\d{10,}$")
    address_status: AddressStatus = Field(default=AddressStatus.INCONCLUSIVE)
    phone_status: PhoneStatus = Field(default=PhoneStatus.INCONCLUSIVE)

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, value: str) -> str:
        """Ensure state code is uppercase."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("address_type", mode="before")
    @classmethod
    def validate_address_type(cls, value: str) -> str:
        """Convert address type to lowercase."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("address_status", mode="before") 
    @classmethod
    def validate_address_status(cls, value: str) -> str:
        """Convert address status to lowercase."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("phone_status", mode="before")
    @classmethod
    def validate_phone_status(cls, value: str) -> str:
        """Convert phone status to lowercase."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("npi", "group_npi")
    @classmethod
    def validate_npi(cls, value: Optional[str]) -> Optional[str]:
        """Validate NPI using Luhn algorithm."""
        if value is None:
            return value

        if not value.isdigit() or len(value) != 10:
            raise ValueError("NPI must be exactly 10 digits")

        # Luhn algorithm validation
        digits = [int(d) for d in value]
        check_digit = digits[-1]
        payload = digits[:-1]

        # Double every second digit from right
        for i in range(len(payload) - 1, -1, -2):
            payload[i] *= 2
            if payload[i] > 9:
                payload[i] -= 9

        total = sum(payload) + 24  # 24 is the sum for the prefix 80840
        calculated_check = (10 - (total % 10)) % 10

        if calculated_check != check_digit:
            raise ValueError(f"Invalid NPI checksum: {value}")

        return value
    
    @field_validator('group_npi', 'group_name', 'specialty', 'middle_name', 'address_line_2', 'phone', mode='before')
    @classmethod
    def handle_nan_values(cls, value):
        """Convert NaN values to None for optional fields."""
        import math
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @field_validator('first_name', 'last_name', 'address_line_1', 'city', 'state', 'postal_code', mode='before')
    @classmethod
    def handle_nan_required_fields(cls, value):
        """Convert NaN values to empty string for required fields, which will trigger validation error."""
        import math
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value

    model_config = ConfigDict(
        json_encoders={UUID: str},
        use_enum_values=True
    )


class ProviderRecord(BaseModel):
    """Extended provider record with deduplication metadata.

    Attributes:
        unique_id: Unique identifier for the record
        provider: Core provider information
        cluster_id: Assigned cluster ID after deduplication
        match_probability: Probability of match with cluster
        is_primary: Whether this is the primary record in cluster
    """

    unique_id: UUID = Field(default_factory=uuid4)
    provider: Provider
    cluster_id: Optional[str] = None
    match_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_primary: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderRecord":
        """Create ProviderRecord from dictionary.

        Args:
            data: Dictionary containing provider information

        Returns:
            ProviderRecord instance

        Raises:
            ValueError: If required fields are missing
        """
        # Map legacy column names to new model
        field_mapping = {
            "firstname": "first_name",
            "lastname": "last_name",
            "middlename": "middle_name",
            "gnpi": "group_npi",
            "primary_spec_desc": "specialty",
            "address1": "address_line_1",
            "address2": "address_line_2",
            "zipcode": "postal_code",
            "address_category": "address_type",
        }

        # Apply field mapping
        mapped_data = {}
        for old_key, new_key in field_mapping.items():
            if old_key in data:
                mapped_data[new_key] = data[old_key]

        # Copy unmapped fields
        for key, value in data.items():
            if key not in field_mapping and key not in mapped_data:
                mapped_data[key] = value

        # Handle null strings
        for key, value in mapped_data.items():
            if value == "null" or value == "NULL":
                mapped_data[key] = None

        # Create provider instance
        provider = Provider(**mapped_data)
        
        return cls(provider=provider)