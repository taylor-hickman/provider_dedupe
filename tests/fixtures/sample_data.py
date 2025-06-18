"""Sample data fixtures for testing."""

import pandas as pd
from typing import Dict, List, Any


def _generate_valid_npi_for_dataset(index: int) -> str:
    """Generate a valid NPI for large dataset based on index."""
    # Use different base patterns to ensure variety while maintaining validity
    base_patterns = [
        "123456789", "123456790", "123456791", "123456792", "123456793",
        "123456794", "123456795", "123456796", "123456797", "123456798"
    ]
    base = base_patterns[index % len(base_patterns)]
    
    # Luhn algorithm for NPI validation
    payload = [int(d) for d in base]
    for i in range(len(payload) - 1, -1, -2):
        payload[i] *= 2
        if payload[i] > 9:
            payload[i] -= 9
    
    total = sum(payload) + 24  # 24 is the sum for the prefix 80840
    check_digit = (10 - (total % 10)) % 10
    
    return base + str(check_digit)


def _generate_valid_group_npi_for_dataset(index: int) -> str:
    """Generate a valid group NPI for large dataset based on index."""
    # Use different base patterns for group NPIs
    base_patterns = [
        "098765432", "098765433", "098765434", "098765435", "098765436",
        "098765437", "098765438", "098765439", "098765440", "098765441"
    ]
    base = base_patterns[index % len(base_patterns)]
    
    # Luhn algorithm for NPI validation
    payload = [int(d) for d in base]
    for i in range(len(payload) - 1, -1, -2):
        payload[i] *= 2
        if payload[i] > 9:
            payload[i] -= 9
    
    total = sum(payload) + 24  # 24 is the sum for the prefix 80840
    check_digit = (10 - (total % 10)) % 10
    
    return base + str(check_digit)


def get_sample_provider_data() -> List[Dict[str, Any]]:
    """Get sample provider data for testing."""
    return [
        {
            "npi": "1234567893",
            "firstname": "John",
            "lastname": "Smith",
            "address1": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "2175551234",
            "gnpi": "0987654320",
            "group_name": "Springfield Medical Group",
            "primary_spec_desc": "Internal Medicine",
            "address_status": "verified",
            "phone_status": "verified",
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
            "gnpi": "0987654338",
            "group_name": "Chicago Health Center",
            "primary_spec_desc": "Family Medicine",
            "address_status": "verified",
            "phone_status": "verified",
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
            "gnpi": "0987654320",
            "group_name": "Springfield Medical Group",
            "primary_spec_desc": "Internal Medicine",
            "address_status": "inconclusive",
            "phone_status": "verified",
        },
        {
            "npi": "1234567919",
            "firstname": "Robert",
            "lastname": "Johnson",
            "address1": "789 Pine St",
            "city": "Peoria",
            "state": "IL",
            "zipcode": "61602",
            "phone": "3095557890",
            "gnpi": None,  # Solo practitioner
            "group_name": None,
            "primary_spec_desc": "Cardiology",
            "address_status": "verified",
            "phone_status": "inconclusive",
        },
        {
            "npi": "1234567893",
            "firstname": "Mary",
            "lastname": "Williams",
            "address1": "321 Elm Dr",
            "city": "Rockford",
            "state": "IL",
            "zipcode": "61101",
            "phone": "8155552345",
            "gnpi": "0987654346",
            "group_name": "Rockford Specialists",
            "primary_spec_desc": "Dermatology",
            "address_status": "verified",
            "phone_status": "verified",
        },
    ]


def get_sample_dataframe() -> pd.DataFrame:
    """Get sample DataFrame for testing."""
    return pd.DataFrame(get_sample_provider_data())


def get_duplicate_variations() -> List[Dict[str, Any]]:
    """Get variations of the same provider for testing normalization."""
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
        {
            "npi": "1234567893",
            "firstname": "John A",
            "lastname": "Smith",
            "address1": "123 N Main St",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "217.555.1234 ext 890",
        },
    ]


def get_quality_issues_data() -> List[Dict[str, Any]]:
    """Get data with various quality issues for testing."""
    return [
        {
            "npi": "1234567893",
            "firstname": "John",
            "lastname": "Smith",
            "address1": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zipcode": "62701",
            "phone": "2175551234",
            "address_status": "verified",
            "phone_status": "verified",
        },
        {
            "npi": "1234567901",
            "firstname": "",  # Missing first name
            "lastname": "Doe",
            "address1": "456 Oak Ave",
            "city": "Chicago",
            "state": "IL",
            "zipcode": "60601",
            "phone": "312555",  # Invalid phone
            "address_status": "inconclusive",
            "phone_status": "invalid",
        },
        {
            "npi": "123456789",  # Invalid NPI (too short)
            "firstname": "Jane",
            "lastname": "",  # Missing last name
            "address1": "null",  # Null string
            "city": "null",
            "state": "IL",
            "zipcode": "",  # Empty zipcode
            "phone": "",
            "address_status": "no",
            "phone_status": "no",
        },
        {
            "npi": "1234567919",
            "firstname": "Robert",
            "lastname": "Johnson",
            "address1": None,  # Actual None
            "city": None,
            "state": "IL",
            "zipcode": None,
            "phone": None,
            "address_status": "inconclusive",
            "phone_status": "inconclusive",
        },
    ]


def get_normalized_results_data() -> List[Dict[str, Any]]:
    """Get sample normalized results for testing output generation."""
    return [
        {
            "unique_id": "uuid-1",
            "cluster_id": "cluster_1",
            "npi": "1234567893",
            "given_name": "john",
            "family_name": "smith",
            "street_address": "123 main street",
            "city": "springfield",
            "state": "IL",
            "postal_code": "62701",
            "phone": "2175551234",
            "cluster_size": 2,
        },
        {
            "unique_id": "uuid-2",
            "cluster_id": "cluster_1",
            "npi": "1234567893",
            "given_name": "jon",
            "family_name": "smith",
            "street_address": "123 main street",
            "city": "springfield",
            "state": "IL",
            "postal_code": "62701",
            "phone": "2175551234",
            "cluster_size": 2,
        },
        {
            "unique_id": "uuid-3",
            "cluster_id": "cluster_2",
            "npi": "1234567901",
            "given_name": "jane",
            "family_name": "doe",
            "street_address": "456 oak avenue",
            "city": "chicago",
            "state": "IL",
            "postal_code": "60601",
            "phone": "3125559876",
            "cluster_size": 1,
        },
    ]


def get_large_dataset(size: int = 1000) -> pd.DataFrame:
    """Generate a larger synthetic dataset for performance testing.
    
    Args:
        size: Number of records to generate
        
    Returns:
        DataFrame with synthetic provider data
    """
    import random
    from faker import Faker
    
    fake = Faker()
    fake.seed_instance(42)  # For reproducible results
    
    data = []
    for i in range(size):
        # Generate some duplicates
        if i % 10 == 0 and i > 0:
            # Create a duplicate of a previous record with variations
            base_record = data[i - random.randint(1, min(9, i))]
            record = base_record.copy()
            # Add slight variations
            record["firstname"] = record["firstname"] + random.choice(["", " Jr", " Sr"])
            record["address1"] = record["address1"].replace("St", "Street")
            # Ensure phone remains 10 digits
            if len(record["phone"]) >= 10:
                record["phone"] = record["phone"][:9] + str(random.randint(0, 9))
            else:
                record["phone"] = fake.numerify("##########")
        else:
            # Generate new record
            record = {
                "npi": _generate_valid_npi_for_dataset(i),
                "firstname": fake.first_name(),
                "lastname": fake.last_name(),
                "address1": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zipcode": fake.zipcode(),
                "phone": fake.numerify("##########"),  # Generate exactly 10 digits
                "gnpi": _generate_valid_group_npi_for_dataset(i) if i % 3 == 0 else None,
                "group_name": fake.company() if i % 3 == 0 else None,
                "primary_spec_desc": random.choice([
                    "Internal Medicine", "Family Medicine", "Cardiology",
                    "Dermatology", "Pediatrics", "Surgery"
                ]),
                "address_status": random.choice(["verified", "inconclusive", "invalid"]),
                "phone_status": random.choice(["verified", "inconclusive", "invalid"]),
            }
        
        data.append(record)
    
    return pd.DataFrame(data)