"""Test utilities for provider deduplication tests."""


def generate_valid_npi(base_digits: str) -> str:
    """Generate a valid NPI by calculating the correct checksum.

    Args:
        base_digits: First 9 digits of the NPI (must be 9 digits)

    Returns:
        Valid 10-digit NPI with correct checksum
    """
    if not base_digits.isdigit() or len(base_digits) != 9:
        raise ValueError("Base digits must be exactly 9 digits")

    # Convert to list of integers
    payload = [int(d) for d in base_digits]

    # Double every second digit from right
    for i in range(len(payload) - 1, -1, -2):
        payload[i] *= 2
        if payload[i] > 9:
            payload[i] -= 9

    # Calculate checksum (24 is the sum for the prefix 80840)
    total = sum(payload) + 24
    check_digit = (10 - (total % 10)) % 10

    return base_digits + str(check_digit)


def generate_valid_group_npi(base_digits: str) -> str:
    """Generate a valid group NPI (same algorithm as individual NPI).

    Args:
        base_digits: First 9 digits of the group NPI

    Returns:
        Valid 10-digit group NPI with correct checksum
    """
    return generate_valid_npi(base_digits)


# Pre-generated valid NPIs for common test use
VALID_TEST_NPIS = {
    "individual_1": generate_valid_npi("123456789"),  # 1234567893
    "individual_2": generate_valid_npi("123456790"),  # 1234567908
    "individual_3": generate_valid_npi("123456791"),  # 1234567916
    "individual_4": generate_valid_npi("123456792"),  # 1234567924
    "individual_5": generate_valid_npi("123456793"),  # 1234567932
    "group_1": generate_valid_group_npi("098765432"),  # 0987654328
    "group_2": generate_valid_group_npi("098765433"),  # 0987654336
    "group_3": generate_valid_group_npi("098765434"),  # 0987654344
}
