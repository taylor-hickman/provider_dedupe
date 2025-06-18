"""Text normalization utilities for data preprocessing."""

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Optional, Pattern

try:
    from text2num import alpha2digit

    HAS_TEXT2NUM = True
except ImportError:
    HAS_TEXT2NUM = False

    def alpha2digit(text: str, lang: str) -> str:
        """Fallback when text2num is not available."""
        return text


class NormalizationStrategy(ABC):
    """Abstract base class for normalization strategies."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize the input text.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        pass


class TextNormalizer:
    """Configurable text normalization with strategy pattern.

    This class provides various text normalization strategies for
    different types of data (names, addresses, phone numbers, etc.).
    """

    # Compiled regex patterns for performance
    _NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
    _MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
    _PHONE_PATTERN = re.compile(r"[^\d]")
    _POSTAL_CODE_PATTERN = re.compile(r"[^\d-]")

    # Abbreviation mappings
    NAME_ABBREVIATIONS: Dict[str, str] = {
        "jr": "junior",
        "sr": "senior",
        "dr": "doctor",
        "mr": "mister",
        "mrs": "missus",
        "ms": "miss",
        "prof": "professor",
        "rev": "reverend",
        "hon": "honorable",
        "st": "saint",
        "capt": "captain",
        "lt": "lieutenant",
        "col": "colonel",
        "gen": "general",
        "maj": "major",
        "sgt": "sergeant",
        "cpl": "corporal",
        "pvt": "private",
    }

    ADDRESS_ABBREVIATIONS: Dict[str, str] = {
        "st": "street",
        "ave": "avenue",
        "blvd": "boulevard",
        "rd": "road",
        "dr": "drive",
        "ln": "lane",
        "pl": "place",
        "pkwy": "parkway",
        "cir": "circle",
        "ct": "court",
        "ter": "terrace",
        "terr": "terrace",
        "hwy": "highway",
        "fwy": "freeway",
        "apt": "apartment",
        "bldg": "building",
        "fl": "floor",
        "ste": "suite",
        "rm": "room",
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west",
        "ne": "northeast",
        "nw": "northwest",
        "se": "southeast",
        "sw": "southwest",
    }

    # State abbreviations
    STATE_ABBREVIATIONS: Dict[str, str] = {
        "alabama": "al",
        "alaska": "ak",
        "arizona": "az",
        "arkansas": "ar",
        "california": "ca",
        "colorado": "co",
        "connecticut": "ct",
        "delaware": "de",
        "florida": "fl",
        "georgia": "ga",
        "hawaii": "hi",
        "idaho": "id",
        "illinois": "il",
        "indiana": "in",
        "iowa": "ia",
        "kansas": "ks",
        "kentucky": "ky",
        "louisiana": "la",
        "maine": "me",
        "maryland": "md",
        "massachusetts": "ma",
        "michigan": "mi",
        "minnesota": "mn",
        "mississippi": "ms",
        "missouri": "mo",
        "montana": "mt",
        "nebraska": "ne",
        "nevada": "nv",
        "new hampshire": "nh",
        "new jersey": "nj",
        "new mexico": "nm",
        "new york": "ny",
        "north carolina": "nc",
        "north dakota": "nd",
        "ohio": "oh",
        "oklahoma": "ok",
        "oregon": "or",
        "pennsylvania": "pa",
        "rhode island": "ri",
        "south carolina": "sc",
        "south dakota": "sd",
        "tennessee": "tn",
        "texas": "tx",
        "utah": "ut",
        "vermont": "vt",
        "virginia": "va",
        "washington": "wa",
        "west virginia": "wv",
        "wisconsin": "wi",
        "wyoming": "wy",
        "district of columbia": "dc",
    }

    def __init__(self) -> None:
        """Initialize the text normalizer with compiled patterns."""
        self._name_pattern = self._compile_abbreviation_pattern(self.NAME_ABBREVIATIONS)
        self._address_pattern = self._compile_abbreviation_pattern(
            self.ADDRESS_ABBREVIATIONS
        )

    @staticmethod
    def _compile_abbreviation_pattern(abbreviations: Dict[str, str]) -> Pattern:
        """Compile abbreviation dictionary into regex pattern.

        Args:
            abbreviations: Dictionary of abbreviations to expansions

        Returns:
            Compiled regex pattern
        """
        # Sort by length descending to match longer patterns first
        sorted_abbrevs = sorted(abbreviations.keys(), key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(abbr) for abbr in sorted_abbrevs) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    @lru_cache(maxsize=10000)
    def normalize_name(self, name: str) -> str:
        """Normalize a person's name.

        Args:
            name: Input name to normalize

        Returns:
            Normalized name in lowercase with expanded abbreviations

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize_name("Dr. John Smith Jr.")
            'doctor john smith junior'
        """
        if not name:
            return ""

        # Convert to string and lowercase
        text = str(name).lower().strip()

        # Remove non-alphanumeric characters
        text = self._NON_ALPHANUMERIC_PATTERN.sub(" ", text)

        # Expand abbreviations
        text = self._name_pattern.sub(
            lambda m: self.NAME_ABBREVIATIONS.get(m.group(1).lower(), m.group(1)), text
        )

        # Clean up whitespace
        text = self._MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

        return text

    @lru_cache(maxsize=10000)
    def normalize_address(self, address: str) -> str:
        """Normalize an address string.

        Args:
            address: Input address to normalize

        Returns:
            Normalized address with expanded abbreviations and converted numbers

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize_address("123 Main St, Apt 5")
            '123 main street apartment 5'
        """
        if not address:
            return ""

        # Convert to string and lowercase
        text = str(address).lower().strip()

        # Convert written numbers to digits
        try:
            text = alpha2digit(text, "en")
        except (ValueError, KeyError):
            # If conversion fails, continue with original text
            pass

        # Remove non-alphanumeric characters
        text = self._NON_ALPHANUMERIC_PATTERN.sub(" ", text)

        # Expand abbreviations
        text = self._address_pattern.sub(
            lambda m: self.ADDRESS_ABBREVIATIONS.get(m.group(1).lower(), m.group(1)),
            text,
        )

        # Clean up whitespace
        text = self._MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

        return text

    @lru_cache(maxsize=10000)
    def normalize_phone(self, phone: str) -> str:
        """Normalize a phone number to digits only.

        Args:
            phone: Input phone number

        Returns:
            Phone number containing only digits

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize_phone("(555) 123-4567 ext 890")
            '5551234567890'
        """
        if not phone:
            return ""

        # Remove all non-digit characters
        return self._PHONE_PATTERN.sub("", str(phone))

    @lru_cache(maxsize=10000)
    def normalize_postal_code(self, postal_code: str) -> str:
        """Normalize a postal code.

        Args:
            postal_code: Input postal code

        Returns:
            Normalized postal code (5 digits for US ZIP codes)

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize_postal_code("12345-6789")
            '12345'
        """
        if not postal_code:
            return ""

        # Remove non-digit and non-hyphen characters
        cleaned = self._POSTAL_CODE_PATTERN.sub("", str(postal_code))

        # For US ZIP codes, return first 5 digits
        digits_only = "".join(filter(str.isdigit, cleaned))
        return digits_only[:5] if digits_only else ""

    @lru_cache(maxsize=10000)
    def normalize_state(self, state: str) -> str:
        """Normalize state name or abbreviation.

        Args:
            state: State name or abbreviation

        Returns:
            Two-letter state abbreviation in uppercase

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize_state("California")
            'CA'
            >>> normalizer.normalize_state("ca")
            'CA'
        """
        if not state:
            return ""

        state_lower = str(state).lower().strip()

        # If already 2-letter abbreviation
        if len(state_lower) == 2 and state_lower.isalpha():
            return state_lower.upper()

        # Look up full state name
        abbreviation = self.STATE_ABBREVIATIONS.get(state_lower, state_lower)
        return abbreviation.upper()

    def normalize_field(self, value: str, field_type: str) -> str:
        """Normalize a field based on its type.

        Args:
            value: Field value to normalize
            field_type: Type of field (name, address, phone, etc.)

        Returns:
            Normalized value

        Raises:
            ValueError: If field_type is not recognized
        """
        normalizers = {
            "name": self.normalize_name,
            "address": self.normalize_address,
            "phone": self.normalize_phone,
            "postal_code": self.normalize_postal_code,
            "state": self.normalize_state,
        }

        normalizer = normalizers.get(field_type)
        if not normalizer:
            raise ValueError(f"Unknown field type: {field_type}")

        return normalizer(value)
