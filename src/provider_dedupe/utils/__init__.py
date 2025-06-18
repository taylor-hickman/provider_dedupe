"""Utility modules for provider deduplication."""

from provider_dedupe.utils.logging import get_logger, setup_logging
from provider_dedupe.utils.normalization import TextNormalizer

__all__ = ["get_logger", "setup_logging", "TextNormalizer"]
