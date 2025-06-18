"""Core deduplication functionality."""

from provider_dedupe.core.config import DeduplicationConfig, Settings
from provider_dedupe.core.deduplicator import ProviderDeduplicator

__all__ = ["ProviderDeduplicator", "DeduplicationConfig", "Settings"]
