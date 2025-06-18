"""Provider Deduplication Package.

A cli tool for deduplicating healthcare provider records
using probabilistic record linkage with the Splink library.
"""

__version__ = "1.0.0"
__author__ = "Your Organization"
__email__ = "contact@example.com"

from provider_dedupe.core.deduplicator import ProviderDeduplicator
from provider_dedupe.models.provider import Provider, ProviderRecord

__all__ = ["ProviderDeduplicator", "Provider", "ProviderRecord"]
