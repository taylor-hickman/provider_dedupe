"""Custom exceptions for the provider deduplication system."""

from typing import Any, Dict, Optional


class ProviderDedupeError(Exception):
    """Base exception for all provider deduplication errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DataValidationError(ProviderDedupeError):
    """Raised when data validation fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize data validation error."""
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)


class ConfigurationError(ProviderDedupeError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize configuration error."""
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)


class DeduplicationError(ProviderDedupeError):
    """Raised when deduplication process fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize deduplication error."""
        super().__init__(message, error_code="DEDUPLICATION_ERROR", **kwargs)


class TrainingError(ProviderDedupeError):
    """Raised when model training fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize training error."""
        super().__init__(message, error_code="TRAINING_ERROR", **kwargs)


class DataLoadError(ProviderDedupeError):
    """Raised when data loading fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize data load error."""
        super().__init__(message, error_code="DATA_LOAD_ERROR", **kwargs)


class OutputError(ProviderDedupeError):
    """Raised when output generation fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize output error."""
        super().__init__(message, error_code="OUTPUT_ERROR", **kwargs)