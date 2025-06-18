"""Structured logging configuration for production use."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False,
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_logs: Whether to output logs in JSON format

    Example:
        >>> setup_logging("DEBUG", Path("app.log"), json_logs=True)
    """
    # Configure Python's logging
    handlers: List[Union[logging.StreamHandler[Any], logging.FileHandler]] = [
        logging.StreamHandler(sys.stdout)
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format="%(message)s",
    )

    # Configure structlog
    processors: List[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", record_count=1000)
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary log context.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, request_id="123", user="john"):
        ...     logger.info("Processing request")
        ...     # Logs will include request_id and user fields
    """

    def __init__(self, logger: structlog.BoundLogger, **context: Any) -> None:
        """Initialize log context.

        Args:
            logger: Logger instance
            **context: Key-value pairs to add to log context
        """
        self.logger = logger
        self.context = context
        self._previous_context: Dict[str, Any] = {}

    def __enter__(self) -> structlog.BoundLogger:
        """Enter context and bind additional fields."""
        for key, value in self.context.items():
            self._previous_context[key] = getattr(self.logger._context, key, None)
        self.logger = self.logger.bind(**self.context)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore previous state."""
        for key in self.context:
            if self._previous_context[key] is None:
                self.logger = self.logger.unbind(key)
            else:
                self.logger = self.logger.bind(**{key: self._previous_context[key]})
