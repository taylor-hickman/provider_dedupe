"""Configuration management for the deduplication system."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BlockingRule(BaseModel):
    """Configuration for a Splink blocking rule."""

    rule: str = Field(..., description="SQL blocking rule expression")
    description: Optional[str] = Field(None, description="Human-readable description")

    @field_validator("rule")
    @classmethod
    def validate_rule_not_empty(cls, value: str) -> str:
        """Validate that rule is not empty."""
        if not value or not value.strip():
            raise ValueError("Blocking rule cannot be empty")
        return value.strip()


class ComparisonConfig(BaseModel):
    """Configuration for field comparison in Splink."""

    column_name: str = Field(..., description="Column name to compare")
    comparison_type: str = Field(..., description="Type of comparison to use")
    term_frequency_adjustments: bool = Field(
        default=False, description="Whether to apply term frequency adjustments"
    )
    thresholds: Optional[List[float]] = Field(
        None, description="Thresholds for distance-based comparisons"
    )

    @field_validator("column_name")
    @classmethod
    def validate_column_name_not_empty(cls, value: str) -> str:
        """Validate that column_name is not empty."""
        if not value or not value.strip():
            raise ValueError("Column name cannot be empty")
        return value.strip()

    @field_validator("comparison_type")
    @classmethod
    def validate_comparison_type_not_empty(cls, value: str) -> str:
        """Validate that comparison_type is not empty."""
        if not value or not value.strip():
            raise ValueError("Comparison type cannot be empty")
        return value.strip()


class DeduplicationConfig(BaseModel):
    """Main configuration for deduplication process."""

    # Splink configuration
    link_type: str = Field(default="dedupe_only", description="Type of record linkage")
    blocking_rules: List[BlockingRule] = Field(
        default_factory=list, description="Blocking rules for candidate generation"
    )
    comparisons: List[ComparisonConfig] = Field(
        default_factory=list, description="Field comparison configurations"
    )
    max_iterations: int = Field(
        default=20, description="Maximum EM algorithm iterations"
    )
    em_convergence: float = Field(default=0.001, description="EM convergence threshold")

    # Deduplication parameters
    match_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Probability threshold for matching",
    )
    min_cluster_size: int = Field(
        default=2, ge=2, description="Minimum cluster size to consider"
    )

    # Performance settings
    max_pairs_for_training: int = Field(
        default=1_000_000, description="Maximum pairs for EM training"
    )
    chunk_size: int = Field(
        default=10_000, description="Chunk size for batch processing"
    )
    use_parallel: bool = Field(default=True, description="Enable parallel processing")

    @classmethod
    def from_file(cls, config_path: Path) -> "DeduplicationConfig":
        """Load configuration from JSON or YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            DeduplicationConfig instance

        Raises:
            ValueError: If file format is not supported
        """
        import json

        if config_path.suffix == ".json":
            with open(config_path) as f:
                data = json.load(f)
        elif config_path.suffix in (".yaml", ".yml"):
            import yaml

            with open(config_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls(**data)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PROVIDER_DEDUPE_",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = Field(default="provider-dedupe", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file: Optional[Path] = Field(None, description="Log file path")

    # Data paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    temp_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "temp", description="Temporary directory"
    )

    # Database
    database_url: Optional[str] = Field(None, description="Database connection string")

    # Performance
    max_workers: int = Field(default=4, description="Maximum worker threads")
    memory_limit_gb: float = Field(default=8.0, description="Memory limit in gigabytes")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {value}")
        return value.upper()

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.model_dump(exclude_none=True)
