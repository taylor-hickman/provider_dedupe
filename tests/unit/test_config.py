"""Unit tests for configuration management."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from provider_dedupe.core.config import (
    BlockingRule,
    ComparisonConfig,
    DeduplicationConfig,
    Settings,
)


class TestBlockingRule:
    """Test cases for BlockingRule model."""

    def test_valid_blocking_rule(self):
        """Test creating a valid blocking rule."""
        rule = BlockingRule(rule="l.npi = r.npi", description="Exact NPI match")

        assert rule.rule == "l.npi = r.npi"
        assert rule.description == "Exact NPI match"

    def test_blocking_rule_without_description(self):
        """Test blocking rule without description."""
        rule = BlockingRule(rule="l.npi = r.npi")

        assert rule.rule == "l.npi = r.npi"
        assert rule.description is None

    def test_empty_rule_validation(self):
        """Test validation of empty rule."""
        with pytest.raises(ValidationError):
            BlockingRule(rule="")


class TestComparisonConfig:
    """Test cases for ComparisonConfig model."""

    def test_valid_comparison_config(self):
        """Test creating a valid comparison configuration."""
        config = ComparisonConfig(
            column_name="npi",
            comparison_type="exact",
            term_frequency_adjustments=True,
            thresholds=[0.8, 0.9],
        )

        assert config.column_name == "npi"
        assert config.comparison_type == "exact"
        assert config.term_frequency_adjustments is True
        assert config.thresholds == [0.8, 0.9]

    def test_comparison_config_defaults(self):
        """Test default values for comparison configuration."""
        config = ComparisonConfig(column_name="name", comparison_type="fuzzy")

        assert config.term_frequency_adjustments is False
        assert config.thresholds is None

    def test_empty_column_name_validation(self):
        """Test validation of empty column name."""
        with pytest.raises(ValidationError):
            ComparisonConfig(column_name="", comparison_type="exact")


class TestDeduplicationConfig:
    """Test cases for DeduplicationConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeduplicationConfig()

        assert config.link_type == "dedupe_only"
        assert config.match_threshold == 0.95
        assert config.max_iterations == 20
        assert config.em_convergence == 0.001
        assert config.min_cluster_size == 2
        assert config.use_parallel is True
        assert isinstance(config.blocking_rules, list)
        assert isinstance(config.comparisons, list)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeduplicationConfig(
            match_threshold=0.90, max_iterations=15, min_cluster_size=3
        )

        assert config.match_threshold == 0.90
        assert config.max_iterations == 15
        assert config.min_cluster_size == 3

    def test_threshold_validation(self):
        """Test match threshold validation."""
        # Valid thresholds
        config = DeduplicationConfig(match_threshold=0.5)
        assert config.match_threshold == 0.5

        config = DeduplicationConfig(match_threshold=1.0)
        assert config.match_threshold == 1.0

        # Invalid thresholds
        with pytest.raises(ValidationError):
            DeduplicationConfig(match_threshold=-0.1)

        with pytest.raises(ValidationError):
            DeduplicationConfig(match_threshold=1.1)

    def test_min_cluster_size_validation(self):
        """Test minimum cluster size validation."""
        # Valid size
        config = DeduplicationConfig(min_cluster_size=2)
        assert config.min_cluster_size == 2

        # Invalid size (too small)
        with pytest.raises(ValidationError):
            DeduplicationConfig(min_cluster_size=1)

    def test_from_file_json(self, tmp_path: Path):
        """Test loading configuration from JSON file."""
        config_data = {
            "match_threshold": 0.90,
            "max_iterations": 15,
            "blocking_rules": [{"rule": "l.npi = r.npi", "description": "NPI match"}],
            "comparisons": [{"column_name": "npi", "comparison_type": "exact"}],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = DeduplicationConfig.from_file(config_file)

        assert config.match_threshold == 0.90
        assert config.max_iterations == 15
        assert len(config.blocking_rules) == 1
        assert len(config.comparisons) == 1

    def test_from_file_yaml(self, tmp_path: Path):
        """Test loading configuration from YAML file."""
        import yaml

        config_data = {
            "match_threshold": 0.85,
            "max_iterations": 25,
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = DeduplicationConfig.from_file(config_file)

        assert config.match_threshold == 0.85
        assert config.max_iterations == 25

    def test_from_file_unsupported_format(self, tmp_path: Path):
        """Test error handling for unsupported file format."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config format"):
            DeduplicationConfig.from_file(config_file)

    def test_from_file_nonexistent(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DeduplicationConfig.from_file(Path("nonexistent.json"))


class TestSettings:
    """Test cases for Settings model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.app_name == "provider-dedupe"
        assert settings.version == "1.0.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.max_workers == 4
        assert settings.memory_limit_gb == 8.0

    @patch.dict("os.environ", {"PROVIDER_DEDUPE_DEBUG": "true"})
    def test_environment_variable_loading(self):
        """Test loading settings from environment variables."""
        settings = Settings()

        assert settings.debug is True

    @patch.dict("os.environ", {"PROVIDER_DEDUPE_LOG_LEVEL": "DEBUG"})
    def test_log_level_from_env(self):
        """Test log level from environment variable."""
        settings = Settings()

        assert settings.log_level == "DEBUG"

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_get_config_dict(self):
        """Test configuration dictionary export."""
        settings = Settings(app_name="test-app", debug=True, log_level="DEBUG")

        config_dict = settings.get_config_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "test-app"
        assert config_dict["debug"] is True
        assert config_dict["log_level"] == "DEBUG"

    def test_path_fields(self, tmp_path: Path):
        """Test Path field handling."""
        settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "output")

        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.output_dir, Path)
        assert settings.data_dir == tmp_path / "data"
        assert settings.output_dir == tmp_path / "output"

    @patch.dict("os.environ", {"PROVIDER_DEDUPE_MAX_WORKERS": "8"})
    def test_integer_field_from_env(self):
        """Test integer field loading from environment."""
        settings = Settings()

        assert settings.max_workers == 8

    @patch.dict("os.environ", {"PROVIDER_DEDUPE_MEMORY_LIMIT_GB": "16.5"})
    def test_float_field_from_env(self):
        """Test float field loading from environment."""
        settings = Settings()

        assert settings.memory_limit_gb == 16.5

    def test_case_insensitive_env_vars(self):
        """Test that environment variables are case insensitive."""
        with patch.dict("os.environ", {"provider_dedupe_debug": "true"}):
            settings = Settings()
            assert settings.debug is True


class TestConfigIntegration:
    """Integration tests for configuration components."""

    def test_complete_config_creation(self, tmp_path: Path):
        """Test creating a complete configuration with all components."""
        config_data = {
            "link_type": "dedupe_only",
            "match_threshold": 0.92,
            "max_iterations": 18,
            "em_convergence": 0.005,
            "min_cluster_size": 3,
            "blocking_rules": [
                {"rule": "l.npi = r.npi", "description": "Exact NPI match"},
                {
                    "rule": "l.given_name = r.given_name AND l.family_name = r.family_name",
                    "description": "Full name match",
                },
            ],
            "comparisons": [
                {
                    "column_name": "npi",
                    "comparison_type": "exact",
                    "term_frequency_adjustments": False,
                },
                {
                    "column_name": "given_name",
                    "comparison_type": "name",
                    "term_frequency_adjustments": True,
                },
                {
                    "column_name": "address",
                    "comparison_type": "levenshtein",
                    "thresholds": [1, 2, 3],
                },
            ],
        }

        config_file = tmp_path / "complete_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = DeduplicationConfig.from_file(config_file)

        assert config.match_threshold == 0.92
        assert len(config.blocking_rules) == 2
        assert len(config.comparisons) == 3

        # Check blocking rules
        assert config.blocking_rules[0].rule == "l.npi = r.npi"
        assert config.blocking_rules[0].description == "Exact NPI match"

        # Check comparisons
        npi_comparison = config.comparisons[0]
        assert npi_comparison.column_name == "npi"
        assert npi_comparison.comparison_type == "exact"

        address_comparison = config.comparisons[2]
        assert address_comparison.thresholds == [1, 2, 3]
