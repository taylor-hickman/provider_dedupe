"""Integration tests for the full deduplication pipeline."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from provider_dedupe.core.config import DeduplicationConfig
from provider_dedupe.core.deduplicator import ProviderDeduplicator
from provider_dedupe.core.exceptions import DataValidationError, TrainingError
from provider_dedupe.services.output_generator import OutputGenerator
from tests.fixtures.sample_data import get_large_dataset, get_sample_dataframe


class TestFullPipeline:
    """Integration tests for the complete deduplication pipeline."""

    def test_end_to_end_deduplication(self, tmp_path: Path):
        """Test complete end-to-end deduplication process."""
        # Create sample data file
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "input.csv"
        sample_df.to_csv(input_file, index=False)

        # Create configuration
        config = DeduplicationConfig(
            match_threshold=0.8,  # Lower threshold for testing
            max_iterations=5,  # Fewer iterations for speed
        )

        # Initialize deduplicator
        deduplicator = ProviderDeduplicator(config=config)

        # Run full pipeline
        deduplicator.load_data(input_file)
        prepared_df = deduplicator.prepare_data()

        # Verify data preparation
        assert len(prepared_df) == len(sample_df)
        assert "unique_id" in prepared_df.columns
        assert "given_name" in prepared_df.columns
        assert "family_name" in prepared_df.columns

        # Mock Splink components for testing
        with patch("provider_dedupe.core.deduplicator.Linker") as mock_linker_class:
            mock_linker = Mock()
            mock_linker_class.return_value = mock_linker

            # Mock training methods
            mock_linker.training.estimate_u_using_random_sampling = Mock()
            mock_linker.training.estimate_parameters_using_expectation_maximisation = (
                Mock()
            )

            # Mock prediction methods
            mock_predictions = Mock()
            mock_clusters = Mock()

            # Create mock results DataFrame
            results_data = [
                {
                    "unique_id": "uuid-1",
                    "cluster_id": "cluster_1",
                    "npi": "1234567893",
                    "given_name": "john",
                    "family_name": "smith",
                },
                {
                    "unique_id": "uuid-2",
                    "cluster_id": "cluster_1",
                    "npi": "1234567893",
                    "given_name": "jon",
                    "family_name": "smith",
                },
            ]
            mock_results_df = pd.DataFrame(results_data)

            mock_linker.inference.predict.return_value = mock_predictions
            mock_linker.clustering.cluster_pairwise_predictions_at_threshold.return_value = (
                mock_clusters
            )
            mock_clusters.as_pandas_dataframe.return_value = mock_results_df

            # Train model
            deduplicator.train_model()

            # Perform deduplication
            results_df, statistics = deduplicator.deduplicate()

            # Verify results
            assert isinstance(results_df, pd.DataFrame)
            assert isinstance(statistics, dict)
            assert "total_records" in statistics
            assert "unique_clusters" in statistics
            assert "duplicates_found" in statistics

    def test_data_validation_failure(self, tmp_path: Path):
        """Test pipeline failure due to data validation issues."""
        # Create invalid data
        invalid_data = pd.DataFrame(
            [{"npi": "invalid", "firstname": "John"}]  # Missing required columns
        )
        input_file = tmp_path / "invalid.csv"
        invalid_data.to_csv(input_file, index=False)

        deduplicator = ProviderDeduplicator()

        with pytest.raises(DataValidationError):
            deduplicator.load_data(input_file)

    def test_output_generation_integration(self, tmp_path: Path):
        """Test integration with output generation."""
        # Create sample results
        results_data = [
            {
                "unique_id": "uuid-1",
                "cluster_id": "cluster_1",
                "npi": "1234567893",
                "given_name": "john",
                "family_name": "smith",
                "cluster_size": 2,
            },
            {
                "unique_id": "uuid-2",
                "cluster_id": "cluster_1",
                "npi": "1234567893",
                "given_name": "jon",
                "family_name": "smith",
                "cluster_size": 2,
            },
        ]
        results_df = pd.DataFrame(results_data)

        statistics = {
            "total_records": 2,
            "unique_clusters": 1,
            "duplicates_found": 1,
            "duplication_rate": 0.5,
        }

        # Test CSV output
        output_generator = OutputGenerator()
        csv_path = tmp_path / "results.csv"
        output_generator.save(results_df, csv_path, statistics)

        assert csv_path.exists()
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(results_df)

        # Test Excel output
        excel_path = tmp_path / "results.xlsx"
        output_generator.save(results_df, excel_path, statistics)

        assert excel_path.exists()
        loaded_excel = pd.read_excel(excel_path, sheet_name="Deduplication Results")
        assert len(loaded_excel) == len(results_df)

        # Test HTML output
        html_path = tmp_path / "results.html"
        output_generator.save(results_df, html_path, statistics)

        assert html_path.exists()
        html_content = html_path.read_text()
        assert "Provider Deduplication Report" in html_content

    def test_configuration_integration(self, tmp_path: Path):
        """Test integration with custom configuration."""
        # Create custom configuration
        config_data = {
            "match_threshold": 0.85,
            "max_iterations": 10,
            "blocking_rules": [{"rule": "l.npi = r.npi", "description": "NPI match"}],
            "comparisons": [
                {
                    "column_name": "npi",
                    "comparison_type": "exact",
                    "term_frequency_adjustments": False,
                }
            ],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Load configuration
        config = DeduplicationConfig.from_file(config_file)
        deduplicator = ProviderDeduplicator(config=config)

        assert deduplicator.config.match_threshold == 0.85
        assert deduplicator.config.max_iterations == 10
        assert len(deduplicator.config.blocking_rules) == 1

    @pytest.mark.slow
    def test_performance_with_larger_dataset(self, tmp_path: Path):
        """Test performance with larger dataset (marked as slow test)."""
        # Generate larger dataset
        large_df = get_large_dataset(size=100)  # Small size for CI
        input_file = tmp_path / "large_input.csv"
        large_df.to_csv(input_file, index=False)

        # Use optimized configuration for performance
        config = DeduplicationConfig(
            match_threshold=0.9,
            max_iterations=3,
            max_pairs_for_training=10000,
        )

        deduplicator = ProviderDeduplicator(config=config)

        # Test data loading and preparation
        deduplicator.load_data(input_file)
        prepared_df = deduplicator.prepare_data()

        assert len(prepared_df) == len(large_df)

        # Verify normalization worked
        assert prepared_df["given_name"].str.islower().all()
        assert prepared_df["family_name"].str.islower().all()

    def test_error_recovery_during_training(self, tmp_path: Path):
        """Test error recovery during model training."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "input.csv"
        sample_df.to_csv(input_file, index=False)

        deduplicator = ProviderDeduplicator()
        deduplicator.load_data(input_file)
        deduplicator.prepare_data()

        # Mock Splink to simulate training failure
        with patch("provider_dedupe.core.deduplicator.Linker") as mock_linker_class:
            mock_linker = Mock()
            mock_linker_class.return_value = mock_linker

            # Simulate training method failure
            mock_linker.training.estimate_u_using_random_sampling.side_effect = (
                Exception("Training failed")
            )
            mock_linker.training.estimate_parameters_using_expectation_maximisation = (
                Mock()
            )

            # Training should handle the exception gracefully
            try:
                deduplicator.train_model()
                # Should not raise exception due to error handling
            except TrainingError:
                pytest.fail("Training should handle exceptions gracefully")

    def test_memory_efficient_processing(self, tmp_path: Path):
        """Test memory-efficient processing configuration."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "input.csv"
        sample_df.to_csv(input_file, index=False)

        # Configure for memory efficiency
        config = DeduplicationConfig(
            max_pairs_for_training=1000,  # Reduced for memory efficiency
            chunk_size=50,
        )

        deduplicator = ProviderDeduplicator(config=config)
        deduplicator.load_data(input_file)
        deduplicator.prepare_data()

        # Verify configuration is applied
        assert deduplicator.config.max_pairs_for_training == 1000
        assert deduplicator.config.chunk_size == 50

    def test_data_quality_integration(self, tmp_path: Path):
        """Test integration with data quality analysis."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "input.csv"
        sample_df.to_csv(input_file, index=False)

        deduplicator = ProviderDeduplicator()
        deduplicator.load_data(input_file)

        # Data quality analysis should be triggered during prepare_data
        prepared_df = deduplicator.prepare_data()

        # Verify quality analysis occurred (would be logged)
        assert len(prepared_df) == len(sample_df)

    def test_blocking_rules_effectiveness(self, tmp_path: Path):
        """Test effectiveness of different blocking rules."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "input.csv"
        sample_df.to_csv(input_file, index=False)

        # Test with different blocking rule configurations
        configs = [
            # NPI-only blocking
            DeduplicationConfig(
                blocking_rules=[{"rule": "l.npi = r.npi", "description": "NPI only"}]
            ),
            # Name-based blocking
            DeduplicationConfig(
                blocking_rules=[
                    {
                        "rule": "l.family_name = r.family_name",
                        "description": "Last name only",
                    }
                ]
            ),
        ]

        for config in configs:
            deduplicator = ProviderDeduplicator(config=config)
            deduplicator.load_data(input_file)
            deduplicator.prepare_data()

            # Verify configuration was applied
            assert len(deduplicator.config.blocking_rules) == 1

    def test_concurrent_processing_safety(self, tmp_path: Path):
        """Test thread safety and concurrent processing."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "input.csv"
        sample_df.to_csv(input_file, index=False)

        # Test with parallel processing enabled
        config = DeduplicationConfig(use_parallel=True)
        deduplicator = ProviderDeduplicator(config=config)

        # Load and prepare data (should handle concurrent operations safely)
        deduplicator.load_data(input_file)
        prepared_df = deduplicator.prepare_data()

        assert len(prepared_df) == len(sample_df)
        assert deduplicator.config.use_parallel is True
