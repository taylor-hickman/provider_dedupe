"""Integration tests for CLI functionality."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from provider_dedupe.cli.main import cli
from tests.fixtures.sample_data import get_sample_dataframe


class TestCLIIntegration:
    """Integration tests for command-line interface."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "Provider Deduplication CLI" in result.output
        assert "dedupe" in result.output
        assert "analyze" in result.output

    def test_init_config_command(self, tmp_path: Path):
        """Test init-config command."""
        runner = CliRunner()
        config_file = tmp_path / "test_config.json"
        
        result = runner.invoke(cli, ["init-config", "--output-file", str(config_file)])
        
        assert result.exit_code == 0
        assert config_file.exists()
        
        # Verify config file content
        with open(config_file) as f:
            config_data = json.load(f)
        
        assert "match_threshold" in config_data
        assert "max_iterations" in config_data

    def test_analyze_command(self, tmp_path: Path):
        """Test analyze command."""
        # Create sample data file
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "reports"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            str(input_file),
            "--output-dir", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Data Quality Analysis Complete" in result.output
        assert output_dir.exists()

    @patch('provider_dedupe.core.deduplicator.Linker')
    def test_dedupe_command_basic(self, mock_linker_class, tmp_path: Path):
        """Test basic dedupe command."""
        # Setup mock
        mock_linker = Mock()
        mock_linker_class.return_value = mock_linker
        
        # Mock training methods
        mock_linker.training.estimate_u_using_random_sampling = Mock()
        mock_linker.training.estimate_parameters_using_expectation_maximisation = Mock()
        
        # Mock prediction methods
        mock_predictions = Mock()
        mock_clusters = Mock()
        
        results_data = [
            {"unique_id": "1", "cluster_id": "cluster_1", "npi": "1234567893"},
            {"unique_id": "2", "cluster_id": "cluster_2", "npi": "1234567901"},
        ]
        mock_results_df = pd.DataFrame(results_data)
        
        mock_linker.inference.predict.return_value = mock_predictions
        mock_linker.clustering.cluster_pairwise_predictions_at_threshold.return_value = mock_clusters
        mock_clusters.as_pandas_dataframe.return_value = mock_results_df
        
        # Create sample data file
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        output_file = tmp_path / "results.csv"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dedupe",
            str(input_file),
            str(output_file),
            "--threshold", "0.9"
        ])
        
        assert result.exit_code == 0
        assert "Deduplication completed successfully" in result.output
        assert output_file.exists()

    @patch('provider_dedupe.core.deduplicator.Linker')
    def test_dedupe_command_with_config(self, mock_linker_class, tmp_path: Path):
        """Test dedupe command with custom configuration."""
        # Setup mock (same as above)
        mock_linker = Mock()
        mock_linker_class.return_value = mock_linker
        mock_linker.training.estimate_u_using_random_sampling = Mock()
        mock_linker.training.estimate_parameters_using_expectation_maximisation = Mock()
        
        mock_predictions = Mock()
        mock_clusters = Mock()
        mock_results_df = pd.DataFrame([{"unique_id": "1", "cluster_id": "cluster_1"}])
        
        mock_linker.inference.predict.return_value = mock_predictions
        mock_linker.clustering.cluster_pairwise_predictions_at_threshold.return_value = mock_clusters
        mock_clusters.as_pandas_dataframe.return_value = mock_results_df
        
        # Create config file
        config_data = {
            "match_threshold": 0.85,
            "max_iterations": 15,
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        # Create sample data file
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        output_file = tmp_path / "results.xlsx"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dedupe",
            str(input_file),
            str(output_file),
            "--config", str(config_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()

    @patch('provider_dedupe.core.deduplicator.Linker')
    def test_dedupe_command_with_report(self, mock_linker_class, tmp_path: Path):
        """Test dedupe command with HTML report generation."""
        # Setup mock
        mock_linker = Mock()
        mock_linker_class.return_value = mock_linker
        mock_linker.training.estimate_u_using_random_sampling = Mock()
        mock_linker.training.estimate_parameters_using_expectation_maximisation = Mock()
        
        mock_predictions = Mock()
        mock_clusters = Mock()
        mock_results_df = pd.DataFrame([
            {"unique_id": "1", "cluster_id": "cluster_1", "cluster_size": 2},
            {"unique_id": "2", "cluster_id": "cluster_1", "cluster_size": 2},
        ])
        
        mock_linker.inference.predict.return_value = mock_predictions
        mock_linker.clustering.cluster_pairwise_predictions_at_threshold.return_value = mock_clusters
        mock_clusters.as_pandas_dataframe.return_value = mock_results_df
        
        # Create sample data file
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        output_file = tmp_path / "results.csv"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dedupe",
            str(input_file),
            str(output_file),
            "--generate-report"
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Check that HTML report was generated
        html_file = tmp_path / "results_report.html"
        assert html_file.exists()

    def test_dedupe_command_file_not_found(self):
        """Test dedupe command with nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dedupe",
            "nonexistent.csv",
            "output.csv"
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_analyze_command_file_not_found(self):
        """Test analyze command with nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            "nonexistent.csv"
        ])
        
        assert result.exit_code != 0

    def test_visualize_command_basic(self, tmp_path: Path):
        """Test visualize command."""
        # Create sample results file
        results_data = [
            {"cluster_id": "cluster_1", "cluster_size": 2},
            {"cluster_id": "cluster_2", "cluster_size": 1},
        ]
        results_df = pd.DataFrame(results_data)
        results_file = tmp_path / "results.csv"
        results_df.to_csv(results_file, index=False)
        
        output_dir = tmp_path / "viz"
        
        # Mock matplotlib to avoid actual plotting
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.hist'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                "visualize",
                str(results_file),
                "--output-dir", str(output_dir)
            ])
            
            if result.exit_code != 0:
                print(f"CLI command failed with exit code {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0
            assert "Visualizations saved" in result.output

    def test_visualize_command_missing_dependencies(self, tmp_path: Path):
        """Test visualize command with missing dependencies."""
        # Create sample results file
        results_data = [{"cluster_id": "cluster_1"}]
        results_df = pd.DataFrame(results_data)
        results_file = tmp_path / "results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Mock missing matplotlib
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
            runner = CliRunner()
            result = runner.invoke(cli, [
                "visualize",
                str(results_file)
            ])
            
            # Should handle missing dependencies gracefully
            assert "Visualization dependencies not installed" in result.output or result.exit_code != 0

    def test_cli_debug_logging(self, tmp_path: Path):
        """Test CLI with debug logging enabled."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--debug",
            "analyze",
            str(input_file)
        ])
        
        # Debug flag should be processed without error
        assert result.exit_code == 0

    def test_cli_log_file_option(self, tmp_path: Path):
        """Test CLI with log file option."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        log_file = tmp_path / "test.log"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--log-file", str(log_file),
            "analyze",
            str(input_file)
        ])
        
        assert result.exit_code == 0
        # Log file should be created (depending on logging setup)

    def test_cli_version_option(self):
        """Test CLI version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid-command"])
        
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_dedupe_command_parameter_validation(self, tmp_path: Path):
        """Test parameter validation in dedupe command."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        output_file = tmp_path / "results.csv"
        
        # Test invalid threshold
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dedupe",
            str(input_file),
            str(output_file),
            "--threshold", "1.5"  # Invalid threshold > 1.0
        ])
        
        # Should handle validation error
        assert result.exit_code != 0

    def test_context_object_handling(self, tmp_path: Path):
        """Test CLI context object handling."""
        sample_df = get_sample_dataframe()
        input_file = tmp_path / "test_input.csv"
        sample_df.to_csv(input_file, index=False)
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze",
            str(input_file)
        ])
        
        # Context should be properly initialized
        assert result.exit_code == 0