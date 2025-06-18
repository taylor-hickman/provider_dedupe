"""Core deduplication engine using Splink."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
from splink import DuckDBAPI, Linker
from splink import comparison_library as cl

from provider_dedupe.core.config import ComparisonConfig, DeduplicationConfig
from provider_dedupe.core.exceptions import (
    DataValidationError,
    DeduplicationError,
    TrainingError,
)
from provider_dedupe.models.provider import ProviderRecord
from provider_dedupe.services.data_loader import DataLoader
from provider_dedupe.services.data_quality import DataQualityAnalyzer
from provider_dedupe.utils.logging import get_logger
from provider_dedupe.utils.normalization import TextNormalizer

logger = get_logger(__name__)


class ProviderDeduplicator:
    """Main deduplication engine for provider records.

    This class orchestrates the entire deduplication process including
    data loading, preprocessing, model training, and clustering.

    Attributes:
        config: Deduplication configuration
        normalizer: Text normalization utility
        data_loader: Data loading service
        quality_analyzer: Data quality analysis service
    """

    def __init__(
        self,
        config: Optional[DeduplicationConfig] = None,
        data_loader: Optional[DataLoader] = None,
        quality_analyzer: Optional[DataQualityAnalyzer] = None,
    ) -> None:
        """Initialize the deduplicator.

        Args:
            config: Deduplication configuration (uses defaults if None)
            data_loader: Data loader service (creates default if None)
            quality_analyzer: Quality analyzer service (creates default if None)
        """
        self.config = config or DeduplicationConfig()
        self.normalizer = TextNormalizer()
        self.data_loader = data_loader or DataLoader()
        self.quality_analyzer = quality_analyzer or DataQualityAnalyzer()

        self._linker: Optional[Linker] = None
        self._df: Optional[pd.DataFrame] = None
        self._prepared_df: Optional[pd.DataFrame] = None

    def load_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate input data.

        Args:
            input_path: Path to input data file

        Returns:
            Loaded DataFrame

        Raises:
            DataValidationError: If data validation fails
        """
        logger.info("Loading data", path=str(input_path))

        # Load data
        self._df = self.data_loader.load(input_path)
        logger.info("Data loaded", record_count=len(self._df))

        # Validate data
        validation_errors = self._validate_data(self._df)
        if validation_errors:
            raise DataValidationError(
                f"Data validation failed: {'; '.join(validation_errors)}"
            )

        return self._df

    def prepare_data(self) -> pd.DataFrame:
        """Prepare data for deduplication.

        Returns:
            Prepared DataFrame ready for Splink

        Raises:
            DeduplicationError: If data hasn't been loaded
        """
        if self._df is None:
            raise DeduplicationError("No data loaded. Call load_data() first.")

        logger.info("Preparing data for deduplication")

        # Create provider records
        records = []
        for _, row in self._df.iterrows():
            try:
                record = ProviderRecord.from_dict(row.to_dict())
                records.append(record)
            except Exception as e:
                logger.warning(
                    "Failed to create provider record",
                    error=str(e),
                    row_index=row.name,
                )

        # Convert to normalized DataFrame
        self._prepared_df = self._normalize_dataframe(records)

        # Generate quality report
        quality_report = self.quality_analyzer.analyze(self._prepared_df)
        logger.info(
            "Data quality analysis complete",
            total_records=quality_report.total_records,
            missing_values=len(quality_report.missing_value_summary),
        )

        return self._prepared_df

    def _normalize_dataframe(self, records: List[ProviderRecord]) -> pd.DataFrame:
        """Normalize provider records into DataFrame for Splink.

        Args:
            records: List of provider records

        Returns:
            Normalized DataFrame
        """
        data = []
        for record in records:
            provider = record.provider
            normalized_row = {
                "unique_id": str(record.unique_id),
                "npi": provider.npi,
                "group_npi": provider.group_npi or "",
                "group_name": self.normalizer.normalize_name(provider.group_name or ""),
                "given_name": self.normalizer.normalize_name(provider.first_name),
                "middle_name": self.normalizer.normalize_name(
                    provider.middle_name or ""
                ),
                "family_name": self.normalizer.normalize_name(provider.last_name),
                "street_address": self.normalizer.normalize_address(
                    provider.address_line_1
                ),
                "city": self.normalizer.normalize_address(provider.city),
                "state": self.normalizer.normalize_state(provider.state),
                "postal_code": self.normalizer.normalize_postal_code(
                    provider.postal_code
                ),
                "phone": self.normalizer.normalize_phone(provider.phone or ""),
                "specialty": provider.specialty or "",
                "address_status": provider.address_status,
                "phone_status": provider.phone_status,
            }
            data.append(normalized_row)

        return pd.DataFrame(data)

    def train_model(self) -> None:
        """Train the Splink deduplication model.

        Raises:
            TrainingError: If model training fails
        """
        if self._prepared_df is None:
            raise DeduplicationError("No prepared data. Call prepare_data() first.")

        logger.info("Initializing Splink model")

        # Create Splink settings
        settings = self._create_splink_settings()

        # Initialize linker
        try:
            self._linker = Linker(
                self._prepared_df,
                settings,
                db_api=DuckDBAPI(),
            )
        except Exception as e:
            raise TrainingError(f"Failed to initialize Splink: {e}")

        # Estimate u probabilities
        logger.info("Estimating u probabilities")
        try:
            self._linker.training.estimate_u_using_random_sampling(
                max_pairs=self.config.max_pairs_for_training
            )
        except Exception as e:
            logger.warning("Failed to estimate u probabilities", error=str(e))

        # Train with blocking rules
        logger.info("Training model with blocking rules")
        training_rules = self._get_training_rules()

        for rule in training_rules:
            try:
                logger.debug("Training with rule", rule=rule)
                self._linker.training.estimate_parameters_using_expectation_maximisation(
                    rule
                )
            except Exception as e:
                logger.warning(
                    "Failed to train with blocking rule",
                    rule=rule,
                    error=str(e),
                )

        logger.info("Model training complete")

    def deduplicate(
        self, threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform deduplication and return results.

        Args:
            threshold: Match probability threshold (uses config default if None)

        Returns:
            Tuple of (deduplicated DataFrame, statistics dictionary)

        Raises:
            DeduplicationError: If model hasn't been trained
        """
        if self._linker is None:
            raise DeduplicationError("Model not trained. Call train_model() first.")

        threshold = threshold or self.config.match_threshold
        logger.info("Starting deduplication", threshold=threshold)

        # Generate predictions
        logger.info("Generating pairwise predictions")
        predictions = self._linker.inference.predict()

        # Cluster at threshold
        logger.info("Clustering predictions")
        clusters = self._linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions, threshold
        )

        # Convert to DataFrame
        results_df = clusters.as_pandas_dataframe()

        # Calculate statistics
        stats = self._calculate_statistics(results_df)
        logger.info(
            "Deduplication complete",
            unique_clusters=stats["unique_clusters"],
            duplicates_found=stats["duplicates_found"],
        )

        return results_df, stats

    def _create_splink_settings(self) -> Dict:
        """Create Splink settings from configuration.

        Returns:
            Splink settings dictionary
        """
        # Build comparisons
        comparisons = []
        for comp_config in self.config.comparisons:
            comparison = self._create_comparison(comp_config)
            if comparison:
                comparisons.append(comparison)

        # Build blocking rules
        blocking_rules = [rule.rule for rule in self.config.blocking_rules]

        return {
            "link_type": self.config.link_type,
            "blocking_rules_to_generate_predictions": blocking_rules,
            "comparisons": comparisons,
            "max_iterations": self.config.max_iterations,
            "em_convergence": self.config.em_convergence,
        }

    def _create_comparison(self, config: ComparisonConfig) -> Any:
        """Create Splink comparison from configuration.

        Args:
            config: Comparison configuration

        Returns:
            Splink comparison object or None if not applicable
        """
        comparison_map = {
            "exact": cl.ExactMatch,
            "name": cl.NameComparison,
            "levenshtein": cl.LevenshteinAtThresholds,
            "jaro_winkler": cl.JaroWinklerAtThresholds,
            "postcode": cl.PostcodeComparison,
        }

        comparison_class = comparison_map.get(config.comparison_type.lower())
        if not comparison_class:
            logger.warning(
                "Unknown comparison type",
                type=config.comparison_type,
                column=config.column_name,
            )
            return None

        # Check if column exists
        if config.column_name not in self._prepared_df.columns:
            logger.warning(
                "Column not found in data",
                column=config.column_name,
            )
            return None

        # Create comparison
        if config.comparison_type.lower() == "levenshtein" and config.thresholds:
            comparison = comparison_class(config.column_name, config.thresholds)
        else:
            comparison = comparison_class(config.column_name)

        # Configure term frequency adjustments
        if hasattr(comparison, "configure"):
            comparison = comparison.configure(
                term_frequency_adjustments=config.term_frequency_adjustments
            )

        return comparison

    def _get_training_rules(self) -> List[str]:
        """Get blocking rules for training.

        Returns:
            List of blocking rules for EM training
        """
        # Use a subset of blocking rules for training
        training_rules = [
            "l.npi = r.npi",
            "l.family_name = r.family_name AND l.given_name = r.given_name",
            "l.phone = r.phone AND length(l.phone) >= 10",
        ]

        # Filter rules based on available columns
        available_rules = []
        for rule in training_rules:
            # Simple check - in production, use proper SQL parsing
            if all(
                col in self._prepared_df.columns
                for col in ["npi", "family_name", "given_name", "phone"]
            ):
                available_rules.append(rule)

        return available_rules

    def _validate_data(self, df: pd.DataFrame) -> List[str]:
        """Validate input data.

        Args:
            df: Input DataFrame

        Returns:
            List of validation error messages
        """
        errors = []

        # Check required columns
        required_columns = {"npi", "firstname", "lastname", "city", "state"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for empty DataFrame
        if df.empty:
            errors.append("Input data is empty")

        # Check NPI format if column exists
        if "npi" in df.columns:
            invalid_npis = df[~df["npi"].astype(str).str.match(r"^\d{10}$")]
            if len(invalid_npis) > 0:
                errors.append(f"Found {len(invalid_npis)} invalid NPI values")

        return errors

    def _calculate_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate deduplication statistics.

        Args:
            results_df: Deduplication results DataFrame

        Returns:
            Dictionary of statistics
        """
        total_records = len(results_df)
        unique_clusters = results_df["cluster_id"].nunique()
        duplicates_found = total_records - unique_clusters

        # Cluster size distribution
        cluster_sizes = results_df.groupby("cluster_id").size()

        return {
            "total_records": total_records,
            "unique_clusters": unique_clusters,
            "duplicates_found": duplicates_found,
            "duplication_rate": (
                duplicates_found / total_records if total_records > 0 else 0
            ),
            "largest_cluster_size": cluster_sizes.max(),
            "average_cluster_size": cluster_sizes.mean(),
            "clusters_by_size": cluster_sizes.value_counts().to_dict(),
        }

    def run_deduplication(
        self, input_path: Union[str, Path]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete deduplication pipeline.

        This is a convenience method that runs load_data, prepare_data,
        train_model, and deduplicate in sequence.

        Args:
            input_path: Path to input data file

        Returns:
            Tuple of (results_df, statistics)
        """
        logger.info(
            "Starting complete deduplication pipeline", input_path=str(input_path)
        )

        # Run the complete pipeline
        self.load_data(input_path)
        self.prepare_data()
        self.train_model()
        results_df, statistics = self.deduplicate()

        logger.info(
            "Deduplication pipeline completed",
            duplicates_found=statistics["duplicates_found"],
        )

        return results_df, statistics

    def save_results(
        self,
        results_df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "csv",
    ) -> None:
        """Save deduplication results to file.

        Args:
            results_df: Results DataFrame to save
            output_path: Path to save the file
            format: Output format ('csv', 'excel', 'json', 'parquet')
        """
        from provider_dedupe.services.output_generator import OutputGenerator

        output_gen = OutputGenerator()
        output_gen.save(results_df, Path(output_path))

        logger.info(
            "Results saved",
            output_path=str(output_path),
            format=format,
            record_count=len(results_df),
        )
