"""Data quality analysis service."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from provider_dedupe.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Container for data quality metrics."""

    total_records: int
    duplicate_records: int
    missing_value_summary: Dict[str, Dict[str, float]]
    field_statistics: Dict[str, Dict[str, Any]]
    data_quality_score: float
    recommendations: List[str] = field(default_factory=list)


class DataQualityAnalyzer:
    """Service for analyzing data quality and generating reports.

    Provides comprehensive data quality analysis including:
    - Missing value analysis
    - Duplicate detection
    - Field-specific statistics
    - Data quality scoring
    - Recommendations for improvement
    """

    def __init__(self) -> None:
        """Initialize the data quality analyzer."""
        self.critical_fields = {"npi", "family_name", "given_name"}
        self.quality_thresholds = {
            "missing_rate": 0.05,  # 5% missing values threshold
            "duplicate_rate": 0.10,  # 10% duplicate threshold
            "min_phone_length": 10,
            "min_postal_length": 5,
        }

    def analyze(self, df: pd.DataFrame) -> QualityMetrics:
        """Perform comprehensive data quality analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            QualityMetrics object with analysis results
        """
        logger.info("Starting data quality analysis", rows=len(df))

        # Calculate basic metrics
        total_records = len(df)
        duplicate_records = self._count_duplicates(df)
        missing_summary = self._analyze_missing_values(df)
        field_stats = self._calculate_field_statistics(df)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            df, missing_summary, duplicate_records
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            df, missing_summary, field_stats
        )

        metrics = QualityMetrics(
            total_records=total_records,
            duplicate_records=duplicate_records,
            missing_value_summary=missing_summary,
            field_statistics=field_stats,
            data_quality_score=quality_score,
            recommendations=recommendations,
        )

        logger.info(
            "Data quality analysis complete",
            quality_score=round(quality_score, 2),
            recommendations_count=len(recommendations),
        )

        return metrics

    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Count potential duplicate records.

        Args:
            df: DataFrame to analyze

        Returns:
            Number of potential duplicates
        """
        # Check for exact duplicates on key fields
        key_fields = ["npi", "given_name", "family_name"]
        available_fields = [f for f in key_fields if f in df.columns]

        if not available_fields:
            return 0

        duplicates = df[df.duplicated(subset=available_fields, keep=False)]
        return len(duplicates)

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze missing values in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing value statistics per column
        """
        missing_summary = {}

        for column in df.columns:
            missing_count = df[column].isna().sum()
            empty_string_count = (df[column] == "").sum()
            total_missing = missing_count + empty_string_count

            if total_missing > 0:
                missing_summary[column] = {
                    "count": int(total_missing),
                    "percentage": round(total_missing / len(df) * 100, 2),
                    "is_critical": column in self.critical_fields,
                }

        return missing_summary

    def _calculate_field_statistics(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for specific fields.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with field-specific statistics
        """
        stats = {}

        # Address status distribution
        if "address_status" in df.columns:
            stats["address_status"] = df["address_status"].value_counts().to_dict()

        # Phone status distribution
        if "phone_status" in df.columns:
            stats["phone_status"] = df["phone_status"].value_counts().to_dict()

        # State distribution
        if "state" in df.columns:
            state_counts = df["state"].value_counts()
            stats["state_distribution"] = {
                "unique_states": len(state_counts),
                "top_states": state_counts.head(5).to_dict(),
            }

        # Phone number validity
        if "phone" in df.columns:
            phone_lengths = df["phone"].str.len()
            stats["phone_validity"] = {
                "valid_count": int((phone_lengths >= 10).sum()),
                "invalid_count": int((phone_lengths < 10).sum()),
                "missing_count": int(df["phone"].isna().sum()),
            }

        # Group practice analysis
        if "group_npi" in df.columns:
            group_counts = df["group_npi"].value_counts()
            stats["group_practice"] = {
                "unique_groups": len(group_counts),
                "largest_group_size": (
                    int(group_counts.iloc[0]) if len(group_counts) > 0 else 0
                ),
                "solo_practitioners": int((df["group_npi"].isna()).sum()),
            }

        return stats

    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        missing_summary: Dict[str, Dict[str, float]],
        duplicate_records: int,
    ) -> float:
        """Calculate overall data quality score (0-100).

        Args:
            df: DataFrame being analyzed
            missing_summary: Missing value analysis results
            duplicate_records: Number of duplicate records

        Returns:
            Quality score between 0 and 100
        """
        scores = []

        # Score for missing values in critical fields
        critical_missing_score = 100
        for field in self.critical_fields:
            if field in missing_summary:
                missing_pct = missing_summary[field]["percentage"]
                field_score = max(
                    0, 100 - (missing_pct * 10)
                )  # 10% penalty per 1% missing
                critical_missing_score = min(critical_missing_score, field_score)
        scores.append(critical_missing_score * 0.4)  # 40% weight

        # Score for overall missing values
        total_missing_pct = (
            sum(m["percentage"] for m in missing_summary.values()) / len(df.columns)
            if len(df.columns) > 0
            else 0
        )
        overall_missing_score = max(0, 100 - (total_missing_pct * 5))
        scores.append(overall_missing_score * 0.2)  # 20% weight

        # Score for duplicates
        duplicate_rate = duplicate_records / len(df) if len(df) > 0 else 0
        duplicate_score = max(0, 100 - (duplicate_rate * 100 * 2))  # 2x penalty
        scores.append(duplicate_score * 0.2)  # 20% weight

        # Score for data validity
        validity_score = 100
        if "phone" in df.columns:
            valid_phones = (df["phone"].str.len() >= 10).sum()
            phone_validity_rate = valid_phones / len(df)
            validity_score *= phone_validity_rate
        scores.append(validity_score * 0.2)  # 20% weight

        return sum(scores)

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        missing_summary: Dict[str, Dict[str, float]],
        field_stats: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate data quality improvement recommendations.

        Args:
            df: DataFrame being analyzed
            missing_summary: Missing value analysis results
            field_stats: Field-specific statistics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check critical fields
        for field in self.critical_fields:
            if field in missing_summary:
                missing_pct = missing_summary[field]["percentage"]
                if missing_pct > self.quality_thresholds["missing_rate"] * 100:
                    recommendations.append(
                        f"Critical field '{field}' has {missing_pct}% missing values. "
                        f"Consider data enrichment or validation."
                    )

        # Check address quality
        if "address_status" in field_stats:
            inconclusive = field_stats["address_status"].get("Inconclusive", 0)
            if inconclusive > len(df) * 0.2:  # More than 20% inconclusive
                recommendations.append(
                    f"{inconclusive} records have inconclusive addresses. "
                    "Consider address validation service."
                )

        # Check phone quality
        if "phone_validity" in field_stats:
            invalid_phones = field_stats["phone_validity"]["invalid_count"]
            if invalid_phones > len(df) * 0.1:  # More than 10% invalid
                recommendations.append(
                    f"{invalid_phones} records have invalid phone numbers. "
                    "Review phone number formatting and validation."
                )

        # Check for high duplicate rate
        duplicate_rate = self._count_duplicates(df) / len(df) if len(df) > 0 else 0
        if duplicate_rate > self.quality_thresholds["duplicate_rate"]:
            recommendations.append(
                f"High duplicate rate detected ({duplicate_rate:.1%}). "
                "Deduplication is strongly recommended."
            )

        # Check for data imbalance
        if "state_distribution" in field_stats:
            top_state_count = sum(
                field_stats["state_distribution"]["top_states"].values()
            )
            if top_state_count > len(df) * 0.8:  # 80% in top 5 states
                recommendations.append(
                    "Data is highly concentrated in a few states. "
                    "Consider if this represents your target population."
                )

        return recommendations

    def generate_report(
        self, metrics: QualityMetrics, output_path: Optional[str] = None
    ) -> str:
        """Generate a formatted data quality report.

        Args:
            metrics: Quality metrics to report
            output_path: Optional path to save report

        Returns:
            Formatted report as string
        """
        report = f"""
# Data Quality Report

## Summary
- **Total Records**: {metrics.total_records:,}
- **Duplicate Records**: {metrics.duplicate_records:,} ({metrics.duplicate_records/metrics.total_records*100:.1f}%)
- **Data Quality Score**: {metrics.data_quality_score:.1f}/100

## Missing Values
"""

        for field, stats in sorted(metrics.missing_value_summary.items()):
            critical_marker = "⚠️ " if stats["is_critical"] else ""
            report += f"- {critical_marker}{field}: {stats['count']:,} ({stats['percentage']:.1f}%)\n"

        report += "\n## Field Statistics\n"
        for category, stats in metrics.field_statistics.items():
            report += f"\n### {category.replace('_', ' ').title()}\n"
            if isinstance(stats, dict):
                for key, value in stats.items():
                    report += f"- {key.replace('_', ' ').title()}: {value}\n"

        if metrics.recommendations:
            report += "\n## Recommendations\n"
            for i, rec in enumerate(metrics.recommendations, 1):
                report += f"{i}. {rec}\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info("Data quality report saved", path=output_path)

        return report
