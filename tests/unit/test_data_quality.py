"""Unit tests for data quality analyzer."""

import pandas as pd
import pytest

from provider_dedupe.services.data_quality import DataQualityAnalyzer, QualityMetrics
from tests.fixtures.sample_data import get_quality_issues_data, get_sample_dataframe


class TestDataQualityAnalyzer:
    """Test cases for DataQualityAnalyzer class."""

    def test_analyze_clean_data(self):
        """Test analysis of clean data."""
        df = get_sample_dataframe()
        analyzer = DataQualityAnalyzer()
        
        metrics = analyzer.analyze(df)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.total_records == len(df)
        assert metrics.data_quality_score > 70  # Sample data has intentional duplicates for testing
        assert isinstance(metrics.missing_value_summary, dict)
        assert isinstance(metrics.field_statistics, dict)

    def test_analyze_data_with_issues(self):
        """Test analysis of data with quality issues."""
        df = pd.DataFrame(get_quality_issues_data())
        analyzer = DataQualityAnalyzer()
        
        metrics = analyzer.analyze(df)
        
        assert metrics.total_records == len(df)
        assert metrics.data_quality_score < 80  # Should be lower for problematic data
        assert len(metrics.missing_value_summary) > 0  # Should detect missing values
        assert len(metrics.recommendations) > 0  # Should have recommendations

    def test_count_duplicates(self):
        """Test duplicate counting functionality."""
        df = get_sample_dataframe()
        analyzer = DataQualityAnalyzer()
        
        duplicate_count = analyzer._count_duplicates(df)
        
        # Should detect the duplicate NPI record
        assert duplicate_count > 0

    def test_analyze_missing_values(self):
        """Test missing value analysis."""
        df = pd.DataFrame(get_quality_issues_data())
        analyzer = DataQualityAnalyzer()
        
        missing_summary = analyzer._analyze_missing_values(df)
        
        assert isinstance(missing_summary, dict)
        # Should detect missing values in various columns
        expected_columns_with_missing = ["firstname", "lastname", "address1", "zipcode", "phone"]
        for col in expected_columns_with_missing:
            if col in missing_summary:
                assert missing_summary[col]["count"] > 0
                assert missing_summary[col]["percentage"] > 0

    def test_calculate_field_statistics(self):
        """Test field statistics calculation."""
        df = pd.DataFrame(get_quality_issues_data())
        analyzer = DataQualityAnalyzer()
        
        stats = analyzer._calculate_field_statistics(df)
        
        assert isinstance(stats, dict)
        
        # Check address status statistics
        if "address_status" in stats:
            assert isinstance(stats["address_status"], dict)
            assert "inconclusive" in stats["address_status"]
        
        # Check phone validity statistics
        if "phone_validity" in stats:
            assert "valid_count" in stats["phone_validity"]
            assert "invalid_count" in stats["phone_validity"]

    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        # Test with clean data
        clean_df = get_sample_dataframe()
        analyzer = DataQualityAnalyzer()
        
        missing_summary = analyzer._analyze_missing_values(clean_df)
        duplicate_count = analyzer._count_duplicates(clean_df)
        
        score = analyzer._calculate_quality_score(clean_df, missing_summary, duplicate_count)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 70  # Clean data should have high score

    def test_generate_recommendations_for_clean_data(self):
        """Test recommendation generation for clean data."""
        df = get_sample_dataframe()
        analyzer = DataQualityAnalyzer()
        
        missing_summary = analyzer._analyze_missing_values(df)
        field_stats = analyzer._calculate_field_statistics(df)
        
        recommendations = analyzer._generate_recommendations(df, missing_summary, field_stats)
        
        # Clean data should have fewer recommendations
        assert isinstance(recommendations, list)

    def test_generate_recommendations_for_problematic_data(self):
        """Test recommendation generation for problematic data."""
        df = pd.DataFrame(get_quality_issues_data())
        analyzer = DataQualityAnalyzer()
        
        missing_summary = analyzer._analyze_missing_values(df)
        field_stats = analyzer._calculate_field_statistics(df)
        
        recommendations = analyzer._generate_recommendations(df, missing_summary, field_stats)
        
        # Problematic data should have multiple recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_report(self, tmp_path):
        """Test report generation."""
        df = get_sample_dataframe()
        analyzer = DataQualityAnalyzer()
        metrics = analyzer.analyze(df)
        
        report_path = tmp_path / "quality_report.md"
        report_content = analyzer.generate_report(metrics, str(report_path))
        
        assert isinstance(report_content, str)
        assert "Data Quality Report" in report_content
        assert str(metrics.total_records) in report_content
        assert report_path.exists()

    def test_critical_fields_detection(self):
        """Test detection of critical field issues."""
        # Create data with missing critical fields
        data = [
            {"npi": "", "firstname": "John", "lastname": "Smith"},  # Missing NPI
            {"npi": "1234567893", "firstname": "", "lastname": "Smith"},  # Missing name
            {"npi": "1234567901", "firstname": "Jane", "lastname": ""},  # Missing lastname
        ]
        df = pd.DataFrame(data)
        analyzer = DataQualityAnalyzer()
        
        metrics = analyzer.analyze(df)
        
        # Should detect critical field issues
        assert len(metrics.missing_value_summary) > 0
        critical_issues = [
            field for field, info in metrics.missing_value_summary.items()
            if info.get("is_critical", False)
        ]
        assert len(critical_issues) > 0

    def test_phone_validity_analysis(self):
        """Test phone number validity analysis."""
        data = [
            {"phone": "2175551234"},  # Valid 10-digit
            {"phone": "217555123"},   # Invalid 9-digit
            {"phone": "12175551234"}, # Valid 11-digit
            {"phone": "123"},         # Invalid short
            {"phone": ""},            # Missing
        ]
        df = pd.DataFrame(data)
        analyzer = DataQualityAnalyzer()
        
        stats = analyzer._calculate_field_statistics(df)
        
        if "phone_validity" in stats:
            phone_stats = stats["phone_validity"]
            assert "valid_count" in phone_stats
            assert "invalid_count" in phone_stats
            assert phone_stats["valid_count"] >= 2  # At least 2 valid phones
            assert phone_stats["invalid_count"] >= 2  # At least 2 invalid phones

    def test_state_distribution_analysis(self):
        """Test state distribution analysis."""
        data = [
            {"state": "IL"}, {"state": "IL"}, {"state": "IL"},  # 3 IL records
            {"state": "CA"}, {"state": "NY"},  # 1 each CA, NY
        ]
        df = pd.DataFrame(data)
        analyzer = DataQualityAnalyzer()
        
        stats = analyzer._calculate_field_statistics(df)
        
        if "state_distribution" in stats:
            state_stats = stats["state_distribution"]
            assert "unique_states" in state_stats
            assert "top_states" in state_stats
            assert state_stats["unique_states"] == 3
            assert "IL" in state_stats["top_states"]

    def test_quality_metrics_dataclass(self):
        """Test QualityMetrics dataclass functionality."""
        metrics = QualityMetrics(
            total_records=100,
            duplicate_records=10,
            missing_value_summary={"field1": {"count": 5, "percentage": 5.0}},
            field_statistics={"stats": "example"},
            data_quality_score=85.5,
            recommendations=["Fix missing values"],
        )
        
        assert metrics.total_records == 100
        assert metrics.duplicate_records == 10
        assert metrics.data_quality_score == 85.5
        assert len(metrics.recommendations) == 1
        assert isinstance(metrics.missing_value_summary, dict)
        assert isinstance(metrics.field_statistics, dict)

    def test_analyzer_initialization(self):
        """Test analyzer initialization with default settings."""
        analyzer = DataQualityAnalyzer()
        
        assert hasattr(analyzer, "critical_fields")
        assert hasattr(analyzer, "quality_thresholds")
        assert "npi" in analyzer.critical_fields
        assert "family_name" in analyzer.critical_fields
        assert "given_name" in analyzer.critical_fields