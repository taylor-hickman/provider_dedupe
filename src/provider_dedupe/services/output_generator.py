"""Output generation service for deduplication results."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from provider_dedupe.core.exceptions import OutputError
from provider_dedupe.utils.logging import get_logger

logger = get_logger(__name__)


class OutputGenerator:
    """Service for generating various output formats from deduplication results.

    Supports multiple output formats including CSV, Excel, JSON, Parquet,
    and HTML reports with customizable templates.
    """

    def __init__(self) -> None:
        """Initialize the output generator."""
        self._formatters = {
            ".csv": self._save_csv,
            ".xlsx": self._save_excel,
            ".json": self._save_json,
            ".parquet": self._save_parquet,
            ".pq": self._save_parquet,
            ".html": self._save_html,
        }

    def save(
        self,
        data: pd.DataFrame,
        output_path: Union[str, Path],
        statistics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Save deduplication results to file.

        Args:
            data: Results DataFrame to save
            output_path: Path for output file
            statistics: Optional statistics to include
            **kwargs: Additional arguments for specific formats

        Raises:
            OutputError: If save operation fails
        """
        output_path = Path(output_path)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate format
        if output_path.suffix.lower() not in self._formatters:
            raise OutputError(
                f"Unsupported output format: {output_path.suffix}. "
                f"Supported formats: {', '.join(self._formatters.keys())}"
            )

        # Save file
        formatter = self._formatters[output_path.suffix.lower()]
        try:
            logger.info(
                "Saving output", path=str(output_path), format=output_path.suffix
            )
            formatter(data, output_path, statistics, **kwargs)
            logger.info("Output saved successfully", path=str(output_path))
        except Exception as e:
            raise OutputError(f"Failed to save output to {output_path}: {e}")

    def _save_csv(
        self,
        data: pd.DataFrame,
        output_path: Path,
        statistics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Save as CSV file.

        Args:
            data: DataFrame to save
            output_path: Output file path
            statistics: Not used for CSV format
            **kwargs: Additional pandas to_csv arguments
        """
        params = {"index": False}
        params.update(kwargs)
        data.to_csv(output_path, **params)

    def _save_excel(
        self,
        data: pd.DataFrame,
        output_path: Path,
        statistics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Save as Excel file with optional statistics sheet.

        Args:
            data: DataFrame to save
            output_path: Output file path
            statistics: Optional statistics for separate sheet
            **kwargs: Additional pandas to_excel arguments
        """
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Write main data
            data.to_excel(writer, sheet_name="Deduplication Results", index=False)

            # Write statistics if provided
            if statistics:
                stats_df = self._statistics_to_dataframe(statistics)
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)

                # Write summary metrics
                summary_data = {
                    "Metric": [
                        "Total Records",
                        "Unique Clusters",
                        "Duplicates Found",
                        "Duplication Rate",
                        "Largest Cluster",
                    ],
                    "Value": [
                        statistics.get("total_records", 0),
                        statistics.get("unique_clusters", 0),
                        statistics.get("duplicates_found", 0),
                        f"{statistics.get('duplication_rate', 0) * 100:.2f}%",
                        statistics.get("largest_cluster_size", 0),
                    ],
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

    def _save_json(
        self,
        data: pd.DataFrame,
        output_path: Path,
        statistics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Save as JSON file.

        Args:
            data: DataFrame to save
            output_path: Output file path
            statistics: Optional statistics to include
            **kwargs: Additional pandas to_json arguments
        """
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "record_count": len(data),
            },
            "statistics": statistics or {},
            "records": data.to_dict(orient="records"),
        }

        import json

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

    def _save_parquet(
        self,
        data: pd.DataFrame,
        output_path: Path,
        statistics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Save as Parquet file.

        Args:
            data: DataFrame to save
            output_path: Output file path
            statistics: Not used for Parquet format
            **kwargs: Additional pandas to_parquet arguments
        """
        params = {"index": False, "compression": "snappy"}
        params.update(kwargs)
        data.to_parquet(output_path, **params)

    def _save_html(
        self,
        data: pd.DataFrame,
        output_path: Path,
        statistics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Save as HTML report.

        Args:
            data: DataFrame to save
            output_path: Output file path
            statistics: Optional statistics to include
            **kwargs: Additional options (template, max_rows, etc.)
        """
        template = kwargs.get("template", "default")
        max_rows = kwargs.get("max_rows", 100)

        if template == "default":
            html_content = self._generate_default_html_report(
                data, statistics, max_rows
            )
        else:
            # Support for custom templates
            html_content = self._generate_custom_html_report(
                data, statistics, template, max_rows
            )

        with open(output_path, "w") as f:
            f.write(html_content)

    def _generate_default_html_report(
        self,
        data: pd.DataFrame,
        statistics: Optional[Dict[str, Any]],
        max_rows: int,
    ) -> str:
        """Generate default HTML report.

        Args:
            data: Results DataFrame
            statistics: Optional statistics
            max_rows: Maximum rows to display

        Returns:
            HTML content as string
        """
        stats = statistics or {}

        # Get sample of large clusters
        large_clusters = data[data.get("cluster_size", 0) > 1].head(max_rows)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Provider Deduplication Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            margin: 0 0 10px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .cluster-badge {{
            display: inline-block;
            padding: 4px 8px;
            background-color: #e74c3c;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Provider Deduplication Report</h1>
        <div class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{stats.get('total_records', 0):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Unique Providers</div>
            <div class="metric-value">{stats.get('unique_clusters', 0):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Duplicates Found</div>
            <div class="metric-value">{stats.get('duplicates_found', 0):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Duplication Rate</div>
            <div class="metric-value">{stats.get('duplication_rate', 0) * 100:.1f}%</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Cluster Size Distribution</h2>
        {self._generate_cluster_distribution_table(stats)}
    </div>
    
    <div class="section">
        <h2>Sample Duplicate Clusters</h2>
        <p>Showing up to {max_rows} records from clusters with duplicates.</p>
        {self._generate_sample_clusters_table(large_clusters)}
    </div>
</body>
</html>"""

        return html

    def _generate_cluster_distribution_table(self, statistics: Dict[str, Any]) -> str:
        """Generate HTML table for cluster size distribution.

        Args:
            statistics: Statistics dictionary

        Returns:
            HTML table as string
        """
        clusters_by_size = statistics.get("clusters_by_size", {})
        if not clusters_by_size:
            return "<p>No cluster distribution data available.</p>"

        html = """<table>
            <tr>
                <th>Cluster Size</th>
                <th>Number of Clusters</th>
                <th>Total Records</th>
            </tr>"""

        for size, count in sorted(clusters_by_size.items()):
            total_records = size * count
            html += f"""
            <tr>
                <td>{size}</td>
                <td>{count:,}</td>
                <td>{total_records:,}</td>
            </tr>"""

        html += "</table>"
        return html

    def _generate_sample_clusters_table(self, data: pd.DataFrame) -> str:
        """Generate HTML table for sample clusters.

        Args:
            data: Sample cluster data

        Returns:
            HTML table as string
        """
        if data.empty:
            return "<p>No duplicate clusters found.</p>"

        # Select columns to display
        display_columns = [
            "cluster_id",
            "npi",
            "given_name",
            "family_name",
            "city",
            "state",
            "phone",
            "cluster_size",
        ]
        available_columns = [col for col in display_columns if col in data.columns]

        html = "<table><tr>"
        for col in available_columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>"
        html += "</tr>"

        for _, row in data.iterrows():
            html += "<tr>"
            for col in available_columns:
                value = row.get(col, "")
                if col == "cluster_size" and value > 1:
                    html += f'<td><span class="cluster-badge">{value}</span></td>'
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"

        html += "</table>"
        return html

    def _generate_custom_html_report(
        self,
        data: pd.DataFrame,
        statistics: Optional[Dict[str, Any]],
        template_path: str,
        max_rows: int,
    ) -> str:
        """Generate HTML report using custom template.

        Args:
            data: Results DataFrame
            statistics: Optional statistics
            template_path: Path to custom template
            max_rows: Maximum rows to display

        Returns:
            HTML content as string
        """
        # This would load and render a custom template
        # For now, fallback to default
        return self._generate_default_html_report(data, statistics, max_rows)

    def _statistics_to_dataframe(self, statistics: Dict[str, Any]) -> pd.DataFrame:
        """Convert statistics dictionary to DataFrame.

        Args:
            statistics: Statistics dictionary

        Returns:
            DataFrame representation of statistics
        """
        rows = []
        for key, value in statistics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append(
                        {
                            "Category": key,
                            "Metric": sub_key,
                            "Value": str(sub_value),
                        }
                    )
            else:
                rows.append(
                    {
                        "Category": "General",
                        "Metric": key,
                        "Value": str(value),
                    }
                )

        return pd.DataFrame(rows)
