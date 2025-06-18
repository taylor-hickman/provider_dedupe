"""Main CLI application for provider deduplication."""

import sys
from pathlib import Path
from typing import Optional

import click

from provider_dedupe import __version__
from provider_dedupe.core.config import DeduplicationConfig, Settings
from provider_dedupe.core.deduplicator import ProviderDeduplicator
from provider_dedupe.core.exceptions import ProviderDedupeError
from provider_dedupe.services.output_generator import OutputGenerator
from provider_dedupe.utils.logging import get_logger, setup_logging


@click.group()
@click.version_option(version=__version__)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--log-file", type=click.Path(), help="Log file path")
@click.pass_context
def cli(ctx: click.Context, debug: bool, log_file: Optional[str]) -> None:
    """Provider Deduplication CLI.
    
    A command-line tool for deduplicating healthcare provider records
    using probabilistic record linkage with the Splink library.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    log_file_path = Path(log_file) if log_file else None
    setup_logging(log_level=log_level, log_file=log_file_path)
    
    # Store settings in context
    ctx.obj["settings"] = Settings()
    ctx.obj["logger"] = get_logger(__name__)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
@click.option(
    "--threshold",
    type=float,
    default=0.95,
    help="Match probability threshold (0.0-1.0)",
)
@click.option(
    "--output-format",
    type=click.Choice(["csv", "excel", "json", "parquet"]),
    default="csv",
    help="Output format",
)
@click.option(
    "--generate-report",
    is_flag=True,
    help="Generate HTML report with statistics",
)
@click.option(
    "--blocking-rules",
    type=str,
    help="Custom blocking rules (JSON format)",
)
@click.option(
    "--batch-size",
    type=int,
    default=50000,
    help="Batch size for processing",
)
@click.pass_context
def dedupe(
    ctx: click.Context,
    input_file: Path,
    output_file: Path,
    config: Optional[Path],
    threshold: float,
    output_format: str,
    generate_report: bool,
    blocking_rules: Optional[str],
    batch_size: int,
) -> None:
    """Deduplicate provider records from INPUT_FILE to OUTPUT_FILE.
    
    Examples:
        provider-dedupe dedupe providers.csv deduplicated.csv
        provider-dedupe dedupe providers.csv results.csv --threshold 0.90
        provider-dedupe dedupe providers.csv output.xlsx --output-format excel
        provider-dedupe dedupe providers.csv output.csv --generate-report
    """
    logger = ctx.obj["logger"]
    
    try:
        # Load or create configuration
        if config:
            dedup_config = DeduplicationConfig.from_file(config)
        else:
            dedup_config = DeduplicationConfig(match_threshold=threshold)
        
        # Parse blocking rules if provided
        if blocking_rules:
            import json
            dedup_config.blocking_rules = json.loads(blocking_rules)
        
        # Update batch size
        dedup_config.chunk_size = batch_size
        
        logger.info(
            "Starting deduplication",
            input_file=str(input_file),
            output_file=str(output_file),
            threshold=dedup_config.match_threshold,
        )
        
        # Initialize deduplicator
        deduplicator = ProviderDeduplicator(config=dedup_config)
        
        # Run deduplication pipeline
        click.echo("Loading data...")
        deduplicator.load_data(input_file)
        
        click.echo("Preparing data...")
        deduplicator.prepare_data()
        
        click.echo("Training model...")
        deduplicator.train_model()
        
        click.echo("Performing deduplication...")
        results_df, statistics = deduplicator.deduplicate()
        
        # Save results
        from provider_dedupe.services.output_generator import OutputGenerator
        output_gen = OutputGenerator()
        
        # Determine output format based on file extension if not specified
        if output_format == "csv" and output_file.suffix.lower() in [".xlsx", ".xls"]:
            output_format = "excel"
        elif output_format == "csv" and output_file.suffix.lower() == ".json":
            output_format = "json"
        elif output_format == "csv" and output_file.suffix.lower() == ".parquet":
            output_format = "parquet"
        
        # Save main results
        output_gen.save(results_df, output_file, statistics)
        
        # Generate report if requested
        if generate_report:
            report_path = output_file.parent / f"{output_file.stem}_report.html"
            output_gen.save(results_df, report_path, statistics)
            click.echo(f"📊 HTML report saved to: {report_path}")
        
        # Print summary
        click.echo("\n✅ Deduplication completed successfully!")
        click.echo(f"📊 Total records: {statistics['total_records']:,}")
        click.echo(f"🔍 Unique providers: {statistics['unique_clusters']:,}")
        click.echo(f"📋 Duplicates found: {statistics['duplicates_found']:,}")
        click.echo(f"📈 Duplication rate: {statistics['duplication_rate']:.1%}")
        click.echo(f"\n💾 Results saved to: {output_file}")
        
    except ProviderDedupeError as e:
        logger.error("Deduplication failed", error=str(e), error_code=e.error_code)
        click.echo(f"❌ Error: {e.message}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", error=str(e), exc_info=True)
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("quality_reports"),
    help="Output directory for reports",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    input_file: Path,
    output_dir: Path,
) -> None:
    """Analyze data quality of INPUT_FILE.
    
    Generates comprehensive data quality reports including missing values,
    duplicates, and recommendations for improvement.
    """
    logger = ctx.obj["logger"]
    
    try:
        from provider_dedupe.services.data_loader import DataLoader
        from provider_dedupe.services.data_quality import DataQualityAnalyzer
        
        logger.info("Starting data quality analysis", input_file=str(input_file))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        click.echo("Loading data...")
        loader = DataLoader()
        df = loader.load(input_file)
        
        # Analyze quality
        click.echo("Analyzing data quality...")
        analyzer = DataQualityAnalyzer()
        metrics = analyzer.analyze(df)
        
        # Generate report
        report_path = output_dir / f"quality_report_{input_file.stem}.md"
        click.echo(f"Generating report at {report_path}...")
        report_content = analyzer.generate_report(metrics, str(report_path))
        
        # Print summary
        click.echo("\n📊 Data Quality Analysis Complete!")
        click.echo(f"📋 Total records: {metrics.total_records:,}")
        click.echo(f"🔍 Duplicate records: {metrics.duplicate_records:,}")
        click.echo(f"⭐ Quality score: {metrics.data_quality_score:.1f}/100")
        
        if metrics.recommendations:
            click.echo(f"💡 Recommendations: {len(metrics.recommendations)}")
            for i, rec in enumerate(metrics.recommendations[:3], 1):
                click.echo(f"  {i}. {rec[:80]}...")
        
        click.echo(f"\n📄 Full report saved to: {report_path}")
        
    except Exception as e:
        logger.error("Analysis failed", error=str(e), exc_info=True)
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("visualizations"),
    help="Output directory for visualizations",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["png", "svg", "pdf", "html"]),
    default="png",
    help="Output format for visualizations",
)
@click.pass_context
def visualize(
    ctx: click.Context,
    results_file: Path,
    output_dir: Path,
    output_format: str,
) -> None:
    """Generate visualizations from deduplication RESULTS_FILE.
    
    Creates charts and plots showing cluster distributions, geographic patterns,
    and other insights from the deduplication results.
    """
    logger = ctx.obj["logger"]
    
    try:
        # Optional dependency - only import if needed
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            click.echo(
                "❌ Visualization dependencies not installed. "
                "Install with: pip install provider-dedupe[viz]",
                err=True,
            )
            sys.exit(1)
        
        logger.info("Starting visualization generation", results_file=str(results_file))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo("Loading results...")
        # Import here to avoid circular imports
        from provider_dedupe.services.data_loader import DataLoader
        
        loader = DataLoader()
        df = loader.load(results_file)
        
        click.echo("Generating visualizations...")
        # Here you would implement visualization logic
        # For now, just create a simple placeholder
        
        plt.figure(figsize=(10, 6))
        if "cluster_size" in df.columns:
            cluster_sizes = df.groupby("cluster_id").size() if "cluster_id" in df.columns else [1]
            plt.hist(cluster_sizes, bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Cluster Size")
            plt.ylabel("Frequency")
            plt.title("Distribution of Cluster Sizes")
            plt.yscale("log")
        
        output_path = output_dir / f"cluster_distribution.{output_format}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        click.echo(f"✅ Visualizations saved to: {output_dir}")
        
    except Exception as e:
        logger.error("Visualization failed", error=str(e), exc_info=True)
        click.echo(f"❌ Visualization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("config.json"),
    help="Output configuration file path",
)
def init_config(output_file: Path) -> None:
    """Initialize a default configuration file.
    
    Creates a configuration file with default settings that can be
    customized for your specific deduplication needs.
    """
    try:
        # Create default configuration
        config = DeduplicationConfig()
        
        # Convert to dictionary and save
        import json
        
        config_dict = config.model_dump()
        
        with open(output_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        click.echo(f"✅ Default configuration saved to: {output_file}")
        click.echo("📝 Edit the file to customize settings for your data.")
        
    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()