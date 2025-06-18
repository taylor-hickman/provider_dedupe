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
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory (default: auto-generated timestamped directory)",
)
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
    "--max-pairs",
    type=int,
    help="Maximum pairs for training",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Quick mode: ultra-strict deduplication for true duplicates only",
)
@click.pass_context
def dedupe(
    ctx: click.Context,
    input_file: Path,
    output_dir: Optional[Path],
    config: Optional[Path],
    threshold: float,
    max_pairs: Optional[int],
    quick: bool,
) -> None:
    """Deduplicate provider records from INPUT_FILE with organized output.
    
    Creates a complete output directory with:
    - Deduplicated CSV results  
    - Interactive HTML report
    - Visualizations and charts
    - Configuration used
    - Summary README
    
    Examples:
        provider-dedupe dedupe providers.csv
        provider-dedupe dedupe data.csv --threshold 0.90
        provider-dedupe dedupe providers.csv --quick  # Ultra-strict mode
        provider-dedupe dedupe data.csv --output-dir my_results/
    """
    logger = ctx.obj["logger"]
    
    try:
        # Import OutputManager here to avoid circular imports
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from output_manager import OutputManager
        
        # Initialize output manager
        output_manager = OutputManager(input_file, output_dir)
        
        # Load or create configuration
        if config:
            dedup_config = DeduplicationConfig.from_file(config)
            config_data = dedup_config.model_dump()
        elif quick:
            # Use ultra-strict configuration for quick mode
            config_data = {
                "link_type": "dedupe_only",
                "blocking_rules": [
                    {
                        "rule": "l.npi = r.npi AND l.family_name = r.family_name AND l.given_name = r.given_name AND l.postal_code = r.postal_code",
                        "description": "Block only on exact NPI, name, and postal code match"
                    }
                ],
                "comparisons": [
                    {"column_name": "npi", "comparison_type": "exact", "term_frequency_adjustments": False},
                    {"column_name": "given_name", "comparison_type": "exact", "term_frequency_adjustments": False},
                    {"column_name": "family_name", "comparison_type": "exact", "term_frequency_adjustments": False},
                    {"column_name": "street_address", "comparison_type": "jaro_winkler", "term_frequency_adjustments": False, "thresholds": [0.90, 0.98]},
                    {"column_name": "postal_code", "comparison_type": "exact", "term_frequency_adjustments": False},
                    {"column_name": "phone", "comparison_type": "exact", "term_frequency_adjustments": False}
                ],
                "max_iterations": 10,
                "em_convergence": 0.001,
                "match_threshold": 0.98,
                "min_cluster_size": 2,
                "max_pairs_for_training": 50000,
                "chunk_size": 5000,
                "use_parallel": True
            }
            dedup_config = DeduplicationConfig(**config_data)
            click.echo("🚀 Using quick mode (ultra-strict): only true duplicates will be grouped")
        else:
            dedup_config = DeduplicationConfig()
            config_data = dedup_config.model_dump()
        
        # Override threshold if provided
        if threshold != 0.95:  # Not default
            dedup_config.match_threshold = threshold
            config_data["match_threshold"] = threshold
            
        # Override max pairs if provided
        if max_pairs:
            dedup_config.max_pairs_for_training = max_pairs
            config_data["max_pairs_for_training"] = max_pairs
        
        logger.info(
            "Starting deduplication",
            input_file=str(input_file),
            output_dir=str(output_manager.output_dir),
            threshold=dedup_config.match_threshold,
            quick_mode=quick,
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
        
        # Print summary
        click.echo("\n✅ Deduplication completed successfully!")
        click.echo(f"📊 Total records: {statistics['total_records']:,}")
        click.echo(f"🔍 Unique providers: {statistics['unique_clusters']:,}")
        click.echo(f"📋 Duplicates found: {statistics['duplicates_found']:,}")
        click.echo(f"📈 Duplication rate: {statistics['duplication_rate']:.1%}")
        
        # Create organized output
        final_output_dir = output_manager.finalize(results_df, config_data, statistics)
        
        click.echo(f"\n🎉 Complete results package created: {final_output_dir}")
        
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