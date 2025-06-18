# Provider Dedupe

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](https://mypy.readthedocs.io/)

A command-line tool for deduplicating healthcare provider data using probabilistic record linkage with the Splink library.

## 📦 Installation

### From PyPI (when published)
```bash
pip install provider-dedupe
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/taylor-hickman/provider_dedupe.git
cd provider_dedupe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,viz,excel,parquet]"

# Install pre-commit hooks (optional)
pre-commit install
```

## 🏃‍♂️ Quick Start

### Basic Usage Examples

```bash
# Example 1: Simple deduplication with default settings
provider-dedupe dedupe providers.csv deduplicated_providers.csv

# Example 2: Deduplication with custom threshold and HTML report
provider-dedupe dedupe providers.csv results.csv --threshold 0.95 --generate-report

# Example 3: Using a configuration file for advanced settings
provider-dedupe dedupe providers.csv output.xlsx --config config.json --generate-report

# Example 4: Analyze data quality before deduplication
provider-dedupe analyze providers.csv --output-dir quality_reports/

# Example 5: Generate visualizations from results
provider-dedupe visualize deduplicated_providers.csv --output-dir visualizations/
```

### Sample Input Data Format
Your CSV file should look like this:
```csv
npi,firstname,lastname,address1,city,state,zipcode
1234567890,JOHN,SMITH,123 MAIN ST,NEW YORK,NY,10001
1234567890,JOHN,SMITH,123 MAIN STREET,NEW YORK,NY,10001
9876543210,JANE,DOE,456 ELM AVE,BOSTON,MA,02101
```

### Command Line Options
```bash
provider-dedupe dedupe --help

Options:
  --threshold FLOAT         Match threshold (0.0-1.0) [default: 0.95]
  --config PATH            Path to configuration file
  --output-format TEXT     Output format: csv, excel, json, parquet [default: csv]
  --generate-report        Generate HTML report with statistics
  --blocking-rules TEXT    Custom blocking rules (JSON format)
  --batch-size INTEGER     Batch size for processing [default: 50000]
  --help                  Show this message and exit
```

### Python API Example
```python
from provider_dedupe import ProviderDeduplicator
from provider_dedupe.core.config import DeduplicationConfig

# Example 1: Basic usage
deduplicator = ProviderDeduplicator()
results_df, stats = deduplicator.run_deduplication("providers.csv")
print(f"Found {stats['duplicates_found']} duplicate records")
print(f"Merged into {stats['unique_providers']} unique providers")

# Example 2: Custom configuration
config = DeduplicationConfig(
    match_threshold=0.98,
    blocking_rules=[
        {"rule": "l.npi = r.npi", "description": "Exact NPI match"},
        {"rule": "l.zipcode = r.zipcode AND l.lastname = r.lastname", 
         "description": "Same ZIP and last name"}
    ]
)
deduplicator = ProviderDeduplicator(config=config)
results_df, stats = deduplicator.run_deduplication("providers.csv")

# Example 3: Save results in multiple formats
deduplicator.save_results(results_df, "output.csv", format="csv")
deduplicator.save_results(results_df, "output.xlsx", format="excel")
deduplicator.save_results(results_df, "output.json", format="json")
```

## 📊 Input Data Format

The system expects CSV files with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `npi` | National Provider Identifier | ✅ |
| `firstname` | Provider first name | ✅ |
| `lastname` | Provider last name | ✅ |
| `address1` | Street address | ✅ |
| `city` | City name | ✅ |
| `state` | State code (2 letters) | ✅ |
| `zipcode` | ZIP/postal code | ✅ |
| `gnpi` | Group NPI | ❌ |
| `group_name` | Organization name | ❌ |
| `primary_spec_desc` | Specialty | ❌ |
| `phone` | Phone number | ❌ |
| `address_status` | Address quality | ❌ |
| `phone_status` | Phone quality | ❌ |

## ⚙️ Configuration

### Configuration File Structure
```json
{
  "match_threshold": 0.95,
  "max_iterations": 20,
  "em_convergence": 0.001,
  "blocking_rules": [
    {
      "rule": "l.npi = r.npi",
      "description": "Exact NPI match"
    }
  ],
  "comparisons": [
    {
      "column_name": "npi",
      "comparison_type": "exact",
      "term_frequency_adjustments": false
    }
  ]
}
```

### Environment Variables
```bash
# Set via .env file or environment
PROVIDER_DEDUPE_LOG_LEVEL=INFO
PROVIDER_DEDUPE_DATA_DIR=/path/to/data
PROVIDER_DEDUPE_OUTPUT_DIR=/path/to/output
PROVIDER_DEDUPE_MAX_WORKERS=4
```

## 🏗️ Architecture

```
src/provider_dedupe/
├── core/                   # Core business logic
│   ├── config.py          # Configuration management
│   ├── deduplicator.py    # Main deduplication engine
│   └── exceptions.py      # Custom exceptions
├── models/                 # Data models
│   └── provider.py        # Provider and record models
├── services/               # Service layer
│   ├── data_loader.py     # Multi-format data loading
│   ├── data_quality.py    # Quality analysis
│   └── output_generator.py # Result output
├── utils/                  # Utilities
│   ├── logging.py         # Structured logging
│   └── normalization.py   # Text normalization
└── cli/                    # Command-line interface
    └── main.py            # CLI commands
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=provider_dedupe --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v

# Run performance tests
pytest tests/performance/ -m performance
```

## 📈 Performance

### Optimization Tips
- Use appropriate blocking rules for your data
- Adjust `max_pairs_for_training` based on available memory
- Enable parallel processing for large datasets
- Consider data preprocessing to improve quality

## 🔧 Development

### Code Quality Tools
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run all quality checks
pre-commit run --all-files
```

### Project Structure
- **src/**: Source code using src layout
- **tests/**: Comprehensive test suite
- **scripts/**: Utility scripts
- **.github/**: CI/CD workflows

## 📚 API Reference

### Core Classes

#### `ProviderDeduplicator`
Main deduplication engine.

```python
class ProviderDeduplicator:
    def __init__(
        self,
        config: Optional[DeduplicationConfig] = None,
        data_loader: Optional[DataLoader] = None,
        quality_analyzer: Optional[DataQualityAnalyzer] = None,
    ) -> None: ...
    
    def load_data(self, input_path: Union[str, Path]) -> pd.DataFrame: ...
    def prepare_data(self) -> pd.DataFrame: ...
    def train_model(self) -> None: ...
    def deduplicate(self, threshold: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]: ...
```

#### `Provider`
Data model for provider information.

```python
class Provider(BaseModel):
    npi: str
    first_name: str
    last_name: str
    address_line_1: str
    city: str
    state: str
    postal_code: str
    # ... additional fields
```

### CLI Commands

#### `dedupe`
Main deduplication command.
```bash
provider-dedupe dedupe INPUT_FILE OUTPUT_FILE [OPTIONS]
```

#### `analyze`
Data quality analysis.
```bash
provider-dedupe analyze INPUT_FILE [OPTIONS]
```

#### `visualize`
Generate visualizations.
```bash
provider-dedupe visualize RESULTS_FILE [OPTIONS]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Run tests and quality checks
6. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all public APIs
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://github.com/taylor-hickman/provider_dedupe)
- **Issues**: [GitHub Issues](https://github.com/taylor-hickman/provider_dedupe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/taylor-hickman/provider_dedupe/discussions)

## 🙏 Acknowledgments

- Built with [Splink](https://github.com/moj-analytical-services/splink) by the UK Ministry of Justice
- Thank you to all contributors and users

## 📊 Metrics

- [![Coverage](https://codecov.io/gh/taylor-hickman/provider_dedupe/branch/main/graph/badge.svg)](https://codecov.io/gh/taylor-hickman/provider_dedupe)
- [![Tests](https://github.com/taylor-hickman/provider_dedupe/workflows/Tests/badge.svg)](https://github.com/taylor-hickman/provider_dedupe/actions)

---

**Built with ❤️ for the open source healthcare data community**