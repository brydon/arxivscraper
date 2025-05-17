# Installation and Usage Guide

## Package Installation

### Option 1: Install from Source

1. Clone or download the package source code
2. Navigate to the package directory
3. Run the installation command:

```bash
pip install -e .
```

This installs the package in development mode, allowing you to make changes to the code.

### Option 2: Create and Install a Wheel

1. Build the package:

```bash
pip install build
python -m build
```

2. Install the wheel:

```bash
pip install dist/arxivscraper-0.1.0-py3-none-any.whl
```

## Directory Structure

After installation, your package should have the following structure:

```
arxivscraper/
├── arxivscraper/
│   ├── __init__.py
│   ├── scraper.py
│   ├── analyzer.py
│   └── cli.py
├── tests/
│   ├── __init__.py
│   └── test_arxivscraper.py
├── pyproject.toml
├── setup.py
├── README.md
└── INSTALL.md  # this file
```

## Usage Examples

### Command Line

After installation, you can use the tool from the command line:

```bash
# Display help
arxivscraper --help

# Scrape papers
arxivscraper scrape --categories math cs --max-papers 100 --start-date 2024-01-01

# Analyze dataset
arxivscraper analyze arxiv_dataset/arxiv_dataset.json
```

### Python API

You can also use the package in your Python code:

```python
from arxivscraper import ArxivScraper, DatasetAnalyzer

# Scrape papers
scraper = ArxivScraper(
    output_dir="arxiv_dataset",
    start_date="2024-01-01"
)
scraper.scrape_categories(["math", "cs"])

# Analyze dataset
analyzer = DatasetAnalyzer("arxiv_dataset/arxiv_dataset.json")
analyzer.analyze()
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest
```

### Adding New Features

1. Modify the code in the `arxivscraper` directory
2. Add tests in the `tests` directory
3. Run the tests to ensure everything works
4. Update the documentation as needed

## Troubleshooting

### Common Issues

- **Rate limiting**: If you're getting HTTP 429 errors, reduce the frequency of requests or increase the delay between requests.
- **Memory issues**: If processing large datasets, consider breaking up the scraping into smaller batches.
- **LaTeX extraction failures**: Some papers may have complex LaTeX that's difficult to extract. The scraper will log these issues and continue with other papers.

### Getting Help

If you encounter any issues, please:

1. Check the logs in `arxivscraper.log`
2. Refer to the documentation in the README file
3. Open an issue on the GitHub repository

## License

This package is released under the MIT license.