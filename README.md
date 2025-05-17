# arXiv Paper Scraper

This project provides tools to scrape papers from arXiv in the math, physics, and computer science categories, create a structured dataset, and analyze the results.

## Overview

The main script (`arxiv_scraper.py`) scrapes papers from arXiv published from January 2024 to the present day. It:

1. Downloads papers from the math, physics, and CS categories
2. Extracts LaTeX source code from the downloaded archives
3. Creates a dataset in the following format:
   ```json
   {
     "authors": ["Author 1", "Author 2", ...],
     "title": "Paper Title",
     "abstract": "Paper Abstract",
     "latex": "Full LaTeX Source Code",
     "category": "Determined LaTeX Category",
     "arxiv_id": "1234.56789",
     "published": "2024-01-15",
     "primary_category": "cs.AI",
     "pdf_url": "https://arxiv.org/pdf/1234.56789.pdf"
   }
   ```
4. Periodically saves the dataset to disk to prevent data loss

Additionally, the `dataset_analyzer.py` script provides visualization and analysis of the collected dataset.

## Requirements

Install the required packages:

```bash
pip install arxiv pandas tqdm matplotlib seaborn wordcloud
```

## Usage

### Running the Scraper

```bash
python arxiv_scraper.py
```

This will:
- Create directories for dataset storage and temporary downloads
- Scrape papers from math, physics, and CS categories
- Process and save the dataset to `arxiv_dataset/arxiv_dataset.json`
- Generate statistics about the dataset

### Analyzing the Dataset

After collecting data, analyze and visualize it with:

```bash
python dataset_analyzer.py arxiv_dataset/arxiv_dataset.json
```

This will generate visualizations including:
- Distribution of papers by category
- Publication trends over time
- Author statistics
- Word clouds of abstracts
- Analysis of LaTeX content

Output visualizations and reports will be saved in the `arxiv_dataset/analysis` directory.

## Customization

You can modify the following parameters in `arxiv_scraper.py`:

- `output_dir`: Directory to save the dataset
- `temp_dir`: Directory for temporary downloads
- `max_papers_per_category`: Maximum number of papers to scrape per category
- `save_interval`: How often to save the dataset (number of papers)
- `start_date`: The start date for scraping papers (format: YYYY-MM-DD)

## How It Works

1. **Querying arXiv**: Uses the arxiv.py library to query the arXiv API for papers in specified categories.
2. **Downloading Papers**: Downloads the paper source files (.tar.gz) from arXiv.
3. **Extracting LaTeX**: Extracts and processes the LaTeX source from the downloaded archives.
4. **Categorization**: Analyzes the LaTeX content to determine its category (theorem, algorithm, experiment, etc.).
5. **Dataset Creation**: Builds a structured dataset with all the extracted information.
6. **Periodic Saving**: Saves the dataset at regular intervals to prevent data loss.

## Notes

- The script is designed to be respectful of arXiv's servers by including delays between requests and using a conservative page size.
- The LaTeX categorization is based on heuristics and may not be 100% accurate.
- Processing large numbers of papers may take significant time and disk space.
- The script handles errors gracefully and continues processing even if individual papers fail.

## Error Handling

The script includes comprehensive error handling:

- Network errors: Retries on connection issues
- Parsing errors: Skips problematic papers
- File system errors: Creates necessary directories and handles permissions
- Unexpected failures: Logs errors and saves progress

All errors are logged to `arxiv_scraper.log` for debugging.

## Advanced Usage

### Resuming a Previous Run

If the script is stopped, it will automatically resume from where it left off by loading the existing dataset and skipping already processed papers.

### Focusing on Specific Categories

To scrape papers from specific arXiv categories, modify the `categories` list in the `main()` function:

```python
# Original
categories = ['math', 'physics', 'cs']

# Modified for specific subcategories
categories = ['cs.AI', 'cs.ML', 'math.CO']
```

### Custom Date Ranges

Adjust the `start_date` parameter to change the time range:

```python
scraper = ArxivScraper(
    # ...
    start_date="2023-01-01"  # Changed from 2024-01-01
)
```

## License

This project is provided as open-source software. Feel free to use, modify, and distribute it according to your needs.