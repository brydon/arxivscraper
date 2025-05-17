# arXiv Paper Scraper

This project provides tools to scrape papers from arXiv in the math, physics, and computer science categories, create a structured dataset, and analyze the results.

## Overview

The scraper efficiently downloads and processes papers from arXiv using parallel processing and optimized settings. It:

1. Downloads papers from specified arXiv categories using parallel processing
2. Extracts LaTeX source code from the downloaded archives using streaming extraction
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
4. Periodically saves the dataset to disk with automatic backups
5. Supports graceful shutdown with Ctrl+C to save progress
6. Provides comprehensive analysis tools for the collected dataset

## Features

- **Parallel Processing**: Process multiple papers simultaneously for faster scraping
- **Streaming Extraction**: Efficient LaTeX extraction without writing to disk
- **Smart Caching**: Cached category detection for better performance
- **Graceful Shutdown**: Safe interruption with progress saving
- **Automatic Backups**: Regular dataset backups with timestamps
- **Comprehensive Analysis**: Detailed statistics and visualizations
- **Rate Limiting**: Respects arXiv's rate limits while maximizing throughput
- **Error Handling**: Robust error recovery and logging

## Requirements

See `INSTALL.md`, but tl;dr `pip install -e .`

A `uv.lock` is also provided for `uv` users.

## Usage

### Running the Scraper

```bash
# Basic usage
arxivscraper scrape

# Advanced usage with custom settings
arxivscraper scrape \
    --output-dir "arxiv_dataset" \
    --temp-dir "/tmp/arxivdownloads" \
    --max-papers 5000 \
    --save-interval 100 \
    --start-date "2024-01-01" \
    --categories math cs physics \
    --max-workers 16
```

This will:
- Create directories for dataset storage and temporary downloads
- Scrape papers from specified categories using parallel processing
- Process and save the dataset to `arxiv_dataset/arxiv_dataset.json`
- Generate statistics about the dataset
- Create automatic backups during processing

### Analyzing the Dataset

After collecting data, analyze and visualize it with:

```bash
arxivscraper analyze arxiv_dataset/arxiv_dataset.json
```

This will generate visualizations including:
- Distribution of papers by category
- Publication trends over time
- Author statistics
- Word clouds of abstracts
- Analysis of LaTeX content

Output visualizations and reports will be saved in the `arxiv_dataset/analysis` directory.

## Configuration

You can customize the scraper's behavior using these parameters:

- `output_dir`: Directory to save the dataset
- `temp_dir`: Directory for temporary downloads
- `max_papers_per_category`: Maximum number of papers to scrape per category
- `save_interval`: How often to save the dataset (number of papers)
- `start_date`: The start date for scraping papers (format: YYYY-MM-DD)
- `max_workers`: Number of parallel workers for processing papers

## How It Works

1. **Querying arXiv**: Uses the arxiv.py library to query the arXiv API for papers in specified categories.
2. **Parallel Processing**: Processes multiple papers simultaneously using a thread pool.
3. **Streaming Extraction**: Extracts LaTeX source directly from the tar.gz files without writing to disk.
4. **Smart Categorization**: Uses cached pattern matching to determine paper categories.
5. **Progress Saving**: Regularly saves progress and creates backups.
6. **Graceful Shutdown**: Handles interruptions by saving progress and cleaning up resources.

## Error Handling

The scraper includes comprehensive error handling:

- Network errors: Retries on connection issues with exponential backoff
- Parsing errors: Skips problematic papers and continues processing
- File system errors: Creates necessary directories and handles permissions
- Unexpected failures: Logs errors and saves progress
- Interruptions: Graceful shutdown with progress saving

All errors are logged to `arxivscraper.log` for debugging.

## Advanced Usage

### Resuming a Previous Run

If the script is stopped, it will automatically resume from where it left off by:
- Loading the existing dataset
- Skipping already processed papers
- Continuing from the last saved state

### Focusing on Specific Categories

To scrape papers from specific arXiv categories:

```bash
arxivscraper scrape --categories math.AG cs.AI physics.acc-ph
```

### Customizing Parallel Processing

Adjust the number of parallel workers based on your system:

```bash
arxivscraper scrape --max-workers 32  # For high-performance systems
arxivscraper scrape --max-workers 4   # For resource-constrained systems
```

## Notes

- The scraper is designed to be respectful of arXiv's servers while maximizing throughput
- LaTeX categorization is based on heuristics and may not be 100% accurate
- Processing large numbers of papers may take significant time and disk space
- The scraper can be safely interrupted at any time with Ctrl+C
- All progress is automatically saved and can be resumed later

## License

This project is provided as open-source software. Feel free to use, modify, and distribute it according to your needs.