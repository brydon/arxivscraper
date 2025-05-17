"""
arXiv Scraper - A tool for scraping arXiv papers and extracting LaTeX content.
"""

from arxivscraper.scraper import ArxivScraper
from arxivscraper.analyzer import DatasetAnalyzer

__version__ = "0.1.0"
__all__ = ["ArxivScraper", "DatasetAnalyzer"]