import argparse
import logging
from pathlib import Path

from arxivscraper.scraper import ArxivScraper
from arxivscraper.analyzer import DatasetAnalyzer


def main():
    """Command line interface for arxivscraper."""
    parser = argparse.ArgumentParser(
        description="Scrape and analyze arXiv papers"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Scrape papers from arXiv")
    scraper_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="arxiv_dataset",
        help="Directory to save the dataset"
    )
    scraper_parser.add_argument(
        "--temp-dir", 
        type=str, 
        default="/tmp/arxivdownloads/",
        help="Directory for temporary downloads"
    )
    scraper_parser.add_argument(
        "--max-papers", 
        type=int, 
        default=5000,
        help="Maximum number of papers to scrape per category"
    )
    scraper_parser.add_argument(
        "--save-interval", 
        type=int, 
        default=50,
        help="How often to save the dataset (number of papers)"
    )
    scraper_parser.add_argument(
        "--start-date", 
        type=str, 
        default="2024-01-01",
        help="Start date for scraping papers (format: YYYY-MM-DD)"
    )
    scraper_parser.add_argument(
        "--categories", 
        type=str, 
        nargs="+", 
        default=["math", "physics", "cs"],
        help="arXiv categories to scrape"
    )
    scraper_parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    # Analyzer command
    analyzer_parser = subparsers.add_parser("analyze", help="Analyze arXiv dataset")
    analyzer_parser.add_argument(
        "dataset",
        type=str,
        help="Path to the JSON dataset file"
    )
    analyzer_parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save analysis results (default: same directory as dataset)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if hasattr(args, "log_level"):
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='arxivscraper.log'
        )
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, args.log_level))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    # Run the appropriate command
    if args.command == "scrape":
        scraper = ArxivScraper(
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            max_papers_per_category=args.max_papers,
            save_interval=args.save_interval,
            start_date=args.start_date
        )
        
        for category in args.categories:
            scraper.scrape_categories([category])
        
        scraper.create_stats()
        scraper.save_dataset(final=True)
        
        print(f"Scraping completed! Dataset saved to {Path(args.output_dir) / 'arxiv_dataset.json'}")
    
    elif args.command == "analyze":
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(args.dataset).parent / "analysis"
            
        analyzer = DatasetAnalyzer(args.dataset, output_dir=output_dir)
        analyzer.analyze()
        
        print(f"Analysis completed! Results saved to {output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()