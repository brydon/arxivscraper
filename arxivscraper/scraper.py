"""
Core scraper functionality for arxivscraper.
"""

import arxiv
import os
import json
import tarfile
import time
import structlog
import datetime
import pandas as pd
from pathlib import Path
import re
import shutil
import tempfile
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set

logger = structlog.get_logger(__name__)


class ArxivScraper:
    def __init__(self, 
                output_dir: str = "arxiv_papers", 
                temp_dir: str = "temp_downloads",
                max_papers_per_category: int = 5000,
                save_interval: int = 100,
                start_date: str = "2024-01-01") -> None:
        """
        Initialize the ArxivScraper.
        
        Args:
            output_dir: Directory to save the dataset
            temp_dir: Directory for temporary downloads
            max_papers_per_category: Maximum number of papers to scrape per category
            save_interval: How often to save the dataset (number of papers)
            start_date: The start date for scraping papers (format: YYYY-MM-DD)
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.max_papers_per_category = max_papers_per_category
        self.save_interval = save_interval
        self.start_date = start_date
        
        # Create output and temp directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize dataset
        self.dataset = []
        self.dataset_file = self.output_dir / "arxiv_dataset.json"
        
        # Load existing dataset if it exists
        if self.dataset_file.exists():
            with open(self.dataset_file, 'r') as f:
                self.dataset = json.load(f)
            logger.info(f"Loaded existing dataset with {len(self.dataset)} papers.")
        
        # Track processed papers to avoid duplicates
        self.processed_ids = {paper['title'] for paper in self.dataset}
        
        # Configure arXiv client with conservative settings to avoid rate limiting
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=5
        )
    
    def build_query(self, categories: List[str]) -> str:
        """
        Build a query string for the arXiv API.
        
        Args:
            categories: List of arXiv categories to search
            
        Returns:
            Query string for the arXiv API
        """
        # Create a query that searches for papers in the given categories
        # and published after the start date
        category_query = " OR ".join([f"cat:{cat}*" for cat in categories])
        date_query = f"submittedDate:[{self.start_date}T00:00:00Z TO now]"
        return f"({category_query}) AND {date_query}"
    
    def extract_latex_from_source(self, source_path: Path) -> Optional[str]:
        """
        Extract LaTeX content from a .tar.gz source file.
        
        Args:
            source_path: Path to the .tar.gz source file
            
        Returns:
            LaTeX content as a string or None if extraction fails
        """
        try:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                # Extract the tar.gz file
                with tarfile.open(source_path, 'r:gz') as tar:
                    tar.extractall(path=temp_extract_dir)
                
                # Find main .tex file (usually the one with \begin{document})
                main_tex = None
                for tex_file in Path(temp_extract_dir).glob('**/*.tex'):
                    with open(tex_file, 'r', errors='ignore') as f:
                        content = f.read()
                        if r'\begin{document}' in content:
                            main_tex = tex_file
                            break
                
                if main_tex is None:
                    # Try to find any .tex file if main wasn't found
                    tex_files = list(Path(temp_extract_dir).glob('**/*.tex'))
                    if tex_files:
                        main_tex = tex_files[0]
                
                if main_tex:
                    with open(main_tex, 'r', errors='ignore') as f:
                        return f.read()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting LaTeX: {str(e)}")
            return None
    
    def determine_latex_category(self, latex_content: str) -> str:
        """
        Attempt to determine the main LaTeX category from its content.
        
        Args:
            latex_content: LaTeX content as a string
            
        Returns:
            Category as a string ('theorem', 'algorithm', 'experiment', etc.)
        """
        # Simple heuristic based on common LaTeX commands and environments
        categories = {
            'theorem': [r'\\begin{theorem}', r'\\begin{lemma}', r'\\begin{corollary}', r'\\newtheorem'],
            'algorithm': [r'\\begin{algorithm}', r'\\procedure', r'\\algorithmiccomment', r'\\State'],
            'experiment': [r'\\experiment', r'experimental setup', r'data collection', r'results show'],
            'review': [r'\\review', r'literature review', r'survey', r'we review', r'overview of'],
            'proof': [r'\\begin{proof}', r'\\end{proof}', r'we prove', r'proof of']
        }
        
        counts = {cat: 0 for cat in categories}
        
        for cat, patterns in categories.items():
            for pattern in patterns:
                counts[cat] += len(re.findall(pattern, latex_content, re.IGNORECASE))
        
        # Return category with highest count, or 'other' if all zero
        max_cat = max(counts.items(), key=lambda x: x[1])
        return max_cat[0] if max_cat[1] > 0 else 'other'
    
    def process_paper(self, paper: arxiv.Result) -> Optional[Dict[str, Any]]:
        """
        Process a single paper from arXiv.
        
        Args:
            paper: arXiv paper result object
            
        Returns:
            Dictionary with paper data or None if processing fails
        """
        source_path = None
        try:
            if paper.title in self.processed_ids:
                logger.debug(f"Skipping already processed paper: {paper.title}")
                return None
            
            logger.info(f"Processing paper: {paper.title}")
            
            # Download source files
            source_path = self.temp_dir / f"{paper.get_short_id()}.tar.gz"
            paper.download_source(dirpath=source_path.parent, filename=source_path.name)
            
            # Extract LaTeX content
            latex_content = self.extract_latex_from_source(source_path)
            if not latex_content:
                logger.warning(f"Could not extract LaTeX content from {paper.title}")
                return None
            
            # Determine LaTeX category
            latex_category = self.determine_latex_category(latex_content)
            
            # Create paper entry
            paper_data = {
                "authors": [author.name for author in paper.authors],
                "title": paper.title,
                "abstract": paper.summary,
                "latex": latex_content,
                "category": latex_category,
                "arxiv_id": paper.get_short_id(),
                "published": paper.published.strftime("%Y-%m-%d"),
                "primary_category": paper.primary_category,
                "pdf_url": paper.pdf_url
            }
            
            # Add to processed IDs
            self.processed_ids.add(paper.title)
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.title}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        finally:
            # Clean up downloaded file
            if source_path and source_path.exists():
                source_path.unlink()
    
    def save_dataset(self, final: bool = False) -> None:
        """
        Save the dataset to disk.
        
        Args:
            final: Whether this is the final save (add timestamp for non-final saves)
        """
        # Save the main dataset
        with open(self.dataset_file, 'w') as f:
            json.dump(self.dataset, f)
        
        # Create a backup with timestamp if not the final save
        if not final:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.output_dir / "tmp" / f"arxiv_dataset_backup_{timestamp}.json"
            shutil.copy(self.dataset_file, backup_file)
        
        logger.info(f"Dataset saved with {len(self.dataset)} papers.")
    
    def scrape_categories(self, categories: List[str]) -> None:
        """
        Scrape papers from the specified arXiv categories.
        
        Args:
            categories: List of arXiv categories to scrape (e.g., ['math', 'cs', 'physics'])
        """
        logger.info(f"Starting scraping for categories: {categories}")
        
        # Build the query
        query = self.build_query(categories)
        logger.info(f"Using query: {query}")
        
        # Set up the search
        search = arxiv.Search(
            query=query, 
            max_results=self.max_papers_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Get the results
        papers_processed = 0
        
        try:
            from tqdm import tqdm
            for paper in tqdm(self.client.results(search), total=self.max_papers_per_category, 
                            desc=f"Scraping {', '.join(categories)}"):
                
                paper_data = self.process_paper(paper)
                
                if paper_data:
                    self.dataset.append(paper_data)
                    papers_processed += 1
                
                # Save periodically
                if papers_processed > 0 and papers_processed % self.save_interval == 0:
                    logger.info(f"Processed {papers_processed} papers. Saving checkpoint...")
                    self.save_dataset()
                
                # Add a small delay to be nice to arXiv servers
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            logger.error(traceback.format_exc())
            # Save what we have so far
            self.save_dataset()
        
        # Final save for this category
        if papers_processed > 0:
            self.save_dataset()
    
    def create_stats(self) -> None:
        """
        Create and save statistics about the dataset.
        """
        if not self.dataset:
            logger.warning("No data to generate statistics.")
            return
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame(self.dataset)
        
        # Generate stats
        stats = {
            "total_papers": len(df),
            "papers_by_category": df['category'].value_counts().to_dict(),
            "papers_by_arxiv_category": df['primary_category'].value_counts().to_dict(),
            "papers_by_date": df['published'].value_counts().to_dict(),
            "average_authors_per_paper": df['authors'].apply(len).mean(),
            "total_unique_authors": len(set([author for authors in df['authors'] for author in authors]))
        }
        
        # Save stats
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
        
        # Print summary
        print("\n===== Dataset Statistics =====")
        print(f"Total papers: {stats['total_papers']}")
        print("\nPapers by category:")
        for cat, count in sorted(stats['papers_by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
        print(f"\nAverage authors per paper: {stats['average_authors_per_paper']:.2f}")
        print(f"Total unique authors: {stats['total_unique_authors']}")
        print("=============================\n")