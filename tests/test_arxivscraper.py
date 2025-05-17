"""
Basic tests for the arxivscraper package.
"""

import pytest
import os
import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from arxivscraper import ArxivScraper, DatasetAnalyzer


class TestArxivScraper:
    """Tests for the ArxivScraper class."""
    
    def test_initialization(self):
        """Test that the scraper initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ArxivScraper(
                output_dir=os.path.join(temp_dir, "output"),
                temp_dir=os.path.join(temp_dir, "temp"),
                max_papers_per_category=10,
                save_interval=5,
                start_date="2024-01-01"
            )
            
            assert scraper.output_dir == Path(os.path.join(temp_dir, "output"))
            assert scraper.temp_dir == Path(os.path.join(temp_dir, "temp"))
            assert scraper.max_papers_per_category == 10
            assert scraper.save_interval == 5
            assert scraper.start_date == "2024-01-01"
            assert os.path.exists(os.path.join(temp_dir, "output"))
            assert os.path.exists(os.path.join(temp_dir, "temp"))
    
    def test_build_query(self):
        """Test the query building functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ArxivScraper(
                output_dir=temp_dir,
                start_date="2024-01-01"
            )
            
            query = scraper.build_query(["math", "cs"])
            assert "cat:math*" in query
            assert "cat:cs*" in query
            assert "OR" in query
            assert "submittedDate:[2024-01-01T00:00:00Z TO now]" in query
    
    def test_determine_latex_category(self):
        """Test the LaTeX category determination."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ArxivScraper(output_dir=temp_dir)
            
            # Test theorem category
            latex = r"\begin{theorem} This is a theorem \end{theorem}"
            assert scraper.determine_latex_category(latex) == "theorem"
            
            # Test algorithm category
            latex = r"\begin{algorithm} This is an algorithm \end{algorithm}"
            assert scraper.determine_latex_category(latex) == "algorithm"
            
            # Test other category
            latex = r"This is just some text without any special environments."
            assert scraper.determine_latex_category(latex) == "other"
    
    @patch('arxiv.Client')
    @patch('arxiv.Search')
    def test_save_dataset(self, mock_search, mock_client):
        """Test that the dataset is saved correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ArxivScraper(output_dir=temp_dir)
            
            # Add some dummy data
            scraper.dataset = [
                {"title": "Paper 1", "authors": ["Author 1"]},
                {"title": "Paper 2", "authors": ["Author 2"]}
            ]
            
            # Save the dataset
            scraper.save_dataset(final=True)
            
            # Check that the file exists
            dataset_file = Path(temp_dir) / "arxiv_dataset.json"
            assert dataset_file.exists()
            
            # Check that the content is correct
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                assert len(data) == 2
                assert data[0]["title"] == "Paper 1"
                assert data[1]["title"] == "Paper 2"


class TestDatasetAnalyzer:
    """Tests for the DatasetAnalyzer class."""
    
    def test_initialization(self):
        """Test that the analyzer initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy dataset
            dataset_file = Path(temp_dir) / "dataset.json"
            with open(dataset_file, 'w') as f:
                json.dump([
                    {
                        "title": "Paper 1",
                        "authors": ["Author 1"],
                        "abstract": "Abstract 1",
                        "latex": "Latex 1",
                        "category": "theorem",
                        "primary_category": "math.CO",
                        "published": "2024-01-15"
                    }
                ], f)
            
            # Initialize the analyzer
            with patch('pandas.DataFrame') as mock_df:
                mock_df.return_value = MagicMock()
                analyzer = DatasetAnalyzer(dataset_file)
                
                assert analyzer.dataset_path == dataset_file
                assert analyzer.output_dir == dataset_file.parent / "analysis"
                assert analyzer.output_dir.exists()