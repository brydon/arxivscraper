"""
Dataset analysis functionality for arxivscraper.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np
from collections import Counter
import re
from typing import Dict, List, Any, Tuple, Optional, Union

class DatasetAnalyzer:
    def __init__(self, dataset_path: str, output_dir: Optional[str] = None) -> None:
        """
        Initialize the DatasetAnalyzer.
        
        Args:
            dataset_path: Path to the JSON dataset file
            output_dir: Path to directory for saving analysis results
        """
        self.dataset_path = Path(dataset_path)
        
        if output_dir is None:
            self.output_dir = self.dataset_path.parent / "analysis"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load dataset
        with open(self.dataset_path, 'r') as f:
            self.data = json.load(f)
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        print(f"Loaded dataset with {len(self.df)} papers")
    
    def analyze(self) -> None:
        """Run all analyses"""
        self.analyze_categories()
        self.analyze_publication_trends()
        self.analyze_author_stats()
        self.analyze_abstracts()
        self.analyze_latex_content()
        print(f"Analysis complete. Results saved to {self.output_dir}")
    
    def analyze_categories(self) -> None:
        """Analyze the distribution of paper categories"""
        plt.figure(figsize=(12, 6))
        
        # Paper categories
        cat_counts = self.df['category'].value_counts()
        ax = sns.barplot(x=cat_counts.index, y=cat_counts.values)
        plt.title('Papers by LaTeX Category')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels on top of bars
        for i, count in enumerate(cat_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latex_categories.png')
        
        # arXiv primary categories
        plt.figure(figsize=(14, 8))
        primary_cat_counts = self.df['primary_category'].value_counts().head(20)  # Top 20
        ax = sns.barplot(x=primary_cat_counts.index, y=primary_cat_counts.values)
        plt.title('Papers by arXiv Primary Category (Top 20)')
        plt.xlabel('arXiv Category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        
        # Add count labels on top of bars
        for i, count in enumerate(primary_cat_counts.values):
            ax.text(i, count + 2, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'arxiv_categories.png')
    
    def analyze_publication_trends(self) -> None:
        """Analyze publication trends over time"""
        self.df['published'] = pd.to_datetime(self.df['published'])
        self.df['month'] = self.df['published'].dt.to_period('M')
        
        # Papers per month
        monthly_counts = self.df.groupby('month').size()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(kind='bar')
        plt.title('Papers Published per Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Papers')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'papers_per_month.png')
        
        # Papers per month by category
        plt.figure(figsize=(14, 8))
        category_monthly = self.df.groupby(['month', 'category']).size().unstack()
        category_monthly.plot(kind='bar', stacked=True)
        plt.title('Papers by Category and Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Papers')
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'papers_by_category_month.png')
    
    def analyze_author_stats(self) -> None:
        """Analyze author statistics"""
        # Number of authors per paper
        self.df['author_count'] = self.df['authors'].apply(len)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['author_count'], bins=range(1, 16), kde=False)
        plt.title('Distribution of Authors per Paper')
        plt.xlabel('Number of Authors')
        plt.ylabel('Count')
        plt.xticks(range(1, 16))
        plt.tight_layout()
        plt.savefig(self.output_dir / 'authors_per_paper.png')
        
        # Most prolific authors
        all_authors = [author for authors in self.df['authors'] for author in authors]
        author_counts = Counter(all_authors)
        top_authors = author_counts.most_common(20)
        
        plt.figure(figsize=(12, 8))
        authors, counts = zip(*top_authors)
        sns.barplot(x=list(counts), y=list(authors))
        plt.title('Top 20 Most Prolific Authors')
        plt.xlabel('Number of Papers')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_authors.png')
    
    def analyze_abstracts(self) -> None:
        """Analyze paper abstracts"""
        # Word cloud of abstracts
        try:
            from wordcloud import WordCloud
            
            abstracts_text = ' '.join(self.df['abstract'])
            
            # Clean the text (remove common words)
            cleaned_text = re.sub(r'\b(?:the|and|of|in|to|a|is|for|we|that|this|are)\b', '', 
                                abstracts_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            wordcloud = WordCloud(width=1200, height=800, background_color='white', 
                                max_words=200, contour_width=3).generate(cleaned_text)
            
            plt.figure(figsize=(16, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'abstract_wordcloud.png')
        except ImportError:
            print("WordCloud package not installed. Skipping word cloud generation.")
        
        # Word frequency by category
        category_words = {}
        for category in self.df['category'].unique():
            abstracts = ' '.join(self.df[self.df['category'] == category]['abstract'])
            cleaned = re.sub(r'\b(?:the|and|of|in|to|a|is|for|we|that|this|are)\b', '', 
                            abstracts, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Count words
            words = re.findall(r'\b\w+\b', cleaned.lower())
            category_words[category] = Counter(words).most_common(10)
        
        # Plot top words by category
        plt.figure(figsize=(16, 12))
        num_categories = len(category_words)
        rows = (num_categories + 1) // 2
        
        for i, (category, word_counts) in enumerate(category_words.items(), 1):
            plt.subplot(rows, 2, i)
            words, counts = zip(*word_counts)
            sns.barplot(x=list(counts), y=list(words))
            plt.title(f'Top Words in {category.capitalize()} Papers')
            plt.xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'words_by_category.png')
    
    def analyze_latex_content(self) -> None:
        """Analyze LaTeX content"""
        # LaTeX command usage
        command_patterns = {
            'theorem': r'\\begin{theorem}',
            'lemma': r'\\begin{lemma}',
            'proof': r'\\begin{proof}',
            'equation': r'\\begin{equation}',
            'figure': r'\\begin{figure}',
            'table': r'\\begin{table}',
            'algorithm': r'\\begin{algorithm}',
            'itemize': r'\\begin{itemize}',
            'enumerate': r'\\begin{enumerate}',
            'section': r'\\section',
            'subsection': r'\\subsection',
            'bibliography': r'\\bibliography'
        }
        
        # Count occurrences by category
        command_counts = {category: {cmd: 0 for cmd in command_patterns} 
                          for category in self.df['category'].unique()}
        
        for i, row in self.df.iterrows():
            category = row['category']
            latex = row['latex']
            
            for cmd, pattern in command_patterns.items():
                command_counts[category][cmd] += len(re.findall(pattern, latex))
        
        # Create heatmap of command usage by category
        plt.figure(figsize=(14, 10))
        heatmap_data = pd.DataFrame(command_counts).T
        
        # Normalize by number of papers in each category
        category_counts = self.df['category'].value_counts()
        for category in heatmap_data.index:
            heatmap_data.loc[category] = heatmap_data.loc[category] / category_counts[category]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Average LaTeX Command Usage by Category')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latex_command_usage.png')
        
        # Generate a report of common LaTeX packages
        package_pattern = r'\\usepackage(?:\[.*?\])?\{(.*?)\}'
        package_counts = Counter()
        
        for latex in self.df['latex']:
            packages = re.findall(package_pattern, latex)
            package_counts.update(packages)
        
        top_packages = package_counts.most_common(30)
        
        with open(self.output_dir / 'latex_packages.txt', 'w') as f:
            f.write("Top LaTeX Packages Used:\n")
            f.write("======================\n\n")
            for package, count in top_packages:
                f.write(f"{package}: {count} papers\n")