"""
CSV Agent Service for Backend
Wraps the CSV Agent for FastAPI usage
"""

import os
import sys
from typing import Dict
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.csv_agent import CSVAgent


class CSVAgentService:
    """Service for managing CSV Agent in FastAPI backend."""
    
    def __init__(self):
        """Initialize CSV Agent Service."""
        self.current_agent = None
        self.current_csv_path = None
    
    def initialize_agent(self, csv_file_path: str) -> None:
        """
        Initialize CSV Agent for a specific file.
        
        Args:
            csv_file_path: Path to CSV file
        """
        # Only create new agent if file changed
        if self.current_csv_path != csv_file_path:
            self.current_agent = CSVAgent(csv_file_path)
            self.current_csv_path = csv_file_path
    
    def query(self, csv_file_path: str, question: str) -> Dict[str, str]:
        """
        Query CSV using agent.
        
        Args:
            csv_file_path: Path to CSV file
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        # Initialize agent for this CSV
        self.initialize_agent(csv_file_path)
        
        # Query the agent
        result = self.current_agent.query(question)
        
        return result
    
    def is_csv_file(self, filename: str) -> bool:
        """
        Check if file is a CSV.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if CSV file
        """
        return filename.lower().endswith('.csv')