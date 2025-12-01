"""
SQLite Database Manager
Handles CSV file loading into SQLite database
"""

import sqlite3
import os
import pandas as pd
from pathlib import Path
from typing import Optional
import re


class SQLiteManager:
    """Manages SQLite database for CSV file storage."""
    
    def __init__(self, db_path: str = "data/csv_database.db"):
        """
        Initialize SQLite manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.close()
    
    @staticmethod
    def sanitize_table_name(filename: str) -> str:
        """
        Sanitize filename to create a valid SQLite table name.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized table name
        """
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        # Replace special characters with underscore
        sanitized = re.sub(r'[^A-Za-z0-9_]', '_', base_name)
        # Remove leading/trailing underscores and ensure it starts with letter or underscore
        sanitized = sanitized.strip('_')
        if not sanitized or sanitized[0].isdigit():
            sanitized = 'table_' + sanitized
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'csv_table'
        return sanitized.lower()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def load_csv_to_table(self, csv_file_path: str, table_name: Optional[str] = None) -> str:
        """
        Load CSV file into SQLite table.
        
        Args:
            csv_file_path: Path to CSV file
            table_name: Optional table name (if None, derived from filename)
            
        Returns:
            Name of the created/updated table
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Get table name from filename if not provided
        if table_name is None:
            filename = os.path.basename(csv_file_path)
            table_name = self.sanitize_table_name(filename)
        
        # Check if table already exists
        if self.table_exists(table_name):
            # Table exists, skip loading (idempotent)
            return table_name
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            # Try with different encoding
            df = pd.read_csv(csv_file_path, encoding='latin-1')
        
        # Connect to database and load data
        conn = sqlite3.connect(self.db_path)
        try:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        finally:
            conn.close()
        
        return table_name
    
    def get_table_schema(self, table_name: str) -> str:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema description as string
        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()
        
        schema_lines = [f"Table: {table_name}"]
        for col in columns:
            col_name, col_type = col[1], col[2]
            schema_lines.append(f"  - {col_name}: {col_type}")
        
        return "\n".join(schema_lines)
    
    def get_connection_string(self) -> str:
        """
        Get SQLAlchemy connection string for LangChain SQL Agent.
        
        Returns:
            SQLite connection string
        """
        return f"sqlite:///{os.path.abspath(self.db_path)}"


