"""
SQL Agent Service for Backend
Uses LangChain SQL Agent with SQLite database for CSV queries (LangChain v0.3)
"""

import os
import sys
from typing import Dict, Optional
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from backend.core.sqlite_manager import SQLiteManager


class SQLAgentService:
    """Service for managing SQL Agent in FastAPI backend using LangChain v0.3."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize SQL Agent Service.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.sqlite_manager = SQLiteManager()
        self.current_agent = None
        self.current_table_name = None
        self.db = None
        self.llm = None
    
    def _initialize_llm(self):
        """Initialize LLM if not already initialized."""
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.environ.get("OPENAI_API_KEY")
            )
    
    def _initialize_db_connection(self, table_name: Optional[str] = None):
        """
        Initialize SQLDatabase connection.
        
        Args:
            table_name: Optional table name to restrict access to specific table
        """
        # Reinitialize if table changed to ensure proper scoping
        if self.db is None or self.current_table_name != table_name:
            connection_string = self.sqlite_manager.get_connection_string()
            # If table_name is provided, include only that table
            if table_name:
                self.db = SQLDatabase.from_uri(
                    connection_string,
                    include_tables=[table_name],
                    sample_rows_in_table_info=3
                )
            else:
                self.db = SQLDatabase.from_uri(connection_string)
    
    def _get_agent_for_table(self, table_name: str):
        """
        Get or create SQL Agent for a specific table using LangChain v0.3.
        
        Args:
            table_name: Name of the SQLite table
        """
        # Only create new agent if table changed
        if self.current_table_name != table_name:
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize DB connection with focus on this table
            self._initialize_db_connection(table_name)
            
            # Create SQL agent using create_sql_agent (LangChain v0.3 compatible)
            self.current_agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="openai-tools",
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=30
            )
            
            self.current_table_name = table_name
    
    def query(self, table_name: str, question: str) -> Dict[str, str]:
        """
        Query SQLite table using SQL Agent.
        
        Args:
            table_name: Name of the SQLite table
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Verify table exists
            if not self.sqlite_manager.table_exists(table_name):
                return {
                    "answer": f"Error: Table '{table_name}' does not exist in the database.",
                    "source": "sql_agent",
                    "table": table_name,
                    "error": "Table not found"
                }
            
            # Get agent for this table
            self._get_agent_for_table(table_name)
            
            # Run agent with the question
            response = self.current_agent.invoke({"input": question})
            
            # Extract answer from response
            if isinstance(response, dict):
                answer = response.get("output", str(response))
            else:
                answer = str(response)
            
            return {
                "answer": answer,
                "source": "sql_agent",
                "table": table_name
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing SQL query: {str(e)}",
                "source": "sql_agent",
                "table": table_name,
                "error": str(e)
            }
    
    def get_table_name_for_file(self, file_name: str) -> str:
        """
        Get SQLite table name for a CSV file.
        
        Args:
            file_name: Name of the CSV file
            
        Returns:
            Sanitized table name
        """
        return self.sqlite_manager.sanitize_table_name(file_name)

