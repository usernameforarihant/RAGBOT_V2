"""
RAG Pipeline for Backend
Wraps the existing RAGChain from utils
"""

import os
import sys
from typing import Dict, List, Tuple
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.rag_chain import RAGChain
from utils.vector_store import VectorStoreManager

from dotenv import load_dotenv
load_dotenv()

class RAGPipeline:
    """RAG pipeline service for FastAPI backend."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.rag_chain = None
        self.vector_store_manager = None
    
    def initialize_chain(self, vector_store_manager: VectorStoreManager) -> None:
        """
        Initialize RAG chain with vector store.
        
        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store_manager = vector_store_manager
        self.rag_chain = RAGChain(vector_store_manager.vector_store)
    
    def query(
        self,
        question: str,
        chat_history: List[Dict] = None
    ) -> Dict:
        """
        Execute RAG query.
        
        Args:
            question: User question
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with answer and source documents
            
        Raises:
            ValueError: If RAG chain not initialized
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize_chain first.")
        
        return self.rag_chain.get_response(question, chat_history)
    
    def get_answer_with_sources(
        self,
        question: str,
        chat_history: List[Dict] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Get answer with formatted sources.
        
        Args:
            question: User question
            chat_history: Previous conversation history
            
        Returns:
            Tuple of (answer, context_documents)
        """
        response = self.query(question, chat_history)
        
        # Format context documents
        context_docs = []
        for doc in response.get("source_documents", []):
            context_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return response["answer"], context_docs
    
    def update_retriever_k(self, k: int) -> None:
        """
        Update number of documents to retrieve.
        
        Args:
            k: Number of documents
            
        Raises:
            ValueError: If RAG chain not initialized
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize_chain first.")
        
        self.rag_chain.update_retriever(k)
    
    def is_initialized(self) -> bool:
        """
        Check if RAG chain is initialized.
        
        Returns:
            True if initialized
        """
        return self.rag_chain is not None