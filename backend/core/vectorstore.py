"""
Vector Store Management for Backend
Wraps the existing VectorStoreManager from utils
"""

import os
import sys
from typing import List, Tuple
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.vector_store import VectorStoreManager
from utils.document_processor import DocumentProcessor
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()


class VectorStoreService:
    """Service for managing vector stores in FastAPI backend."""
    
    def __init__(self, chroma_dir: str = "data/chroma_db"):
        """
        Initialize vector store service.
        
        Args:
            chroma_dir: Directory for ChromaDB storage
        """
        self.chroma_dir = chroma_dir
        self.document_processor = DocumentProcessor()
        
        # Ensure directory exists
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    
    def check_embeddings_exist(self, collection_name: str) -> bool:
        """
        Check if embeddings exist for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if embeddings exist
        """
        vector_store_path = os.path.join(self.chroma_dir, collection_name)
        return os.path.exists(vector_store_path) and len(os.listdir(vector_store_path)) > 0
    
    def create_or_load_vectorstore(
        self,
        source: str,
        collection_name: str,
        source_type: str = None
    ) -> Tuple[VectorStoreManager, bool]:
        """
        Create or load vector store for a document.
        
        Args:
            source: File path or URL
            collection_name: Collection name
            source_type: Type of source (pdf, docx, txt, url)
            
        Returns:
            Tuple of (VectorStoreManager, embeddings_created)
        """
        vector_store_manager = VectorStoreManager(self.chroma_dir)
        
        if self.check_embeddings_exist(collection_name):
            # Load existing embeddings
            vector_store_manager.load_existing(collection_name)
            return vector_store_manager, False
        else:
            # Create new embeddings
            documents = self.document_processor.load_document(source, source_type)
            text_chunks = self.document_processor.split_documents(documents)
            vector_store_manager.create_vector_store(text_chunks, collection_name)
            return vector_store_manager, True
    
    def load_existing_vectorstore(self, collection_name: str) -> VectorStoreManager:
        """
        Load existing vector store.
        
        Args:
            collection_name: Collection name
            
        Returns:
            VectorStoreManager instance
            
        Raises:
            ValueError: If embeddings don't exist
        """
        if not self.check_embeddings_exist(collection_name):
            raise ValueError(f"Embeddings not found for collection: {collection_name}")
        
        vector_store_manager = VectorStoreManager(self.chroma_dir)
        vector_store_manager.load_existing(collection_name)
        return vector_store_manager
    
    def process_and_store(
        self,
        source: str,
        collection_name: str,
        source_type: str = None
    ) -> Tuple[VectorStoreManager, int, bool]:
        """
        Process document and store in vector database.
        
        Args:
            source: File path or URL
            collection_name: Collection name
            source_type: Type of source (pdf, docx, txt, url)
            
        Returns:
            Tuple of (VectorStoreManager, num_chunks, embeddings_created)
        """
        vector_store_manager = VectorStoreManager(self.chroma_dir)
        
        if self.check_embeddings_exist(collection_name):
            # Load existing
            vector_store_manager.load_existing(collection_name)
            # Try to get count (approximation)
            num_chunks = 0
            return vector_store_manager, num_chunks, False
        else:
            # Create new
            documents = self.document_processor.load_document(source, source_type)
            text_chunks = self.document_processor.split_documents(documents)
            vector_store_manager.create_vector_store(text_chunks, collection_name)
            return vector_store_manager, len(text_chunks), True