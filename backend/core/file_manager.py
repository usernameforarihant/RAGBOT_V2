"""
File Management Module
Handles file uploads, storage, and listing
"""

import os
from pathlib import Path
from typing import List, Tuple
import hashlib


class FileManager:
    """Manages file operations for the RAG system."""
    
    def __init__(self, upload_dir: str = "data/uploaded_docs", urls_file: str = "data/saved_urls.txt"):
        """
        Initialize file manager.
        
        Args:
            upload_dir: Directory for uploaded files
            urls_file: File to store URLs
        """
        self.upload_dir = upload_dir
        self.urls_file = urls_file
        
        # Create directories
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)
    
    def save_file(self, file_content: bytes, file_name: str) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: File content as bytes
            file_name: Name of the file
            
        Returns:
            Full path to saved file
        """
        file_path = os.path.join(self.upload_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        return file_path
    
    def save_url(self, url: str) -> None:
        """
        Save URL to file.
        
        Args:
            url: URL to save
        """
        # Check if URL already exists
        existing_urls = []
        if os.path.exists(self.urls_file):
            with open(self.urls_file, 'r') as f:
                existing_urls = [line.strip() for line in f if line.strip()]
        
        # Only add if not already present
        if url not in existing_urls:
            with open(self.urls_file, 'a') as f:
                f.write(url + '\n')
    
    def get_all_documents(self) -> List[str]:
        """
        Get list of all uploaded documents and URLs.
        
        Returns:
            List of document names
        """
        docs = []
        
        # Get uploaded files
        if os.path.exists(self.upload_dir):
            files = [f for f in os.listdir(self.upload_dir) 
                    if f.endswith(('.pdf', '.docx', '.doc', '.txt'))]
            docs.extend(sorted(files))
        
        # Get saved URLs
        if os.path.exists(self.urls_file):
            with open(self.urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
                docs.extend([f"[URL] {url}" for url in urls])
        
        return docs
    
    def get_file_path(self, file_name: str) -> str:
        """
        Get full path for a file.
        
        Args:
            file_name: Name of the file
            
        Returns:
            Full file path
        """
        return os.path.join(self.upload_dir, file_name)
    
    def get_document_type(self, doc_name: str) -> str:
        """
        Determine document type from name.
        
        Args:
            doc_name: Document name
            
        Returns:
            Document type (pdf, docx, txt, url)
        """
        if doc_name.startswith('[URL]'):
            return 'url'
        ext = os.path.splitext(doc_name)[1].lower()
        type_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'txt'
        }
        return type_map.get(ext, 'unknown')
    
    def get_collection_name(self, doc_name: str) -> str:
        """
        Generate collection name for document.
        
        Args:
            doc_name: Document name
            
        Returns:
            Collection name for ChromaDB
        """
        if doc_name.startswith('[URL]'):
            url = doc_name.replace('[URL] ', '')
            # Create safe collection name from URL
            return 'url_' + hashlib.md5(url.encode()).hexdigest()[:16]
        else:
            # Remove extension and clean name
            name = doc_name.replace('.pdf', '').replace('.docx', '').replace('.doc', '').replace('.txt', '')
            # Replace spaces and special characters
            name = name.replace(' ', '_').replace('-', '_').lower()
            # Remove any non-alphanumeric characters except underscore
            name = ''.join(c for c in name if c.isalnum() or c == '_')
            return name
    
    def file_exists(self, file_name: str) -> bool:
        """
        Check if file exists.
        
        Args:
            file_name: Name of the file
            
        Returns:
            True if file exists
        """
        file_path = self.get_file_path(file_name)
        return os.path.exists(file_path)
    
    def get_source_path(self, doc_name: str) -> str:
        """
        Get source path for document (file path or URL).
        
        Args:
            doc_name: Document name
            
        Returns:
            Source path or URL
        """
        if doc_name.startswith('[URL]'):
            return doc_name.replace('[URL] ', '')
        else:
            return self.get_file_path(doc_name)