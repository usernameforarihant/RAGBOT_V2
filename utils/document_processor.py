"""Document Processing Module
Handles PDF, DOCX, TXT, and URL loading using LangChain v0.3.x"""

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os


class DocumentProcessor:
    """Handles document loading and processing for multiple formats."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 300):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF using PyMuPDFLoader.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = file_path
            doc.metadata['type'] = 'pdf'
        
        return documents
    
    def load_docx(self, file_path: str) -> List[Document]:
        """
        Load DOCX using Docx2txtLoader.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List of Document objects
        """
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = file_path
            doc.metadata['type'] = 'docx'
        
        return documents
    
    def load_txt(self, file_path: str) -> List[Document]:
        """
        Load TXT using TextLoader.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            List of Document objects
        """
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = file_path
            doc.metadata['type'] = 'txt'
        
        return documents
    
    def load_url(self, url: str) -> List[Document]:
        """
        Load content from URL using WebBaseLoader.
        
        Args:
            url: URL to load
            
        Returns:
            List of Document objects
        """
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = url
            doc.metadata['type'] = 'url'
        
        return documents
    
    def load_document(self, source: str, source_type: str = None) -> List[Document]:
        """
        Load document based on file type or URL.
        
        Args:
            source: File path or URL
            source_type: Type of source ('pdf', 'docx', 'txt', 'url')
                        If None, will be inferred from file extension   x
            
        Returns:
            List of Document objects
        """
        # Infer type if not provided
        if source_type is None:
            if source.startswith('http://') or source.startswith('https://'):
                source_type = 'url'
            else:
                ext = os.path.splitext(source)[1].lower()
                type_map = {
                    '.pdf': 'pdf',
                    '.docx': 'docx',
                    '.doc': 'docx',
                    '.txt': 'txt'
                }
                source_type = type_map.get(ext)
        
        # Load based on type
        if source_type == 'pdf':
            return self.load_pdf(source)
        elif source_type == 'docx':
            return self.load_docx(source)
        elif source_type == 'txt':
            return self.load_txt(source)
        elif source_type == 'url':
            return self.load_url(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
        
        return chunks
    
    def process_document(self, source: str, source_type: str = None) -> List[Document]:
        """
        Complete pipeline: load and split document.
        
        Args:
            source: File path or URL
            source_type: Type of source (optional)
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_document(source, source_type)
        chunks = self.split_documents(documents)
        return chunks