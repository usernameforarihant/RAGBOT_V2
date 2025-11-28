"""
Response Models for FastAPI
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    status: str = Field(..., description="Status of upload")
    file_name: str = Field(..., description="Name of uploaded file")
    files_available: List[str] = Field(..., description="List of all available files")
    message: str = Field(..., description="Status message")
    embeddings_created: bool = Field(..., description="Whether new embeddings were created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "file_name": "document.pdf",
                "files_available": ["document.pdf", "report.docx"],
                "message": "File processed successfully",
                "embeddings_created": True
            }
        }


class ContextDocument(BaseModel):
    """Model for context document."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    context: List[ContextDocument] = Field(..., description="Retrieved context documents")
    memory: Dict[str, Any] = Field(..., description="Conversation memory state")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "This document discusses...",
                "context": [
                    {
                        "content": "Sample text from document...",
                        "metadata": {"page": 1, "source": "document.pdf"}
                    }
                ],
                "memory": {
                    "messages": 5,
                    "last_updated": "2024-01-01T00:00:00"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = Field(default="error", description="Status")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")