"""
Request Models for FastAPI
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="User's question")
    selected_file: str = Field(..., description="Selected document name")
    session_id: str = Field(..., description="Session identifier for memory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is this document about?",
                "selected_file": "document.pdf",
                "session_id": "user_123_session_456"
            }
        }