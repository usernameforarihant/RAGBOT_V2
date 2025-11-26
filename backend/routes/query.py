"""
Query Route
Handles RAG queries with persistent memory
"""

from fastapi import APIRouter, HTTPException
from typing import List

from backend.core.file_manager import FileManager
from backend.core.vectorstore import VectorStoreService
from backend.core.rag_pipeline import RAGPipeline
from backend.core.memory_manager import MemoryService
from backend.models.request_models import QueryRequest
from backend.models.response_models import QueryResponse, ContextDocument, ErrorResponse


router = APIRouter(prefix="/query", tags=["query"])

# Initialize services
file_manager = FileManager()
vectorstore_service = VectorStoreService()
memory_service = MemoryService()


@router.post("/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document using RAG with persistent memory.
    
    - Loads the vector store for the selected file
    - Retrieves relevant context
    - Generates answer using RAG
    - Maintains conversation memory
    """
    try:
        question = request.question
        selected_file = request.selected_file
        session_id = request.session_id
        
        # Get collection name
        collection_name = file_manager.get_collection_name(selected_file)
        
        # Check if embeddings exist
        if not vectorstore_service.check_embeddings_exist(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Embeddings not found for {selected_file}. Please upload the file first."
            )
        
        # Load vector store
        vector_store_manager = vectorstore_service.load_existing_vectorstore(collection_name)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize_chain(vector_store_manager)
        
        # Load conversation memory
        memory, chat_history = memory_service.load_memory(collection_name, session_id)
        
        # Get answer with sources
        answer, context_docs = rag_pipeline.get_answer_with_sources(question, chat_history)
        
        # Save user question and assistant answer to memory
        memory_service.save_message(collection_name, session_id, "user", question)
        
        source_contents = [doc["content"] for doc in context_docs]
        memory_service.save_message(collection_name, session_id, "assistant", answer, source_contents)
        
        # Get updated memory state
        memory_state = memory_service.get_memory_state(collection_name, session_id)
        
        # Format context documents for response
        context_response = [
            ContextDocument(
                content=doc["content"],
                metadata=doc["metadata"]
            )
            for doc in context_docs
        ]
        
        return QueryResponse(
            answer=answer,
            context=context_response,
            memory=memory_state
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                status="error",
                message="Failed to process query",
                detail=str(e)
            ).dict()
        )


@router.delete("/memory")
async def clear_memory(selected_file: str, session_id: str):
    """
    Clear conversation memory for a document and session.
    """
    try:
        collection_name = file_manager.get_collection_name(selected_file)
        memory_service.clear_memory(collection_name, session_id)
        
        return {
            "status": "ok",
            "message": "Memory cleared successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                status="error",
                message="Failed to clear memory",
                detail=str(e)
            ).dict()
        )