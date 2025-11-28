# """
# Query Route
# Handles RAG queries with persistent memory
# """

# from fastapi import APIRouter, HTTPException
# from typing import List

# from backend.core.file_manager import FileManager
# from backend.core.vectorstore import VectorStoreService
# from backend.core.rag_pipeline import RAGPipeline
# from backend.core.memory_manager import MemoryService
# from backend.models.request_models import QueryRequest
# from backend.models.response_models import QueryResponse, ContextDocument, ErrorResponse


# router = APIRouter(prefix="/query", tags=["query"])

# # Initialize services
# file_manager = FileManager()
# vectorstore_service = VectorStoreService()
# memory_service = MemoryService()


# @router.post("/", response_model=QueryResponse)
# async def query_document(request: QueryRequest):
#     """
#     Query a document using RAG with persistent memory.
    
#     - Loads the vector store for the selected file
#     - Retrieves relevant context
#     - Generates answer using RAG
#     - Maintains conversation memory
#     """
#     try:
#         question = request.question
#         selected_file = request.selected_file
#         session_id = request.session_id
        
#         # Get collection name
#         collection_name = file_manager.get_collection_name(selected_file)
        
#         # Check if embeddings exist
#         if not vectorstore_service.check_embeddings_exist(collection_name):
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Embeddings not found for {selected_file}. Please upload the file first."
#             )
        
#         # Load vector store
#         vector_store_manager = vectorstore_service.load_existing_vectorstore(collection_name)
        
#         # Initialize RAG pipeline
#         rag_pipeline = RAGPipeline()
#         rag_pipeline.initialize_chain(vector_store_manager)
        
#         # Load conversation memory
#         memory, chat_history = memory_service.load_memory(collection_name, session_id)
        
#         # Get answer with sources
#         answer, context_docs = rag_pipeline.get_answer_with_sources(question, chat_history)
        
#         # Save user question and assistant answer to memory
#         memory_service.save_message(collection_name, session_id, "user", question)
        
#         source_contents = [doc["content"] for doc in context_docs]
#         memory_service.save_message(collection_name, session_id, "assistant", answer, source_contents)
        
#         # Get updated memory state
#         memory_state = memory_service.get_memory_state(collection_name, session_id)
        
#         # Format context documents for response
#         context_response = [
#             ContextDocument(
#                 content=doc["content"],
#                 metadata=doc["metadata"]
#             )
#             for doc in context_docs
#         ]
        
#         return QueryResponse(
#             answer=answer,
#             context=context_response,
#             memory=memory_state
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=ErrorResponse(
#                 status="error",
#                 message="Failed to process query",
#                 detail=str(e)
#             ).dict()
#         )


# @router.delete("/memory")
# async def clear_memory(selected_file: str, session_id: str):
#     """
#     Clear conversation memory for a document and session.
#     """
#     try:
#         collection_name = file_manager.get_collection_name(selected_file)
#         memory_service.clear_memory(collection_name, session_id)
        
#         return {
#             "status": "ok",
#             "message": "Memory cleared successfully"
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=ErrorResponse(
#                 status="error",
#                 message="Failed to clear memory",
#                 detail=str(e)
#             ).dict()
#         )

"""
Query Route
Handles RAG queries with persistent memory (and CSV Agent for CSV files)
"""

from fastapi import APIRouter, HTTPException
from typing import List

from backend.core.file_manager import FileManager
from backend.core.vectorstore import VectorStoreService
from backend.core.rag_pipeline import RAGPipeline
from backend.core.memory_manager import MemoryService
from backend.core.csv_agent_service import CSVAgentService
from backend.models.request_models import QueryRequest
from backend.models.response_models import QueryResponse, ContextDocument, ErrorResponse


router = APIRouter(prefix="/query", tags=["query"])

# Initialize services
file_manager = FileManager()
vectorstore_service = VectorStoreService()
memory_service = MemoryService()
csv_agent_service = CSVAgentService()


@router.post("/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document using RAG (or CSV Agent for CSV files) with persistent memory.
    
    - For CSV files: Uses CSV Agent with pandas for accurate calculations
    - For other files: Uses RAG with vector store retrieval
    - Maintains conversation memory for both approaches
    """
    try:
        question = request.question
        selected_file = request.selected_file
        session_id = request.session_id
        
        # Get collection name and file path
        collection_name = file_manager.get_collection_name(selected_file)
        
        # Check if this is a CSV file
        is_csv = file_manager.get_document_type(selected_file) == 'csv'
        
        if is_csv:
            # === CSV AGENT APPROACH ===
            
            # Get file path
            if selected_file.startswith('[URL]'):
                raise HTTPException(
                    status_code=400,
                    detail="CSV Agent does not support URLs. Please upload CSV files directly."
                )
            
            csv_path = file_manager.get_file_path(selected_file)
            
            # Check if file exists
            if not file_manager.file_exists(selected_file):
                raise HTTPException(
                    status_code=404,
                    detail=f"CSV file not found: {selected_file}"
                )
            
            # Load conversation memory
            memory, chat_history = memory_service.load_memory(collection_name, session_id)
            
            # Query CSV using agent
            result = csv_agent_service.query(csv_path, question)
            
            answer = result.get("answer", "No answer generated")
            
            # Save to memory
            memory_service.save_message(collection_name, session_id, "user", question)
            memory_service.save_message(collection_name, session_id, "assistant", answer)
            
            # Get updated memory state
            memory_state = memory_service.get_memory_state(collection_name, session_id)
            
            # Return response (no context documents for CSV agent)
            return QueryResponse(
                answer=answer,
                context=[
                    ContextDocument(
                        content=f"CSV Agent Analysis: Analyzed {selected_file} using pandas dataframe operations.",
                        metadata={"source": selected_file, "type": "csv_agent"}
                    )
                ],
                memory=memory_state
            )
            
        else:
            # === RAG APPROACH (for PDF, DOCX, TXT, URLs) ===
            
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