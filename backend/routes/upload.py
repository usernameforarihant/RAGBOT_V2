"""
Upload Route
Handles file uploads and URL processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from backend.core.file_manager import FileManager
from backend.core.vectorstore import VectorStoreService
from backend.models.response_models import UploadResponse, ErrorResponse


router = APIRouter(prefix="/upload", tags=["upload"])

# Initialize services
file_manager = FileManager()
vectorstore_service = VectorStoreService()


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a file (PDF, DOCX, TXT).
    
    - Saves the file to disk
    - Checks if embeddings exist
    - Creates embeddings if needed
    - Returns list of available files
    """
    try:
        # Read file content
        file_content = await file.read()
        file_name = file.filename
        
        # Save file
        file_path = file_manager.save_file(file_content, file_name)
        
        # Get collection name
        collection_name = file_manager.get_collection_name(file_name)
        
        # Check if embeddings exist
        embeddings_exist = vectorstore_service.check_embeddings_exist(collection_name)
        
        if embeddings_exist:
            # Load existing embeddings
            message = f"Found existing embeddings for {file_name}. Loading..."
            embeddings_created = False
        else:
            # Create new embeddings
            message = f"Creating new embeddings for {file_name}..."
            doc_type = file_manager.get_document_type(file_name)
            _, embeddings_created = vectorstore_service.create_or_load_vectorstore(
                file_path,
                collection_name,
                doc_type
            )
            message = f"Successfully created embeddings for {file_name}"
        
        # Get all available files
        files_available = file_manager.get_all_documents()
        
        return UploadResponse(
            status="ok",
            file_name=file_name,
            files_available=files_available,
            message=message,
            embeddings_created=embeddings_created
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                status="error",
                message="Failed to process file",
                detail=str(e)
            ).dict()
        )


@router.post("/url")
async def upload_url(url: str):
    """
    Process a URL.
    
    - Fetches content from URL
    - Checks if embeddings exist
    - Creates embeddings if needed
    - Saves URL to list
    """
    try:
        if not url.startswith('http://') and not url.startswith('https://'):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL. Must start with http:// or https://"
            )
        
        # Generate document name for URL
        doc_name = f"[URL] {url}"
        collection_name = file_manager.get_collection_name(doc_name)
        
        # Check if embeddings exist
        embeddings_exist = vectorstore_service.check_embeddings_exist(collection_name)
        
        if embeddings_exist:
            message = f"Found existing embeddings for this URL. Loading..."
            embeddings_created = False
        else:
            # Create new embeddings
            message = f"Fetching and creating embeddings for URL..."
            _, embeddings_created = vectorstore_service.create_or_load_vectorstore(
                url,
                collection_name,
                'url'
            )
            # Save URL
            file_manager.save_url(url)
            message = f"Successfully created embeddings from URL"
        
        # Get all available files
        files_available = file_manager.get_all_documents()
        
        return UploadResponse(
            status="ok",
            file_name=doc_name,
            files_available=files_available,
            message=message,
            embeddings_created=embeddings_created
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                status="error",
                message="Failed to process URL",
                detail=str(e)
            ).dict()
        )


@router.get("/files")
async def list_files():
    """
    Get list of all available documents.
    """
    try:
        files = file_manager.get_all_documents()
        return {
            "status": "ok",
            "files": files
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                status="error",
                message="Failed to list files",
                detail=str(e)
            ).dict()
        )