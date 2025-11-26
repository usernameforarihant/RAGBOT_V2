"""
RAG Chatbot with Streamlit UI + FastAPI Backend
This version calls FastAPI endpoints instead of direct processing
"""

import streamlit as st
import requests
import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload/"
URL_ENDPOINT = f"{API_BASE_URL}/upload/url"
QUERY_ENDPOINT = f"{API_BASE_URL}/query/"
FILES_ENDPOINT = f"{API_BASE_URL}/upload/files"


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def get_uploaded_documents():
    """Get list of all uploaded documents from API."""
    try:
        response = requests.get(FILES_ENDPOINT)
        if response.status_code == 200:
            return response.json().get("files", [])
        return []
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []


def upload_file_to_api(uploaded_file):
    """Upload file to FastAPI backend."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(UPLOAD_ENDPOINT, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None


def upload_url_to_api(url):
    """Upload URL to FastAPI backend."""
    try:
        response = requests.post(URL_ENDPOINT, params={"url": url})
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"URL processing failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return None


def query_api(question, selected_file, session_id):
    """Query the FastAPI backend."""
    try:
        payload = {
            "question": question,
            "selected_file": selected_file,
            "session_id": session_id
        }
        response = requests.post(QUERY_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error querying: {e}")
        return None


def get_document_type(doc_name: str) -> str:
    """Determine document type from name."""
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Chatbot with Multi-Format Knowledge Base")
    st.markdown("Upload documents (PDF, DOCX, TXT) or add URLs to chat with your content using AI")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # Tabs for different input types
        tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Add URL"])
        
        with tab1:
            # Upload new file
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'docx', 'doc', 'txt'],
                help="Upload PDF, DOCX, or TXT files"
            )
            
            if uploaded_file is not None:
                if st.button("Process File", type="primary", key="process_file"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = upload_file_to_api(uploaded_file)
                        if result:
                            if result.get("embeddings_created"):
                                st.success(f"✓ Created embeddings for {result['file_name']}")
                            else:
                                st.info(f"✓ Loaded existing embeddings for {result['file_name']}")
                            st.rerun()
        
        with tab2:
            # Add URL
            st.subheader("Add URL")
            url_input = st.text_input(
                "Enter URL",
                placeholder="https://example.com/article",
                help="Enter a web page URL to process"
            )
            
            if url_input:
                if st.button("Process URL", type="primary", key="process_url"):
                    if url_input.startswith('http://') or url_input.startswith('https://'):
                        with st.spinner("Processing URL..."):
                            result = upload_url_to_api(url_input)
                            if result:
                                if result.get("embeddings_created"):
                                    st.success(f"✓ Created embeddings from URL")
                                else:
                                    st.info(f"✓ Loaded existing embeddings from URL")
                                st.rerun()
                    else:
                        st.error("Please enter a valid URL starting with http:// or https://")
        
        st.divider()
        
        # Select existing document
        st.subheader("Select Document for Chat")
        existing_docs = get_uploaded_documents()
        
        if existing_docs:
            selected_doc = st.selectbox(
                "Available Documents",
                options=[""] + existing_docs,
                format_func=lambda x: "Select a document..." if x == "" else x
            )
            
            if selected_doc and selected_doc != st.session_state.current_doc:
                if st.button("Load Document", type="primary"):
                    st.session_state.current_doc = selected_doc
                    st.session_state.messages = []  # Clear messages when switching docs
                    st.success(f"✓ Loaded {selected_doc}")
                    st.rerun()
            
            # Show current document
            if st.session_state.current_doc:
                doc_type = get_document_type(st.session_state.current_doc)
                type_emoji = {
                    'pdf': '📕',
                    'docx': '📘',
                    'txt': '📄',
                    'url': '🔗'
                }
                st.info(f"{type_emoji.get(doc_type, '📄')} Current: {st.session_state.current_doc}")
        else:
            st.warning("No documents uploaded yet. Upload a file or add a URL to get started!")
        
        st.divider()
        
        # Clear chat history button
        if st.session_state.current_doc:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if not st.session_state.current_doc:
        st.info("👈 Please select or upload a document from the sidebar to start chatting")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source["content"][:300] + "..." 
                               if len(source["content"]) > 300 
                               else source["content"])
                        if source.get("metadata"):
                            st.caption(f"Metadata: {source['metadata']}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query API
                response = query_api(
                    prompt,
                    st.session_state.current_doc,
                    st.session_state.session_id
                )
                
                if response:
                    answer = response["answer"]
                    context = response["context"]
                    
                    # Display response
                    st.markdown(answer)
                    
                    # Display sources
                    if context:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(context, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(doc["content"][:300] + "..." 
                                       if len(doc["content"]) > 300 
                                       else doc["content"])
                                if doc.get("metadata"):
                                    st.caption(f"Metadata: {doc['metadata']}")
                                st.divider()
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": context
                    })
                    
                    st.rerun()


if __name__ == "__main__":
    main()