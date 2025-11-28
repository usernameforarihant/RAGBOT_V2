
# """
# RAG Chatbot with Streamlit UI
# LangChain v0.3.x compatible implementation
# Supports: PDF, DOCX, TXT, CSV, and URLs
# """

# import streamlit as st
# import os
# from pathlib import Path
# from utils.document_processor import DocumentProcessor
# from utils.vector_store import VectorStoreManager
# from utils.rag_chain import RAGChain
# from utils.memory import ConversationMemory
# from dotenv import load_dotenv
# load_dotenv()

# # Configuration
# UPLOAD_DIR = "data/uploaded_docs"
# CHROMA_DIR = "data/chroma_db"
# CONVERSATION_DIR = "conversations"
# URLS_FILE = "data/saved_urls.txt"

# # Create necessary directories
# for dir_path in [UPLOAD_DIR, CHROMA_DIR, CONVERSATION_DIR, "data"]:
#     Path(dir_path).mkdir(parents=True, exist_ok=True)


# def initialize_session_state():
#     """Initialize Streamlit session state variables."""
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "current_doc" not in st.session_state:
#         st.session_state.current_doc = None
#     if "vector_store_manager" not in st.session_state:
#         st.session_state.vector_store_manager = None
#     if "rag_chain" not in st.session_state:
#         st.session_state.rag_chain = None
#     if "memory" not in st.session_state:
#         st.session_state.memory = None


# def get_uploaded_documents():
#     """Get list of all uploaded documents and URLs."""
#     docs = []
    
#     # Get uploaded files
#     if os.path.exists(UPLOAD_DIR):
#         files = [
#             f
#             for f in os.listdir(UPLOAD_DIR)
#             if f.endswith(('.pdf', '.docx', '.doc', '.txt', '.csv'))
#         ]
#         docs.extend(files)
    
#     # Get saved URLs
#     if os.path.exists(URLS_FILE):
#         with open(URLS_FILE, 'r') as f:
#             urls = [line.strip() for line in f if line.strip()]
#             docs.extend([f"[URL] {url}" for url in urls])
    
#     return docs


# def save_url(url: str):
#     """Save URL to file."""
#     with open(URLS_FILE, 'a') as f:
#         f.write(url + '\n')


# def get_document_type(doc_name: str) -> str:
#     """Determine document type from name."""
#     if doc_name.startswith('[URL]'):
#         return 'url'
#     ext = os.path.splitext(doc_name)[1].lower()
#     type_map = {
#         '.pdf': 'pdf',
#         '.docx': 'docx',
#         '.doc': 'docx',
#         '.txt': 'txt',
#         '.csv': 'csv'
#     }
#     return type_map.get(ext, 'unknown')


# def get_collection_name(doc_name: str) -> str:
#     """Generate collection name for document."""
#     if doc_name.startswith('[URL]'):
#         url = doc_name.replace('[URL] ', '')
#         # Create safe collection name from URL
#         import hashlib
#         return 'url_' + hashlib.md5(url.encode()).hexdigest()[:16]
#     else:
#         return (
#             doc_name.replace('.pdf', '')
#             .replace('.docx', '')
#             .replace('.doc', '')
#             .replace('.txt', '')
#             .replace('.csv', '')
#             .replace(' ', '_')
#             .lower()
#         )


# def check_embeddings_exist(doc_name):
#     """Check if embeddings already exist for a document."""
#     collection_name = get_collection_name(doc_name)
#     vector_store_path = os.path.join(CHROMA_DIR, collection_name)
#     return os.path.exists(vector_store_path) and os.listdir(vector_store_path)


# def process_file(uploaded_file, file_name):
#     """Process uploaded file and create/load embeddings."""
#     with st.spinner(f"Processing {file_name}..."):
#         # Save file to disk
#         file_path = os.path.join(UPLOAD_DIR, file_name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Check if embeddings exist
#         collection_name = get_collection_name(file_name)
        
#         if check_embeddings_exist(file_name):
#             st.info(f"✓ Found existing embeddings for {file_name}. Loading...")
#             # Load existing vector store
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.load_existing(collection_name)
#         else:
#             st.info(f"Creating new embeddings for {file_name}...")
#             # Process document and create embeddings
#             doc_processor = DocumentProcessor()
#             documents = doc_processor.load_document(file_path)
#             text_chunks = doc_processor.split_documents(documents)
            
#             # Create vector store
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.create_vector_store(text_chunks, collection_name)
#             st.success(f"✓ Created {len(text_chunks)} embeddings for {file_name}")
        
#         return vector_store_manager, collection_name


# def process_url(url: str):
#     """Process URL and create/load embeddings."""
#     with st.spinner(f"Processing URL..."):
#         doc_name = f"[URL] {url}"
#         collection_name = get_collection_name(doc_name)
        
#         if check_embeddings_exist(doc_name):
#             st.info(f"✓ Found existing embeddings for this URL. Loading...")
#             # Load existing vector store
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.load_existing(collection_name)
#         else:
#             st.info(f"Fetching and creating embeddings for URL...")
#             # Process URL and create embeddings
#             doc_processor = DocumentProcessor()
#             documents = doc_processor.load_url(url)
#             text_chunks = doc_processor.split_documents(documents)
            
#             # Create vector store
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.create_vector_store(text_chunks, collection_name)
            
#             # Save URL
#             save_url(url)
#             st.success(f"✓ Created {len(text_chunks)} embeddings from URL")
        
#         return vector_store_manager, collection_name


# def load_document_for_chat(doc_name):
#     """Load selected document for chatting."""
#     if not doc_name:
#         return
    
#     # Reset chat history when switching documents
#     if st.session_state.current_doc != doc_name:
#         st.session_state.messages = []
#         st.session_state.current_doc = doc_name
    
#     # Get collection name and document path
#     collection_name = get_collection_name(doc_name)
    
#     # Determine if URL or file
#     if doc_name.startswith('[URL]'):
#         url = doc_name.replace('[URL] ', '')
        
#         if check_embeddings_exist(doc_name):
#             # Load existing embeddings
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.load_existing(collection_name)
#         else:
#             # Process URL if embeddings don't exist
#             doc_processor = DocumentProcessor()
#             documents = doc_processor.load_url(url)
#             text_chunks = doc_processor.split_documents(documents)
            
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.create_vector_store(text_chunks, collection_name)
#     else:
#         # Handle file
#         file_path = os.path.join(UPLOAD_DIR, doc_name)
        
#         if check_embeddings_exist(doc_name):
#             # Load existing embeddings
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.load_existing(collection_name)
#         else:
#             # Process file if embeddings don't exist
#             doc_processor = DocumentProcessor()
#             documents = doc_processor.load_document(file_path)
#             text_chunks = doc_processor.split_documents(documents)
            
#             vector_store_manager = VectorStoreManager(CHROMA_DIR)
#             vector_store_manager.create_vector_store(text_chunks, collection_name)
    
#     # Initialize RAG chain and memory
#     st.session_state.vector_store_manager = vector_store_manager
#     st.session_state.rag_chain = RAGChain(vector_store_manager.vector_store)
    
#     # Load conversation memory for this document
#     memory_file = os.path.join(CONVERSATION_DIR, f"{collection_name}_memory.json")
#     st.session_state.memory = ConversationMemory(memory_file)
#     st.session_state.messages = st.session_state.memory.load_history()


# def main():
#     """Main Streamlit application."""
#     st.set_page_config(
#         page_title="RAG Chatbot",
#         page_icon="📚",
#         layout="wide"
#     )
    
#     st.title("📚 RAG Chatbot with Multi-Format Knowledge Base")
#     st.markdown("Upload documents (PDF, DOCX, TXT, CSV) or add URLs to chat with your content using AI")
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Check for OpenAI API key
#     if "OPENAI_API_KEY" not in os.environ:
#         st.error("⚠️ OPENAI_API_KEY environment variable not set!")
#         st.info("Please set your OpenAI API key: `export OPENAI_API_KEY='your-key-here'`")
#         st.stop()
    
#     # Sidebar for document management
#     with st.sidebar:
#         st.header("📄 Document Management")
        
#         # Tabs for different input types
#         tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Add URL"])
        
#         with tab1:
#             # Upload new file
#             st.subheader("Upload Document")
#             uploaded_file = st.file_uploader(
#                 "Choose a file",
#                 type=['pdf', 'docx', 'doc', 'txt', 'csv'],
#                 help="Upload PDF, DOCX, TXT, or CSV files"
#             )
            
#             if uploaded_file is not None:
#                 if st.button("Process File", type="primary", key="process_file"):
#                     vector_store_manager, collection_name = process_file(
#                         uploaded_file,
#                         uploaded_file.name
#                     )
#                     st.success(f"✓ {uploaded_file.name} processed successfully!")
#                     st.rerun()
        
#         with tab2:
#             # Add URL
#             st.subheader("Add URL")
#             url_input = st.text_input(
#                 "Enter URL",
#                 placeholder="https://example.com/article",
#                 help="Enter a web page URL to process"
#             )
            
#             if url_input:
#                 if st.button("Process URL", type="primary", key="process_url"):
#                     if url_input.startswith('http://') or url_input.startswith('https://'):
#                         vector_store_manager, collection_name = process_url(url_input)
#                         st.success(f"✓ URL processed successfully!")
#                         st.rerun()
#                     else:
#                         st.error("Please enter a valid URL starting with http:// or https://")
        
#         st.divider()
        
#         # Select existing document
#         st.subheader("Select Document for Chat")
#         existing_docs = get_uploaded_documents()
        
#         if existing_docs:
#             selected_doc = st.selectbox(
#                 "Available Documents",
#                 options=[""] + existing_docs,
#                 format_func=lambda x: "Select a document..." if x == "" else x
#             )
            
#             if selected_doc and selected_doc != st.session_state.current_doc:
#                 if st.button("Load Document", type="primary"):
#                     load_document_for_chat(selected_doc)
#                     st.success(f"✓ Loaded {selected_doc}")
#                     st.rerun()
            
#             # Show current document
#             if st.session_state.current_doc:
#                 doc_type = get_document_type(st.session_state.current_doc)
#                 type_emoji = {
#                     'pdf': '📕',
#                     'docx': '📘',
#                     'txt': '📄',
#                     'csv': '📊',
#                     'url': '🔗'
#                 }
#                 st.info(f"{type_emoji.get(doc_type, '📄')} Current: {st.session_state.current_doc}")
                
#                 # Show embedding status
#                 if check_embeddings_exist(st.session_state.current_doc):
#                     st.success("✓ Embeddings loaded")
#         else:
#             st.warning("No documents uploaded yet. Upload a file or add a URL to get started!")
        
#         st.divider()
        
#         # Clear chat history button
#         if st.session_state.current_doc:
#             if st.button("🗑️ Clear Chat History"):
#                 st.session_state.messages = []
#                 if st.session_state.memory:
#                     st.session_state.memory.clear_history()
#                 st.rerun()
    
#     # Main chat interface
#     if not st.session_state.current_doc:
#         st.info("👈 Please select or upload a document from the sidebar to start chatting")
#         st.stop()
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             # Show sources if available
#             if "sources" in message and message["sources"]:
#                 with st.expander("📚 View Sources"):
#                     for i, source in enumerate(message["sources"], 1):
#                         st.markdown(f"**Source {i}:**")
#                         st.text(source[:300] + "..." if len(source) > 300 else source)
#                         st.divider()
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your document..."):
#         # Add user message to chat
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Get response from RAG chain
#                     response = st.session_state.rag_chain.get_response(
#                         prompt,
#                         st.session_state.messages
#                     )
                    
#                     # Display response
#                     st.markdown(response["answer"])
                    
#                     # Display sources
#                     if response.get("source_documents"):
#                         with st.expander("📚 View Sources"):
#                             for i, doc in enumerate(response["source_documents"], 1):
#                                 st.markdown(f"**Source {i}:**")
#                                 st.text(doc.page_content[:300] + "..." 
#                                        if len(doc.page_content) > 300 
#                                        else doc.page_content)
#                                 st.divider()
                    
#                     # Add assistant message to chat
#                     sources = [doc.page_content for doc in response.get("source_documents", [])]
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "content": response["answer"],
#                         "sources": sources
#                     })
                    
#                     # Save conversation history
#                     if st.session_state.memory:
#                         st.session_state.memory.save_history(st.session_state.messages)
                    
#                 except Exception as e:
#                     st.error(f"Error generating response: {str(e)}")
#                     st.error("Please try again or check your OpenAI API key.")


# if __name__ == "__main__":
#     main()