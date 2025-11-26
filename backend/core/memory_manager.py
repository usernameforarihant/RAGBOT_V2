"""
Memory Management for Backend
Wraps the existing ConversationMemory from utils
"""

import os
import sys
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.memory import ConversationMemory


class MemoryService:
    """Service for managing conversation memory in FastAPI backend."""
    
    def __init__(self, conversation_dir: str = "conversations"):
        """
        Initialize memory service.
        
        Args:
            conversation_dir: Directory for storing conversation history
        """
        self.conversation_dir = conversation_dir
        Path(conversation_dir).mkdir(parents=True, exist_ok=True)
    
    def get_memory_file_path(self, collection_name: str, session_id: str) -> str:
        """
        Get memory file path for a collection and session.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            
        Returns:
            Path to memory file
        """
        # Create session-specific memory file
        # Clean session_id to make it file-system safe
        safe_session_id = session_id.replace('/', '_').replace('\\', '_')
        memory_file = f"{collection_name}_{safe_session_id}_memory.json"
        return os.path.join(self.conversation_dir, memory_file)
    
    def load_memory(self, collection_name: str, session_id: str) -> Tuple[ConversationMemory, List[Dict]]:
        """
        Load conversation memory for a session.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            
        Returns:
            Tuple of (ConversationMemory, message_history)
        """
        memory_file = self.get_memory_file_path(collection_name, session_id)
        memory = ConversationMemory(memory_file)
        history = memory.load_history()
        return memory, history
    
    def save_message(
        self,
        collection_name: str,
        session_id: str,
        role: str,
        content: str,
        sources: List[str] = None
    ) -> None:
        """
        Save a message to memory.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            sources: Optional source documents
        """
        memory_file = self.get_memory_file_path(collection_name, session_id)
        memory = ConversationMemory(memory_file)
        memory.add_message(role, content, sources)
    
    def save_history(
        self,
        collection_name: str,
        session_id: str,
        messages: List[Dict]
    ) -> None:
        """
        Save entire message history.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            messages: List of message dictionaries
        """
        memory_file = self.get_memory_file_path(collection_name, session_id)
        memory = ConversationMemory(memory_file)
        memory.save_history(messages)
    
    def get_memory_state(self, collection_name: str, session_id: str) -> Dict:
        """
        Get current memory state.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            
        Returns:
            Dictionary with memory state information
        """
        memory, history = self.load_memory(collection_name, session_id)
        
        return {
            "messages": len(history),
            "last_updated": datetime.now().isoformat(),
            "session_id": session_id,
            "collection": collection_name
        }
    
    def clear_memory(self, collection_name: str, session_id: str) -> None:
        """
        Clear conversation memory.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
        """
        memory_file = self.get_memory_file_path(collection_name, session_id)
        memory = ConversationMemory(memory_file)
        memory.clear_history()
    
    def get_recent_messages(
        self,
        collection_name: str,
        session_id: str,
        n: int = 10
    ) -> List[Dict]:
        """
        Get n most recent messages.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            n: Number of recent messages to retrieve
            
        Returns:
            List of recent message dictionaries
        """
        memory, history = self.load_memory(collection_name, session_id)
        return history[-n:] if history else []
    
    def memory_exists(self, collection_name: str, session_id: str) -> bool:
        """
        Check if memory file exists for session.
        
        Args:
            collection_name: Collection name
            session_id: Session identifier
            
        Returns:
            True if memory file exists
        """
        memory_file = self.get_memory_file_path(collection_name, session_id)
        return os.path.exists(memory_file)