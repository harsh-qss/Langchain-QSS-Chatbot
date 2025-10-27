"""
Simple LangChain-based PDF Chatbot that auto-loads PDFs from backend/pdfs folder.
Uses Google Gemini (FREE) for AI responses.
Supports live PDF folder monitoring for real-time updates.
"""

import os
import glob
import threading
from typing import List, Dict, Any, Callable, Optional
from dotenv import load_dotenv

# LangChain imports (0.1.20 - stable version)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Local utilities
from utils import (
    process_pdfs_to_documents,
    format_sources,
    get_faiss_document_sources,
    remove_documents_from_faiss_by_source,
    add_documents_to_faiss
)
from pdf_watcher import PDFWatcher

# Load environment variables
load_dotenv()


class PDFChatbot:
    """Simple PDF Chatbot that auto-loads all PDFs from backend/pdfs folder with live updates."""

    def __init__(self, enable_watcher: bool = True, on_update_callback: Optional[Callable] = None):
        """
        Initialize chatbot and auto-load PDFs from backend/pdfs folder.

        Args:
            enable_watcher: Whether to enable live PDF folder monitoring (default: True)
            on_update_callback: Optional callback function for PDF updates (signature: callback(event_type, filename))
        """
        print("Initializing PDF Chatbot...")

        # Configuration from .env
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.pdf_folder = os.path.join(os.getcwd(), "backend", "pdfs")

        # Thread-safe FAISS updates
        self.faiss_lock = threading.Lock()

        # Update callback for Streamlit notifications
        self.on_update_callback = on_update_callback

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Initialize embeddings
        print("Loading embedding model...")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize Gemini LLM (FREE)
        print("Initializing Google Gemini...")
        api_key = os.getenv("GOOGLE_API_KEY")
        # Try different model names that work with langchain-google-genai
        model = os.getenv("GEMINI_MODEL", "models/gemini-pro")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )

        # Auto-load PDFs from backend/pdfs folder
        self.vector_store = None
        self.qa_chain = None
        self._auto_load_pdfs()

        # Initialize PDF folder watcher (live monitoring)
        self.watcher = None
        if enable_watcher:
            self._start_pdf_watcher()

    def _auto_load_pdfs(self):
        """Automatically load all PDFs from backend/pdfs folder."""
        # Get all PDF files from backend/pdfs folder
        pdf_folder = os.path.join(os.getcwd(), "backend", "pdfs")
        pdf_pattern = os.path.join(pdf_folder, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)

        if not pdf_files:
            print(f"Warning: No PDF files found in {pdf_folder}")
            print(f"Please add PDF files to {pdf_folder} folder")
            return

        print(f"Found {len(pdf_files)} PDF file(s) in backend/pdfs:")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")

        # Process PDFs into documents
        print("Processing PDFs...")
        documents = process_pdfs_to_documents(
            pdf_files,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        if not documents:
            print("Warning: No text extracted from PDFs")
            return

        print(f"Created {len(documents)} document chunks")

        # Create vector store
        print("Building vector store...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

        # Create QA chain
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )

        print("Chatbot ready!")

    def _start_pdf_watcher(self):
        """Start the PDF folder watcher in a background thread."""
        try:
            self.watcher = PDFWatcher(
                watch_directory=self.pdf_folder,
                callback=self._on_pdf_event,
                debounce_delay=2.0
            )
            self.watcher.start()
        except Exception as e:
            print(f"Warning: Could not start PDF watcher: {str(e)}")
            self.watcher = None

    def _on_pdf_event(self, event_type: str, file_path: str):
        """
        Handle PDF file system events.

        Args:
            event_type: One of 'created', 'deleted', 'modified'
            file_path: Full path to the file
        """
        filename = os.path.basename(file_path)
        print(f"\n📄 PDF Event: {event_type.upper()} - {filename}")

        try:
            if event_type == "created":
                self._handle_pdf_created(file_path)
            elif event_type == "deleted":
                self._handle_pdf_deleted(file_path)
            elif event_type == "modified":
                self._handle_pdf_modified(file_path)
        except Exception as e:
            print(f"Error handling PDF event: {str(e)}")
            if self.on_update_callback:
                self.on_update_callback("error", f"Error: {str(e)}")

    def _handle_pdf_created(self, file_path: str):
        """Handle a new PDF being added."""
        filename = os.path.basename(file_path)

        # Skip if FAISS not initialized yet
        if not self.vector_store:
            print(f"Skipping {filename}: FAISS not initialized yet")
            return

        with self.faiss_lock:
            try:
                print(f"Processing new PDF: {filename}")
                # Process the new PDF
                documents = process_pdfs_to_documents(
                    [file_path],
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

                if documents:
                    # Add documents to existing FAISS index
                    self.vector_store = add_documents_to_faiss(
                        self.vector_store,
                        documents,
                        self.embeddings
                    )
                    print(f"✅ Added {len(documents)} chunks from {filename}")
                    self._update_retriever()

                    # Notify via callback
                    if self.on_update_callback:
                        self.on_update_callback("created", filename)
            except Exception as e:
                print(f"Error processing new PDF {filename}: {str(e)}")
                if self.on_update_callback:
                    self.on_update_callback("error", f"Failed to add {filename}")

    def _handle_pdf_deleted(self, file_path: str):
        """Handle a PDF being deleted."""
        filename = os.path.basename(file_path)

        # Skip if FAISS not initialized yet
        if not self.vector_store:
            print(f"Skipping deletion of {filename}: FAISS not initialized yet")
            return

        with self.faiss_lock:
            try:
                print(f"Removing PDF from index: {filename}")
                # Remove all chunks from this PDF
                removed_count = remove_documents_from_faiss_by_source(
                    self.vector_store,
                    filename
                )

                if removed_count > 0:
                    print(f"✅ Removed {removed_count} chunks of {filename}")
                    self._update_retriever()

                    # Notify via callback
                    if self.on_update_callback:
                        self.on_update_callback("deleted", filename)
                else:
                    print(f"⚠️ No chunks found for {filename} in index")
            except Exception as e:
                print(f"Error removing PDF {filename}: {str(e)}")
                if self.on_update_callback:
                    self.on_update_callback("error", f"Failed to remove {filename}")

    def _handle_pdf_modified(self, file_path: str):
        """Handle a PDF being modified (remove old, add new)."""
        filename = os.path.basename(file_path)

        # Skip if FAISS not initialized yet
        if not self.vector_store:
            print(f"Skipping {filename}: FAISS not initialized yet")
            return

        with self.faiss_lock:
            try:
                print(f"Updating modified PDF: {filename}")

                # Remove old version
                removed_count = remove_documents_from_faiss_by_source(
                    self.vector_store,
                    filename
                )
                print(f"Removed {removed_count} old chunks")

                # Process updated PDF
                documents = process_pdfs_to_documents(
                    [file_path],
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

                if documents:
                    # Add new version
                    self.vector_store = add_documents_to_faiss(
                        self.vector_store,
                        documents,
                        self.embeddings
                    )
                    print(f"✅ Updated {filename} with {len(documents)} new chunks")
                    self._update_retriever()

                    # Notify via callback
                    if self.on_update_callback:
                        self.on_update_callback("modified", filename)
            except Exception as e:
                print(f"Error updating PDF {filename}: {str(e)}")
                if self.on_update_callback:
                    self.on_update_callback("error", f"Failed to update {filename}")

    def _update_retriever(self):
        """
        Recreate the retriever and QA chain after FAISS index changes.
        This ensures the chain uses the latest vector store.
        """
        try:
            if not self.vector_store:
                return

            # Recreate retriever with updated FAISS index
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            # Recreate QA chain with updated retriever
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            print("✅ Retriever updated successfully")
        except Exception as e:
            print(f"Error updating retriever: {str(e)}")

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        if not self.qa_chain:
            return {
                "answer": "No PDFs loaded. Please add PDF files to backend/pdfs folder and restart the application.",
                "sources": "",
                "source_documents": []
            }

        # Handle greetings and general conversation
        question_lower = question.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']

        if question_lower in greetings or any(greeting in question_lower for greeting in greetings):
            return {
                "answer": "Hello! I'm your AI assistant.\n\n What would you like to know?",
                "sources": "",
                "source_documents": []
            }

        # Check if question is likely about the documents or general knowledge
        # Keywords that suggest PDF/document-specific questions
        pdf_keywords = [
            'company', 'policy', 'policies', 'leave', 'maternity', 'dress code',
            'headquarters', 'office', 'employee', 'document', 'pdf', 'according to',
            'qss', 'technooft', 'organization', 'work', 'salary', 'benefit'
        ]

        is_pdf_question = any(keyword in question_lower for keyword in pdf_keywords)

        # For PDF-specific questions, use the QA chain (searches documents)
        if is_pdf_question:
            response = self.qa_chain({"question": question})
            answer = response.get("answer", "")
            source_documents = response.get("source_documents", [])
            sources = format_sources(source_documents)

            return {
                "answer": answer,
                "sources": sources,
                "source_documents": source_documents
            }

        # For general questions, use LLM directly without document search
        else:
            try:
                # Use the LLM directly for general knowledge questions
                from langchain.schema import HumanMessage

                messages = [HumanMessage(content=question)]
                response = self.llm(messages)
                answer = response.content if hasattr(response, 'content') else str(response)

                return {
                    "answer": answer + "\n\n_Note: This is a general answer. For company-specific information, please ask about company policies._",
                    "sources": "",
                    "source_documents": []
                }
            except Exception as e:
                # Fallback to QA chain if direct LLM call fails
                response = self.qa_chain({"question": question})
                answer = response.get("answer", "")

                return {
                    "answer": answer,
                    "sources": "",
                    "source_documents": []
                }

    def is_ready(self) -> bool:
        """Check if chatbot is ready (has PDFs loaded)."""
        return self.qa_chain is not None

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

    def get_chat_history(self) -> List[tuple]:
        """Get conversation history as list of (question, answer) tuples."""
        if hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            history = []
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append((messages[i].content, messages[i + 1].content))
            return history
        return []


    def __del__(self):
        """Clean up watcher on chatbot deletion."""
        if self.watcher:
            try:
                self.watcher.stop()
            except Exception:
                pass


def create_chatbot(enable_watcher: bool = True, on_update_callback: Optional[Callable] = None) -> PDFChatbot:
    """
    Create and return a PDFChatbot instance.

    Args:
        enable_watcher: Whether to enable live PDF folder monitoring
        on_update_callback: Optional callback for PDF updates
    """
    return PDFChatbot(enable_watcher=enable_watcher, on_update_callback=on_update_callback)


if __name__ == "__main__":
    print("Testing chatbot...")
    bot = create_chatbot()
    if bot.is_ready():
        print("Chatbot is ready!")
    else:
        print("Chatbot not ready - no PDFs found")
