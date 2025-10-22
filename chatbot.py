"""
Simple LangChain-based PDF Chatbot that auto-loads PDFs from backend/pdfs folder.
Uses Google Gemini (FREE) for AI responses.
"""

import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports (0.1.20 - stable version)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Local utilities
from utils import process_pdfs_to_documents, format_sources

# Load environment variables
load_dotenv()


class PDFChatbot:
    """Simple PDF Chatbot that auto-loads all PDFs from backend/pdfs folder."""

    def __init__(self):
        """Initialize chatbot and auto-load PDFs from backend/pdfs folder."""
        print("Initializing PDF Chatbot...")

        # Configuration from .env
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

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

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        if not self.qa_chain:
            return {
                "answer": "No PDFs loaded. Please add PDF files to backend/pdfs folder and restart the application.",
                "sources": "",
                "source_documents": []
            }

        # Get response from QA chain
        response = self.qa_chain({"question": question})

        # Extract answer and sources
        answer = response.get("answer", "")
        source_documents = response.get("source_documents", [])
        sources = format_sources(source_documents)

        return {
            "answer": answer,
            "sources": sources,
            "source_documents": source_documents
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


def create_chatbot() -> PDFChatbot:
    """Create and return a PDFChatbot instance."""
    return PDFChatbot()


if __name__ == "__main__":
    print("Testing chatbot...")
    bot = create_chatbot()
    if bot.is_ready():
        print("Chatbot is ready!")
    else:
        print("Chatbot not ready - no PDFs found")
