"""
Utility functions for PDF processing and text chunking.
Compatibility-safe for both legacy and modern LangChain package layouts.
"""

import os
from typing import List
from pypdf import PdfReader

# Text splitter import (modern -> fallback legacy)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None  # raised later with clear message

# Document import (modern -> fallback legacy)
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        Document = None  # raised later with clear message


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a single string
    """
    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = None

            if page_text:
                text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")

        return "\n".join(text_parts).strip()
    except Exception as e:
        raise Exception(f"Error extracting text from {pdf_path}: {str(e)}")


def extract_text_from_multiple_pdfs(pdf_files: List[str]) -> List[tuple]:
    """
    Extract text from multiple PDF files.

    Args:
        pdf_files: List of PDF file paths

    Returns:
        List of tuples containing (filename, extracted_text)
    """
    documents = []

    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        try:
            text = extract_text_from_pdf(pdf_file)
            documents.append((filename, text))
            print(f"[OK] Successfully processed: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {str(e)}")

    return documents


def chunk_text(text: str, source_name: str, chunk_size: int = 1000,
               chunk_overlap: int = 200) -> List[Document]:
    """
    Split text into chunks for vector storage.

    Args:
        text: Text to split
        source_name: Name of the source document
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of LangChain Document objects with metadata
    """
    if RecursiveCharacterTextSplitter is None:
        raise ImportError(
            "RecursiveCharacterTextSplitter not found. Install `langchain-text-splitters` "
            "or use an older langchain version that exposes `langchain.text_splitter`."
        )

    if Document is None:
        raise ImportError(
            "LangChain Document type not found. Install a compatible langchain/langchain-core package."
        )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text) if text else []

    # Create Document objects with metadata
    documents: List[Document] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        # Create Document with common kwargs used by both modern and legacy Document classes
        try:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "chunk_id": i,
                    "total_chunks": total
                }
            )
        except TypeError:
            # Some very old or nonstandard wrappers might use different ctor names
            # Attempt alternate construction patterns
            try:
                doc = Document(chunk, {"source": source_name, "chunk_id": i, "total_chunks": total})
            except Exception as e:
                raise RuntimeError(f"Unable to construct Document object: {e}")
        documents.append(doc)

    return documents


def process_pdfs_to_documents(pdf_files: List[str], chunk_size: int = 1000,
                               chunk_overlap: int = 200) -> List[Document]:
    """
    Complete pipeline: Extract text from PDFs and convert to chunked documents.

    Args:
        pdf_files: List of PDF file paths
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of LangChain Document objects ready for embedding
    """
    all_documents: List[Document] = []

    # Extract text from all PDFs
    extracted_data = extract_text_from_multiple_pdfs(pdf_files)

    # Chunk each document
    for filename, text in extracted_data:
        if text and text.strip():
            chunks = chunk_text(text, filename, chunk_size, chunk_overlap)
            all_documents.extend(chunks)
            print(f"  Created {len(chunks)} chunks from {filename}")

    print(f"\nTotal documents created: {len(all_documents)}")
    return all_documents


def format_sources(source_documents: List[Document]) -> str:
    """
    Format source documents into a readable citation string.

    Args:
        source_documents: List of retrieved documents

    Returns:
        Formatted string with source information
    """
    if not source_documents:
        return ""

    sources = []
    seen_sources = set()

    for doc in source_documents:
        # Depending on Document shape, metadata may be dict-like or accessible differently
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source", "Unknown")
        if source not in seen_sources:
            sources.append(source)
            seen_sources.add(source)

    if sources:
        return "\n\n**Sources:** " + ", ".join(sources)
    return ""
