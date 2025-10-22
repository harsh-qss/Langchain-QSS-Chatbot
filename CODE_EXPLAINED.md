# 💻 Code Explanation - Deep Dive into chatbot.py

This document explains the **actual code** line-by-line, focusing on `chatbot.py` (the brain of the application).

---

## 📁 File Overview

### **Main Files:**
1. **chatbot.py** - Core AI logic (THIS DOCUMENT)
2. **app.py** - Web interface (Streamlit)
3. **utils.py** - PDF processing helpers

---

## 🧠 chatbot.py - Complete Code Explanation

### **File Structure:**
```python
Lines 1-23:   Imports and Setup
Lines 26-70:  PDFChatbot.__init__() - Initialization
Lines 72-121: PDFChatbot._auto_load_pdfs() - Load PDFs
Lines 123-190: PDFChatbot.ask() - Answer questions
Lines 192-194: PDFChatbot.is_ready() - Check status
Lines 196-198: PDFChatbot.clear_memory() - Reset chat
Lines 165-167: create_chatbot() - Factory function
```

---

## 📦 Section 1: Imports (Lines 1-23)

### **Code:**
```python
"""
Simple LangChain-based PDF Chatbot that auto-loads PDFs from backend/pdfs folder.
Uses Google Gemini (FREE) for AI responses.
"""

import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
```

### **Explanation:**

**Line 1-4: Docstring**
- Documents what this file does
- Helps developers understand purpose

**Line 6: `import os`**
- Operating system functions
- Used for: file paths, environment variables
- Example: `os.getenv("GOOGLE_API_KEY")`

**Line 7: `import glob`**
- Find files matching patterns
- Used for: finding all PDFs in folder
- Example: `glob.glob("backend/pdfs/*.pdf")`

**Line 8: `from typing import List, Dict, Any`**
- Type hints for better code documentation
- `List[str]` = list of strings
- `Dict[str, Any]` = dictionary with string keys, any values
- Helps IDEs show you what types to expect

**Line 9: `from dotenv import load_dotenv`**
- Loads environment variables from `.env` file
- Keeps secrets (API keys) separate from code
- Called once at startup

---

### **Code:**
```python
# LangChain imports (0.1.20 - stable version)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
```

### **Explanation:**

**Line 12: `from langchain.schema import Document`**
- Document = container for text + metadata
- Example:
  ```python
  doc = Document(
      page_content="Text from PDF",
      metadata={"source": "file.pdf", "page": 1}
  )
  ```

**Line 13: `from langchain_community.embeddings import HuggingFaceEmbeddings`**
- Converts text → numbers (embeddings)
- Uses pre-trained model: `all-MiniLM-L6-v2`
- Runs locally on your computer (no API needed)

**Line 14: `from langchain_community.vectorstores import FAISS`**
- FAISS = Facebook AI Similarity Search
- Vector database that stores embeddings
- Super fast searching (millions of vectors in milliseconds)

**Line 15: `from langchain.memory import ConversationBufferMemory`**
- Stores conversation history
- Remembers previous questions/answers
- Makes chatbot conversational

**Line 16: `from langchain.chains import ConversationalRetrievalChain`**
- Chain = sequence of steps executed automatically
- This chain: search PDFs → get context → ask AI → return answer
- "Conversational" = includes memory

**Line 17: `from langchain_google_genai import ChatGoogleGenerativeAI`**
- Google Gemini AI wrapper
- Sends prompts to Google's API
- Gets AI responses

---

### **Code:**
```python
# Local utilities
from utils import process_pdfs_to_documents, format_sources

# Load environment variables
load_dotenv()
```

### **Explanation:**

**Line 20: `from utils import ...`**
- `process_pdfs_to_documents`: Read PDFs → split → create Document objects
- `format_sources`: Turn source documents into readable text

**Line 23: `load_dotenv()`**
- Reads `.env` file
- Loads variables like `GOOGLE_API_KEY` into `os.environ`
- Called once when module loads

---

## 🏗️ Section 2: PDFChatbot Class - Initialization (Lines 26-70)

### **Code:**
```python
class PDFChatbot:
    """Simple PDF Chatbot that auto-loads all PDFs from backend/pdfs folder."""

    def __init__(self):
        """Initialize chatbot and auto-load PDFs from backend/pdfs folder."""
        print("Initializing PDF Chatbot...")
```

### **Explanation:**

**Line 26: `class PDFChatbot:`**
- Defines a new class (blueprint for creating chatbot objects)
- Contains all chatbot functionality

**Line 29: `def __init__(self):`**
- Constructor = runs when you create a chatbot
- Example: `bot = PDFChatbot()` → `__init__` runs automatically
- Sets up everything needed for chatbot to work

**Line 31: `print("Initializing PDF Chatbot...")`**
- Shows user what's happening
- Helpful for debugging

---

### **Code:**
```python
        # Configuration from .env
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
```

### **Explanation:**

**Line 34: `self.chunk_size = ...`**
- `self.` = stores this variable in the object (accessible everywhere in class)
- `os.getenv("CHUNK_SIZE", "1000")` = get value from .env, default to "1000"
- `int(...)` = convert string to integer
- Result: `self.chunk_size = 1000`

**Why chunk_size = 1000?**
- PDFs are too large for AI to read at once
- Split into 1000-character chunks
- Each chunk is small enough to process

**Line 35: `self.chunk_overlap = 200`**
- Overlap = last 200 chars of chunk1 = first 200 chars of chunk2
- Prevents losing context at boundaries
- Example:
  ```
  Chunk 1: "...maternity leave is 6 months paid."
  Chunk 2: "6 months paid maternity leave can be extended..."
           ↑ 200 char overlap ensures context
  ```

---

### **Code:**
```python
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
```

### **Explanation:**

**Line 38-42: Create Memory Object**

**`ConversationBufferMemory`**
- Stores all conversation messages in a buffer (list)
- Simple but uses memory (RAM)
- Alternative: `ConversationSummaryMemory` (summarizes to save space)

**`memory_key="chat_history"`**
- Internal variable name for storing messages
- The chain looks for `chat_history` to get previous messages

**`return_messages=True`**
- Return messages as LangChain Message objects (not plain strings)
- Needed for conversational chains

**`output_key="answer"`**
- Which key in response to save to memory
- Our chain returns: `{"answer": "...", "sources": "..."}`
- Only save `answer` to memory (not sources)

**How it works:**
```python
# After first question:
memory = {
    "chat_history": [
        HumanMessage("What is maternity leave?"),
        AIMessage("Maternity leave is 6 months paid...")
    ]
}

# After second question:
memory = {
    "chat_history": [
        HumanMessage("What is maternity leave?"),
        AIMessage("Maternity leave is 6 months paid..."),
        HumanMessage("Can it be extended?"),  # ← "it" = maternity leave (from context)
        AIMessage("Yes, to 8 months...")
    ]
}
```

---

### **Code:**
```python
        # Initialize embeddings
        print("Loading embedding model...")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
```

### **Explanation:**

**Line 47: `model_name = ...`**
- Gets embedding model name from .env
- Default: `sentence-transformers/all-MiniLM-L6-v2`
- This is a free, open-source model from HuggingFace

**Line 48-52: Create Embeddings Object**

**`HuggingFaceEmbeddings(...)`**
- Downloads model first time (~90MB)
- Stored in: `~/.cache/huggingface/`
- Next runs: loads from cache (fast)

**`model_kwargs={'device': 'cpu'}`**
- Run on CPU (not GPU)
- GPU would be faster but requires special setup
- CPU is fine for our use case

**`encode_kwargs={'normalize_embeddings': True}`**
- Normalize = make all embeddings same length (unit vectors)
- Better for similarity comparison
- Math: divides each number by vector length

**What happens:**
```python
text = "Maternity leave is 6 months"
embedding = self.embeddings.embed_query(text)
# Returns: [0.023, 0.156, -0.089, ..., 0.234]  # 768 numbers
```

---

### **Code:**
```python
        # Initialize Gemini LLM (FREE)
        print("Initializing Google Gemini...")
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_MODEL", "models/gemini-pro")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
```

### **Explanation:**

**Line 55-59: Get Configuration**

**`api_key = os.getenv("GOOGLE_API_KEY")`**
- Gets your API key from .env
- Example: `"AIzaSyAr2XSg3Hbi3yFY5SWSqwF5tgLcNdpAzqI"`

**`if not api_key: raise ValueError(...)`**
- If no API key found, crash with clear error message
- Better than mysterious error later

**Line 60-65: Create LLM Object**

**`ChatGoogleGenerativeAI(...)`**
- Wrapper around Google's Gemini API
- Handles authentication, requests, responses

**`model=model`**
- Which Gemini model to use
- Example: `"models/gemini-2.5-flash"`
- Flash = faster, cheaper; Pro = more accurate

**`temperature=0.7`**
- Controls randomness (0.0 to 1.0)
- 0.0 = deterministic (same question = same answer)
- 1.0 = very creative/random
- 0.7 = balanced (recommended for Q&A)

**`convert_system_message_to_human=True`**
- Gemini doesn't support "system" messages
- Converts system messages to human messages automatically
- Needed for chains to work

---

### **Code:**
```python
        # Auto-load PDFs from backend/pdfs folder
        self.vector_store = None
        self.qa_chain = None
        self._auto_load_pdfs()
```

### **Explanation:**

**Line 68-69: Initialize Variables**
- `self.vector_store = None` → will hold FAISS database
- `self.qa_chain = None` → will hold the Q&A chain
- Set to `None` first (will be created if PDFs found)

**Line 70: `self._auto_load_pdfs()`**
- Calls method to load PDFs (explained next)
- `_` prefix = private method (internal use only)
- Runs automatically during initialization

---

## 📂 Section 3: Auto-Load PDFs (Lines 72-121)

### **Code:**
```python
    def _auto_load_pdfs(self):
        """Automatically load all PDFs from backend/pdfs folder."""
        # Get all PDF files from backend/pdfs folder
        pdf_folder = os.path.join(os.getcwd(), "backend", "pdfs")
        pdf_pattern = os.path.join(pdf_folder, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
```

### **Explanation:**

**Line 75: `pdf_folder = os.path.join(...)`**
- `os.getcwd()` = current working directory
- Example: `"C:\Users\QSS\Downloads\pdf-chatbot\pdf-chatbot"`
- `os.path.join()` = safely combine paths (works on Windows/Mac/Linux)
- Result: `"C:\Users\QSS\Downloads\pdf-chatbot\pdf-chatbot\backend\pdfs"`

**Line 76: `pdf_pattern = os.path.join(pdf_folder, "*.pdf")`**
- `"*.pdf"` = wildcard pattern (any file ending in .pdf)
- Result: `"C:\...\backend\pdfs\*.pdf"`

**Line 77: `pdf_files = glob.glob(pdf_pattern)`**
- Finds all files matching pattern
- Returns list of full paths
- Example:
  ```python
  [
      "C:\\...\\backend\\pdfs\\Maternity_Policy.pdf",
      "C:\\...\\backend\\pdfs\\Leave_Policy.pdf",
      "C:\\...\\backend\\pdfs\\Dress_Code.pdf"
  ]
  ```

---

### **Code:**
```python
        if not pdf_files:
            print(f"Warning: No PDF files found in {pdf_folder}")
            print(f"Please add PDF files to {pdf_folder} folder")
            return
```

### **Explanation:**

**Line 79-82: Handle No PDFs**

**`if not pdf_files:`**
- If list is empty (no PDFs found)
- `[]` is "falsy" in Python

**`print(...) print(...)`**
- Show helpful warnings
- Tell user where to add PDFs

**`return`**
- Exit method early
- `vector_store` and `qa_chain` stay `None`
- Chatbot won't be ready but won't crash

---

### **Code:**
```python
        print(f"Found {len(pdf_files)} PDF file(s) in backend/pdfs:")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")
```

### **Explanation:**

**Line 84: `print(f"Found {len(pdf_files)} PDF file(s)...")`**
- f-string = formatted string (variables in `{}`)
- `len(pdf_files)` = count of PDFs
- Example output: `"Found 4 PDF file(s) in backend/pdfs:"`

**Line 85-86: Loop and Print**
- `for pdf in pdf_files:` = loop through each PDF path
- `os.path.basename(pdf)` = get just filename (not full path)
- Example:
  ```
  "C:\...\Maternity_Policy.pdf" → "Maternity_Policy.pdf"
  ```
- Output:
  ```
  Found 4 PDF file(s) in backend/pdfs:
    - Maternity_Policy.pdf
    - Leave_Policy.pdf
    - Dress_Code.pdf
    - Company_Info.pdf
  ```

---

### **Code:**
```python
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
```

### **Explanation:**

**Line 89-93: Call utils.py Function**

**`process_pdfs_to_documents(...)`**
- Defined in `utils.py`
- Takes list of PDF paths
- Returns list of Document objects

**What it does:**
1. Read each PDF
2. Extract text from all pages
3. Split text into chunks (1000 chars, 200 overlap)
4. Create Document object for each chunk
5. Add metadata (source file, chunk number)

**Parameters:**
- `pdf_files` = list of paths
- `chunk_size=1000` = max chars per chunk
- `chunk_overlap=200` = overlap between chunks

**Returns:**
```python
[
    Document(
        page_content="Maternity leave policy provides...",
        metadata={"source": "Maternity_Policy.pdf", "chunk_id": 0}
    ),
    Document(
        page_content="...6 months paid leave...",
        metadata={"source": "Maternity_Policy.pdf", "chunk_id": 1}
    ),
    # ... 148 more chunks
]
```

**Line 95-97: Error Handling**
- If no text extracted (maybe PDFs are images)
- Print warning and exit

**Line 99: Print Success**
- Example: `"Created 150 document chunks"`

---

### **Code:**
```python
        # Create vector store
        print("Building vector store...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
```

### **Explanation:**

**Line 102-103: Build FAISS Database**

**`FAISS.from_documents(documents, self.embeddings)`**

**What happens step-by-step:**

1. **For each Document:**
   ```python
   text = "Maternity leave policy provides 6 months..."
   ```

2. **Convert to embedding:**
   ```python
   embedding = self.embeddings.embed_query(text)
   # Returns: [0.023, 0.156, -0.089, ..., 0.234]  # 768 numbers
   ```

3. **Store in FAISS:**
   ```python
   FAISS stores:
   - Embedding vectors (768 floats × 150 chunks)
   - Original documents (text + metadata)
   - Index for fast searching
   ```

4. **Build index:**
   - FAISS organizes embeddings for fast similarity search
   - Uses special algorithms (clustering, indexing)
   - Searches millions of vectors in milliseconds

**Result:**
- `self.vector_store` now contains searchable database of all PDF chunks
- Ready for questions!

---

### **Code:**
```python
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
```

### **Explanation:**

**Line 106-110: Create Retriever**

**`self.vector_store.as_retriever(...)`**
- Retriever = object that searches the database
- Wrapper around FAISS with extra LangChain features

**`search_type="similarity"`**
- Use similarity search (cosine similarity)
- Alternative: `"mmr"` (maximum marginal relevance - more diverse results)

**`search_kwargs={"k": 4}`**
- `k=4` = return top 4 most similar chunks
- More chunks = more context but slower
- 4 is good balance

**How it works:**
```python
question = "What is maternity leave?"

# 1. Convert question to embedding
q_embedding = [0.12, 0.78, -0.23, ...]

# 2. Compare with all chunk embeddings (cosine similarity)
scores = [
    (chunk_0, 0.89),  # Very similar!
    (chunk_1, 0.85),  # Very similar!
    (chunk_47, 0.72), # Somewhat similar
    (chunk_12, 0.68), # Somewhat similar
    (chunk_3, 0.23),  # Not similar
    ...
]

# 3. Return top 4
return [chunk_0, chunk_1, chunk_47, chunk_12]
```

---

**Line 112-118: Create QA Chain**

**`ConversationalRetrievalChain.from_llm(...)`**
- Factory method to create chain
- Configures all components automatically

**Parameters:**

**`llm=self.llm`**
- The AI model (Google Gemini)
- Used for generating answers

**`retriever=retriever`**
- The search engine (FAISS)
- Used for finding relevant chunks

**`memory=self.memory`**
- Conversation history
- Used for context in follow-up questions

**`return_source_documents=True`**
- Include source chunks in response
- Needed for showing citations

**`verbose=False`**
- Don't print debug info
- Set to `True` to see internal steps

**What the chain does:**
```
User Question
    ↓
1. Retrieve relevant chunks (retriever)
    ↓
2. Get conversation history (memory)
    ↓
3. Build prompt:
   "Context: [chunks]
    History: [previous messages]
    Question: [user question]
    Answer:"
    ↓
4. Send to LLM (Gemini)
    ↓
5. Get answer
    ↓
6. Save to memory
    ↓
Return: {answer, source_documents}
```

---

## 💬 Section 4: Asking Questions (Lines 123-190)

### **Code:**
```python
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        if not self.qa_chain:
            return {
                "answer": "No PDFs loaded. Please add PDF files to backend/pdfs folder and restart the application.",
                "sources": "",
                "source_documents": []
            }
```

### **Explanation:**

**Line 123: Method Signature**
- `def ask(self, question: str) -> Dict[str, Any]:`
- Takes: `question` (string)
- Returns: dictionary with answer, sources, documents

**Line 125-130: Check if Ready**
- `if not self.qa_chain:` = if no PDFs loaded
- Return error message
- Prevents crash

---

### **Code:**
```python
        # Handle greetings and general conversation
        question_lower = question.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']

        if question_lower in greetings or any(greeting in question_lower for greeting in greetings):
            return {
                "answer": "Hello! I'm your AI assistant. I can help you with:\n\n✅ General questions (math, facts, explanations, coding, etc.)\n✅ Company policy questions from loaded PDFs:\n   - Company headquarters and information\n   - Leave policies\n   - Maternity policy\n   - Dress code policy\n\nWhat would you like to know?",
                "sources": "",
                "source_documents": []
            }
```

### **Explanation:**

**Line 133: Normalize Question**
- `.lower()` = convert to lowercase ("Hi" → "hi")
- `.strip()` = remove spaces from start/end ("  hi  " → "hi")
- Easier to compare

**Line 134: List of Greetings**
- All common greetings in one list
- Easy to add more

**Line 136: Check if Greeting**
- `question_lower in greetings` = exact match
- `any(greeting in question_lower for ...)` = partial match
  - Example: "hi there" contains "hi"

**If greeting:**
- Return friendly welcome message
- No PDF search needed
- No API call to Gemini

---

### **Code:**
```python
        # Check if question is likely about the documents or general knowledge
        # Keywords that suggest PDF/document-specific questions
        pdf_keywords = [
            'company', 'policy', 'policies', 'leave', 'maternity', 'dress code',
            'headquarters', 'office', 'employee', 'document', 'pdf', 'according to',
            'qss', 'technooft', 'organization', 'work', 'salary', 'benefit'
        ]

        is_pdf_question = any(keyword in question_lower for keyword in pdf_keywords)
```

### **Explanation:**

**Line 145-149: Define PDF Keywords**
- Words that likely mean user wants PDF info
- Add more based on your PDFs

**Line 151: Check if PDF Question**
- `any(keyword in question_lower for keyword in pdf_keywords)`
- Loops through keywords
- Returns `True` if ANY keyword found

**Examples:**
```python
"What is the company headquarters?" → is_pdf_question = True (has "company" and "headquarters")
"What is 2+2?" → is_pdf_question = False (no keywords)
"Tell me about the leave policy" → is_pdf_question = True (has "leave" and "policy")
```

---

### **Code:**
```python
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
```

### **Explanation:**

**Line 155: Call QA Chain**

**`self.qa_chain({"question": question})`**

**What happens (RAG process):**

1. **Retrieve:**
   ```python
   # Convert question to embedding
   q_emb = embeddings.embed_query(question)

   # Search FAISS
   chunks = vector_store.similarity_search(q_emb, k=4)
   # Returns top 4 most relevant chunks
   ```

2. **Build Context:**
   ```python
   context = "\n\n".join([chunk.page_content for chunk in chunks])
   # Combines all chunk text
   ```

3. **Build Prompt:**
   ```python
   prompt = f"""
   Use the following context to answer the question.

   Context:
   {context}

   Chat History:
   {memory.chat_history}

   Question: {question}

   Answer:
   """
   ```

4. **Call Gemini API:**
   ```python
   response = llm.call(prompt)
   # Sends to Google's servers
   # Waits for response (1-3 seconds)
   ```

5. **Return:**
   ```python
   {
       "answer": "According to the policy...",
       "source_documents": [chunk1, chunk2, chunk3, chunk4]
   }
   ```

**Line 156-158: Extract Response**
- `response.get("answer", "")` = get answer (empty string if missing)
- `response.get("source_documents", [])` = get chunks (empty list if missing)
- `format_sources(...)` = convert chunks to readable text

**Example:**
```python
source_documents = [
    Document(..., metadata={"source": "Maternity_Policy.pdf"}),
    Document(..., metadata={"source": "Leave_Policy.pdf"})
]

sources = format_sources(source_documents)
# Returns: "Maternity_Policy.pdf, Leave_Policy.pdf"
```

**Line 160-164: Return Result**
- Dictionary with three keys
- App.py will display this to user

---

### **Code:**
```python
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
```

### **Explanation:**

**Line 168-169: Import and Create Message**
- `HumanMessage` = represents user message
- `messages = [HumanMessage(content=question)]`
- LLMs work with message objects (not plain strings)

**Line 170: Call LLM Directly**
- `self.llm(messages)` = send to Gemini
- **No PDF search!**
- **No context added!**
- Pure AI knowledge

**Example:**
```python
question = "What is 2+2?"
messages = [HumanMessage(content="What is 2+2?")]
response = llm(messages)
# API call to Google: just the question
# Response: "2+2 equals 4"
```

**Line 171: Extract Answer**
- `response.content` = answer text
- `hasattr(...)` = check if object has attribute (safe)
- Fallback to `str(response)` if no `.content`

**Line 173-177: Return with Note**
- Add note at end of answer
- Reminds user this is general (not from PDFs)

**Line 178-189: Error Handling**
- `try/except` = catch errors
- If direct LLM call fails, use QA chain as backup
- Prevents crash

---

## ✅ Section 5: Helper Methods (Lines 192-198)

### **Code:**
```python
    def is_ready(self) -> bool:
        """Check if chatbot is ready (has PDFs loaded)."""
        return self.qa_chain is not None
```

### **Explanation:**

**Simple check:**
- If `qa_chain` exists → PDFs loaded → ready
- If `qa_chain` is `None` → no PDFs → not ready

**Used in app.py:**
```python
if not chatbot.is_ready():
    st.warning("No PDFs found!")
```

---

### **Code:**
```python
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()
```

### **Explanation:**

**Clear all messages:**
- `self.memory.clear()` = delete chat history
- Start fresh conversation
- Chatbot "forgets" previous questions

**Used when:**
- User clicks "Clear Chat" button
- Want to start new topic

---

## 🏭 Section 6: Factory Function (Lines 165-167)

### **Code:**
```python
def create_chatbot() -> PDFChatbot:
    """Create and return a PDFChatbot instance."""
    return PDFChatbot()
```

### **Explanation:**

**Factory function:**
- Simple wrapper
- Could add configuration logic here later
- Example:
  ```python
  def create_chatbot(provider="gemini"):
      return PDFChatbot(provider)
  ```

**Used in app.py:**
```python
chatbot = create_chatbot()
```

---

## 🔄 Complete Example Flow

### **User asks: "What is the maternity policy?"**

```python
# 1. Question received
question = "What is the maternity policy?"

# 2. Ask method called
chatbot.ask(question)

# 3. Check greeting? No
question_lower = "what is the maternity policy?"
is_greeting = False  # Not in greetings list

# 4. Check PDF keywords? Yes
"maternity" in pdf_keywords → True
"policy" in pdf_keywords → True
is_pdf_question = True

# 5. Call QA chain
response = self.qa_chain({"question": question})

    # 5a. Retriever searches FAISS
    q_embedding = [0.12, 0.78, ...]
    chunks = vector_store.similarity_search(q_embedding, k=4)
    # Returns 4 most relevant chunks

    # 5b. Build context
    context = """
    Chunk 1: Maternity leave policy provides 6 months paid leave...
    Chunk 2: ...can be extended to 8 months in special cases...
    Chunk 3: All female employees are eligible for maternity leave...
    Chunk 4: Benefits include full salary and health coverage...
    """

    # 5c. Build prompt
    prompt = f"""
    Context: {context}
    Question: {question}
    Answer:
    """

    # 5d. API call to Gemini
    POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
    Body: {"prompt": prompt, ...}

    # 5e. Receive response
    {
        "answer": "According to the policy, maternity leave is 6 months paid, extendable to 8 months in special cases...",
        "source_documents": [chunk1, chunk2, chunk3, chunk4]
    }

# 6. Format sources
sources = "Maternity_Policy.pdf, Leave_Policy.pdf"

# 7. Return to app
return {
    "answer": "According to the policy...",
    "sources": "Maternity_Policy.pdf, Leave_Policy.pdf",
    "source_documents": [...]
}

# 8. App displays answer with sources
```

---

## 🎯 Key Concepts Summary

### **1. Embeddings**
- Text → 768 numbers
- Similar text → similar numbers
- Used for searching

### **2. FAISS**
- Stores embeddings
- Fast similarity search
- Returns most relevant chunks

### **3. RAG (Retrieval-Augmented Generation)**
- Retrieve → Augment → Generate
- Search PDFs → Add context → AI answer

### **4. Chains**
- Automate multi-step processes
- Connect retriever + memory + LLM
- Handle everything automatically

### **5. Memory**
- Store conversation history
- Enable follow-up questions
- Maintain context

---

## 📊 Performance Notes

### **Initialization (First Time):**
- Download embedding model: ~30 seconds
- Load PDFs: ~5-10 seconds
- Build FAISS: ~2-3 seconds
- **Total: ~40-45 seconds**

### **Initialization (Subsequent Times):**
- Load PDFs: ~5-10 seconds
- Build FAISS: ~2-3 seconds
- **Total: ~7-13 seconds**

### **Per Question:**
- Greeting: <1ms (instant)
- General question: 1-2 seconds (API call)
- PDF question: 2-4 seconds (search + API call)

### **Memory Usage:**
- Embedding model: ~500MB
- FAISS index: ~50-100MB (depends on PDF count)
- **Total: ~550-600MB**

---

## 🐛 Common Issues & Solutions

### **Issue 1: "No PDFs found"**
**Cause:** No PDFs in backend/pdfs/
**Solution:** Add PDFs to folder, restart

### **Issue 2: Slow responses**
**Cause:** First question loads embedding model
**Solution:** Normal, subsequent questions faster

### **Issue 3: Wrong answers**
**Cause:** Question not matching PDF keywords
**Solution:** Add more keywords to `pdf_keywords` list

### **Issue 4: Out of memory**
**Cause:** Too many PDFs
**Solution:** Reduce PDFs or increase chunk_size

---

**Now you understand exactly how chatbot.py works! 🎉**

Every line of code explained, from imports to API calls to returning answers.
