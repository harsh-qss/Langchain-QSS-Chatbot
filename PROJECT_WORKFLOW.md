# 🔄 Project Workflow - Understanding the PDF Chatbot

This document explains **exactly** how the PDF Chatbot works, step-by-step. Perfect for beginners to LangChain, GenAI, and coding.

---

## 📚 Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [Key Concepts Explained](#key-concepts-explained)
3. [Technology Stack](#technology-stack)
4. [Complete Code Flow](#complete-code-flow)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [How LangChain Works Here](#how-langchain-works-here)
7. [API Calls Explained](#api-calls-explained)

---

## What is This Project?

A chatbot that:
- ✅ Reads multiple PDF files
- ✅ Understands what's inside them
- ✅ Answers your questions based on the PDF content
- ✅ Can also answer general questions (like a normal AI)

**Example:**
- You upload company policy PDFs
- You ask: "What is the maternity leave policy?"
- Chatbot searches PDFs and gives you the answer with source

---

## Key Concepts Explained

### 1. **LangChain** 🔗
Think of it as a "toolkit" that helps us build AI applications easily.

**What it does:**
- Connects different AI components together (like LEGO blocks)
- Handles PDF reading, text splitting, searching, and AI responses
- Makes complex AI tasks simple

### 2. **Embeddings** 🧮
Converting text into numbers (vectors) so computers can understand similarity.

**Simple Analogy:**
- "Dog" → [0.2, 0.8, 0.1, ...]
- "Puppy" → [0.21, 0.79, 0.12, ...]
- "Car" → [0.9, 0.1, 0.3, ...]

Notice: "Dog" and "Puppy" have similar numbers (close meaning), but "Car" is different.

**In Our Project:**
- We convert PDF text chunks into embeddings
- When you ask a question, we convert it to embeddings
- Find PDF chunks with similar embeddings
- Those chunks likely contain the answer!

### 3. **Vector Database (FAISS)** 🗄️
A special database that stores embeddings and finds similar ones quickly.

**Simple Analogy:**
Like a library where:
- Books are organized by topic similarity (not alphabetically)
- You describe what you want, and it finds the most relevant books instantly

**In Our Project:**
- Stores all PDF text chunks as embeddings
- When you ask a question, it finds the most relevant chunks in milliseconds

### 4. **Retrieval-Augmented Generation (RAG)** 🎯
The core technique this chatbot uses.

**How it works:**
1. **Retrieve:** Find relevant information from PDFs
2. **Augment:** Add that information to the question
3. **Generate:** Let AI create an answer based on retrieved info

**Without RAG:**
- Question: "What is our maternity policy?"
- AI: "I don't know" (AI doesn't have your company docs)

**With RAG:**
- Question: "What is our maternity policy?"
- System finds relevant PDF section: "Maternity leave is 6 months paid..."
- AI gets question + PDF content
- AI: "According to your policy, maternity leave is 6 months paid..."

### 5. **Chunks** ✂️
Breaking large PDFs into smaller pieces.

**Why?**
- AI has a limit on how much text it can read at once
- Smaller chunks = more precise searching
- Better accuracy

**Example:**
```
Big PDF (100 pages) → Split into 150 chunks of ~1000 characters each
```

### 6. **Prompts** 💬
Instructions we give to the AI.

**Example Prompt:**
```
You are a helpful assistant. Answer the question based on the following context:

Context: {retrieved_pdf_text}
Question: {user_question}

Answer:
```

---

## Technology Stack

### **1. Streamlit** 🖥️
**What:** Python library for creating web interfaces
**Role:** Creates the chat interface you see in browser
**File:** `app.py`

### **2. LangChain** 🔗
**What:** Framework for building AI applications
**Role:** Orchestrates everything - PDF processing, searching, AI responses
**File:** `chatbot.py`, `utils.py`

### **3. Google Gemini** 🤖
**What:** Google's AI model (like ChatGPT)
**Role:** Generates intelligent answers to your questions
**API Call:** Happens inside LangChain

### **4. FAISS** 🗄️
**What:** Vector database by Facebook
**Role:** Stores and searches PDF embeddings super fast
**File:** `chatbot.py` (line 104)

### **5. Sentence Transformers** 🧮
**What:** Creates embeddings from text
**Role:** Converts text to numbers for similarity search
**File:** `chatbot.py` (line 47)

### **6. PyPDF** 📄
**What:** PDF reading library
**Role:** Extracts text from PDF files
**File:** `utils.py` (line 40)

---

## Complete Code Flow

### 🚀 **Step 1: Application Starts** (`app.py`)

```
User runs: streamlit run app.py
↓
app.py loads
↓
Creates chatbot instance (calls chatbot.py)
```

**Code:** `app.py` line 17-21
```python
if "chatbot" not in st.session_state:
    st.session_state.chatbot = create_chatbot()
```

---

### 📂 **Step 2: Chatbot Initialization** (`chatbot.py`)

#### **2.1: Load Environment Variables**
```
Reads .env file
↓
Gets API key, model name, settings
```

**Code:** `chatbot.py` line 23
```python
load_dotenv()  # Loads GOOGLE_API_KEY, etc.
```

#### **2.2: Initialize Embeddings Model**
```
Loads sentence-transformers model
↓
Downloads model (first time only, ~90MB)
↓
Ready to convert text → embeddings
```

**Code:** `chatbot.py` line 44-51
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**What happens:** Downloads AI model that converts text to numbers

#### **2.3: Initialize Google Gemini**
```
Connects to Google's AI API
↓
Uses your API key from .env
↓
Ready to generate answers
```

**Code:** `chatbot.py` line 54-65
```python
self.llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7
)
```

**What happens:** Establishes connection to Google's servers

#### **2.4: Auto-Load PDFs**
```
Scans backend/pdfs/ folder
↓
Finds all .pdf files
↓
Calls _auto_load_pdfs() method
```

**Code:** `chatbot.py` line 73-120

---

### 📄 **Step 3: PDF Processing** (`utils.py`)

#### **3.1: Read PDF Files**
```python
For each PDF:
    Open file
    ↓
    Extract text from each page
    ↓
    Combine all pages into one big text
```

**Code:** `utils.py` line 29-54
```python
reader = PdfReader(pdf_path)
for page in reader.pages:
    text = page.extract_text()
```

**Example Output:**
```
"Page 1: Company Policy... Page 2: Leave Policy is..."
```

#### **3.2: Split Text into Chunks**
```python
Take big text
↓
Split into ~1000 character chunks
↓
Overlap 200 characters between chunks
```

**Code:** `utils.py` line 107-115
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(text)
```

**Why overlap?** So important info at chunk boundaries doesn't get lost.

**Example:**
```
Chunk 1: "...leave policy is 6 months..."
Chunk 2: "...6 months paid maternity leave..."
         ↑ Overlap ensures context
```

#### **3.3: Create Document Objects**
```python
For each chunk:
    Create Document object
    ↓
    Add metadata (source file name, chunk number)
```

**Code:** `utils.py` line 118-138
```python
doc = Document(
    page_content=chunk,
    metadata={"source": "Maternity_Policy.pdf", "chunk_id": 5}
)
```

---

### 🗄️ **Step 4: Create Vector Database** (`chatbot.py`)

#### **4.1: Convert Chunks to Embeddings**
```python
For each chunk:
    Text → Sentence Transformer → Embedding (768 numbers)
```

**Example:**
```
"Maternity leave is 6 months" → [0.23, 0.67, -0.12, ..., 0.45]
                                   ↑
                                768 numbers
```

#### **4.2: Store in FAISS**
```python
FAISS database created
↓
All embeddings stored
↓
Ready for fast searching
```

**Code:** `chatbot.py` line 104
```python
self.vector_store = FAISS.from_documents(documents, self.embeddings)
```

**What happens:**
- FAISS creates an index (like a table of contents)
- Optimizes for fast similarity search
- All in memory (RAM) for speed

#### **4.3: Create Retriever**
```python
Retriever = search engine for our PDFs
↓
Configured to return top 4 most relevant chunks
```

**Code:** `chatbot.py` line 107-110
```python
retriever = self.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

---

### 🔗 **Step 5: Create QA Chain** (`chatbot.py`)

#### **What is a Chain?**
A "chain" connects multiple steps together automatically.

**Our QA Chain does:**
```
User Question
    ↓
1. Search PDFs (Retriever)
    ↓
2. Find top 4 relevant chunks
    ↓
3. Combine chunks with question
    ↓
4. Send to Google Gemini
    ↓
5. Get AI answer
    ↓
Return answer + sources
```

**Code:** `chatbot.py` line 112-118
```python
self.qa_chain = ConversationalRetrievalChain.from_llm(
    llm=self.llm,              # Google Gemini
    retriever=retriever,        # PDF search
    memory=self.memory,         # Remember conversation
    return_source_documents=True
)
```

---

### 💬 **Step 6: User Asks a Question** (`app.py`)

#### **6.1: User Types in Chat**
```
User types: "What is the maternity policy?"
↓
Streamlit captures input
↓
Calls chatbot.ask(question)
```

**Code:** `app.py` line 44-56

#### **6.2: Question Routing** (`chatbot.py`)

**The chatbot is smart! It decides:**

```python
Is it a greeting? (hi, hello)
    ↓ YES → Return friendly greeting
    ↓ NO
    ↓
Does it mention PDF keywords? (company, policy, leave, etc.)
    ↓ YES → Search PDFs
    ↓ NO → Use AI directly for general answer
```

**Code:** `chatbot.py` line 132-190

---

### 🔍 **Step 7: PDF Search (RAG Process)**

#### **7.1: Question → Embedding**
```python
"What is the maternity policy?"
    ↓
Sentence Transformer
    ↓
[0.15, 0.82, -0.34, ..., 0.61]  # 768 numbers
```

#### **7.2: Similarity Search in FAISS**
```python
FAISS compares question embedding with all chunk embeddings
    ↓
Calculates similarity scores
    ↓
Returns top 4 most similar chunks
```

**Example Results:**
```
Chunk #47 from Maternity_Policy.pdf (similarity: 0.89)
Chunk #48 from Maternity_Policy.pdf (similarity: 0.85)
Chunk #12 from Leave_Policy.pdf (similarity: 0.72)
Chunk #03 from Employee_Benefits.pdf (similarity: 0.68)
```

#### **7.3: Create Context**
```python
Retrieved chunks are combined:

context = """
Chunk 1: Maternity leave policy provides 6 months paid leave...
Chunk 2: ...extended to 8 months in special cases...
Chunk 3: All female employees are eligible...
Chunk 4: Benefits include full salary and health coverage...
"""
```

---

### 🤖 **Step 8: API Call to Google Gemini**

#### **8.1: Build Prompt**
```
LangChain automatically creates a prompt like:

System: You are a helpful assistant. Answer based on the context.

Context:
Maternity leave policy provides 6 months paid leave...
extended to 8 months in special cases...
All female employees are eligible...
Benefits include full salary and health coverage...

Question: What is the maternity policy?

Answer:
```

#### **8.2: Send to Google Gemini API**
```python
Request sent over internet
    ↓
Google's servers receive prompt
    ↓
Gemini AI processes it
    ↓
Generates intelligent answer
    ↓
Response sent back
```

**Code:** Happens inside `chatbot.py` line 155
```python
response = self.qa_chain({"question": question})
```

**Actual API Call (behind the scenes):**
```
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
Headers: {
    "Authorization": "Bearer YOUR_API_KEY"
}
Body: {
    "prompt": "System: You are...",
    "temperature": 0.7,
    "max_tokens": 1024
}
```

#### **8.3: Receive Response**
```python
{
    "answer": "According to the policy, maternity leave is 6 months paid,
               extendable to 8 months in special cases. All female employees
               are eligible and receive full salary plus health coverage.",
    "source_documents": [chunk1, chunk2, chunk3, chunk4]
}
```

---

### 📤 **Step 9: Format and Display Response** (`app.py`)

#### **9.1: Extract Answer and Sources**
```python
answer = "According to the policy..."
sources = "Maternity_Policy.pdf, Leave_Policy.pdf"
```

**Code:** `chatbot.py` line 156-158
```python
answer = response.get("answer", "")
source_documents = response.get("source_documents", [])
sources = format_sources(source_documents)
```

#### **9.2: Display in Chat**
```python
Streamlit shows:
    Bot: According to the policy, maternity leave is 6 months...

    📄 Sources: Maternity_Policy.pdf, Leave_Policy.pdf
```

**Code:** `app.py` line 58-64

---

### 🧠 **Step 10: Conversation Memory**

#### **How Memory Works:**
```python
Every question and answer is stored in memory:

Memory:
    User: "What is maternity leave?"
    Bot: "6 months paid..."

    User: "Can it be extended?"  ← Knows "it" = maternity leave
    Bot: "Yes, to 8 months..."
```

**Code:** `chatbot.py` line 37-42
```python
self.memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

---

## File-by-File Breakdown

### **1. app.py** - User Interface

**Purpose:** Creates the web page you see

**Key Sections:**
```python
Lines 10-14:  Configure page (title, icon)
Lines 17-24:  Initialize chatbot (loads PDFs)
Lines 27-33:  Check if PDFs loaded
Lines 36-41:  Display chat history
Lines 44-79:  Handle user input and show responses
Lines 82-97:  Sidebar (about, clear chat)
```

**Flow:**
```
User opens browser
    ↓
Streamlit loads app.py
    ↓
Creates chatbot
    ↓
Shows chat interface
    ↓
Waits for user input
```

---

### **2. chatbot.py** - Brain of the Application

**Purpose:** All AI logic lives here

**Key Sections:**
```python
Lines 11-17:  Imports (LangChain, AI models, etc.)
Lines 29-70:  __init__ - Initialize everything
Lines 73-120: _auto_load_pdfs - Load and process PDFs
Lines 123-190: ask - Handle user questions
Lines 192-194: is_ready - Check if chatbot ready
Lines 196-198: clear_memory - Reset conversation
```

**Main Classes/Functions:**
- `PDFChatbot` class - Main chatbot
- `create_chatbot()` - Factory function to create chatbot

---

### **3. utils.py** - PDF Processing Utilities

**Purpose:** Read and prepare PDFs

**Key Functions:**
```python
Lines 29-54:  extract_text_from_pdf - Read one PDF
Lines 57-78:  extract_text_from_multiple_pdfs - Read many PDFs
Lines 81-140: chunk_text - Split text into chunks
Lines 143-169: process_pdfs_to_documents - Complete pipeline
Lines 172-198: format_sources - Format source citations
```

---

### **4. .env** - Configuration

**Purpose:** Store secrets and settings

**Contents:**
```env
GOOGLE_API_KEY=your_key_here       # API authentication
GEMINI_MODEL=models/gemini-2.5-flash  # Which AI model
CHUNK_SIZE=1000                     # How big are chunks
CHUNK_OVERLAP=200                   # Overlap between chunks
```

---

### **5. requirements.txt** - Dependencies

**Purpose:** List all required libraries

**Key Libraries:**
```
langchain==0.1.20              # Main framework
langchain-google-genai==1.0.1  # Google AI integration
faiss-cpu>=1.8.0               # Vector database
sentence-transformers>=2.5.1   # Embeddings
pypdf>=4.1.0                   # PDF reading
streamlit>=1.32.0              # Web interface
```

---

## How LangChain Works Here

### **LangChain Components Used:**

#### **1. Document Loaders** ✅
- **What:** Read data from sources (PDFs)
- **Where:** `utils.py` - `extract_text_from_pdf()`
- **How:** PyPDF extracts text, LangChain wraps it in Document objects

#### **2. Text Splitters** ✂️
- **What:** Break large text into chunks
- **Where:** `utils.py` - `RecursiveCharacterTextSplitter`
- **How:** Intelligently splits on paragraphs, sentences, then characters

#### **3. Embeddings** 🧮
- **What:** Convert text to vectors
- **Where:** `chatbot.py` - `HuggingFaceEmbeddings`
- **How:** Uses sentence-transformers model locally

#### **4. Vector Stores** 🗄️
- **What:** Store and search embeddings
- **Where:** `chatbot.py` - `FAISS`
- **How:** Creates searchable index in memory

#### **5. Retrievers** 🔍
- **What:** Find relevant documents
- **Where:** `chatbot.py` - `vector_store.as_retriever()`
- **How:** Searches FAISS for similar embeddings

#### **6. Chains** 🔗
- **What:** Connect multiple steps
- **Where:** `chatbot.py` - `ConversationalRetrievalChain`
- **How:** Orchestrates: retrieve → prompt → AI → response

#### **7. Memory** 🧠
- **What:** Remember conversation history
- **Where:** `chatbot.py` - `ConversationBufferMemory`
- **How:** Stores all previous messages

#### **8. LLMs** 🤖
- **What:** AI model that generates text
- **Where:** `chatbot.py` - `ChatGoogleGenerativeAI`
- **How:** Sends prompts to Google's API

---

## API Calls Explained

### **API Call #1: Google Gemini (Answer Generation)**

**When:** Every time user asks a PDF question

**Where in Code:** `chatbot.py` line 155
```python
response = self.qa_chain({"question": question})
```

**What Happens Behind the Scenes:**
```
1. LangChain builds prompt with context
2. Sends HTTPS request to Google's API
3. Endpoint: https://generativelanguage.googleapis.com/v1beta/...
4. Headers: Authorization with your API key
5. Body: JSON with prompt, model name, settings
6. Google processes request (takes 1-3 seconds)
7. Returns JSON with answer
8. LangChain parses response
```

**Request Structure:**
```json
{
  "contents": [{
    "parts": [{
      "text": "Context: ....\n\nQuestion: What is maternity policy?\n\nAnswer:"
    }]
  }],
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 1024
  }
}
```

**Response Structure:**
```json
{
  "candidates": [{
    "content": {
      "parts": [{
        "text": "According to the policy, maternity leave is..."
      }]
    }
  }]
}
```

### **API Call #2: HuggingFace (Embeddings) - LOCAL**

**When:** During PDF loading and question embedding

**Where in Code:** `chatbot.py` line 47

**Important:** This does NOT call an API! It runs locally on your computer.

**What Happens:**
```
1. Text input
2. Loads model from disk (sentence-transformers)
3. Runs through neural network
4. Outputs 768 numbers
5. All happens on your CPU (no internet needed)
```

---

## Summary: Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ USER STARTS APPLICATION                                      │
│ Command: streamlit run app.py                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ APP.PY - Streamlit Interface                                 │
│ • Load page configuration                                    │
│ • Initialize session state                                   │
│ • Call create_chatbot()                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CHATBOT.PY - Initialization                                  │
│ • Load .env variables                                        │
│ • Initialize HuggingFace embeddings (local)                  │
│ • Connect to Google Gemini API                               │
│ • Create conversation memory                                 │
│ • Call _auto_load_pdfs()                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ UTILS.PY - PDF Processing                                    │
│ • Find all PDFs in backend/pdfs/                             │
│ • For each PDF:                                              │
│   - Extract text from all pages                              │
│   - Split into 1000-char chunks (200 overlap)                │
│   - Create Document objects with metadata                    │
│ • Return all chunks                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CHATBOT.PY - Create Vector Store                             │
│ • Convert all chunks to embeddings (768 numbers each)        │
│ • Store in FAISS vector database                             │
│ • Create retriever (search engine)                           │
│ • Build ConversationalRetrievalChain                         │
│ • Chatbot ready!                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ APP.PY - Display Chat Interface                              │
│ • Show welcome message                                       │
│ • Display chat input box                                     │
│ • Wait for user question...                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ USER ASKS QUESTION                                           │
│ Example: "What is the maternity policy?"                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CHATBOT.PY - Question Routing                                │
│ • Check if greeting → Return welcome message                 │
│ • Check if PDF question (keywords) → Search PDFs             │
│ • Otherwise → Use AI directly                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ (PDF Question Path)
┌─────────────────────────────────────────────────────────────┐
│ RETRIEVAL (RAG - Step 1)                                     │
│ • Convert question to embedding (768 numbers)                │
│ • Search FAISS for similar chunks                            │
│ • Get top 4 most relevant chunks                             │
│ • Calculate similarity scores                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ AUGMENTATION (RAG - Step 2)                                  │
│ • Combine retrieved chunks                                   │
│ • Build context from chunks                                  │
│ • Add conversation history                                   │
│ • Create complete prompt                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ GENERATION (RAG - Step 3) - API CALL                         │
│ • Send prompt to Google Gemini API                           │
│ • Endpoint: generativelanguage.googleapis.com                │
│ • Include API key in headers                                 │
│ • Temperature: 0.7                                           │
│ • Wait for response (1-3 seconds)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CHATBOT.PY - Process Response                                │
│ • Extract answer text                                        │
│ • Extract source documents                                   │
│ • Format sources (file names)                                │
│ • Save to conversation memory                                │
│ • Return to app.py                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ APP.PY - Display Response                                    │
│ • Show answer in chat                                        │
│ • Show sources in expandable section                         │
│ • Add to chat history                                        │
│ • Wait for next question...                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### **1. RAG (Retrieval-Augmented Generation) is the Core**
```
Your PDFs → Split → Embed → Store → Search → Add to Prompt → AI Answer
```

### **2. No Training Required**
- We don't train the AI on your PDFs
- We just give AI the relevant chunks when needed
- Fast and efficient!

### **3. Two Types of Processing**
- **PDF Questions:** Search docs + AI
- **General Questions:** Just AI

### **4. One API Call Per Question**
- Only to Google Gemini
- Embeddings run locally (free!)

### **5. Memory Makes it Conversational**
- Remembers previous messages
- Understands context ("it", "that", etc.)

---

**This is exactly how your PDF Chatbot works! 🎉**

Every question goes through this flow, combining retrieval (search) with generation (AI) to give accurate answers based on your documents.
