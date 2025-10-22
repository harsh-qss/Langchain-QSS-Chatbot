# 📚 PDF Chatbot - AI-Powered Document Q&A

A beautiful, intelligent chatbot that answers questions from your PDF documents using **LangChain**, **LangGraph**, and state-of-the-art AI models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.20-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red.svg)

---

## ✨ Features

- 📤 **Multi-PDF Upload**: Process multiple documents simultaneously
- 🤖 **Multiple LLM Support**: Choose between Gemini, OpenAI, or Anthropic
- 💬 **Conversational Memory**: Maintains context across conversations
- 🔍 **Source Citations**: Shows which documents answers came from
- 🎨 **Elegant UI**: Beautiful Streamlit interface
- 🔄 **LangGraph Workflow**: Visualized retrieval and response orchestration
- ⚡ **Efficient Retrieval**: FAISS/Chroma vector stores for fast similarity search
- 🧩 **Modular Design**: Clean, maintainable code structure

---

## 🏗️ Project Structure

```
pdf-chatbot/
├── app.py                    # Streamlit web interface
├── chatbot.py                # LangChain chatbot logic
├── langgraph_workflow.py     # LangGraph orchestration
├── utils.py                  # PDF processing utilities
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── uploaded_pdfs/           # Temporary PDF storage (auto-created)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API key for at least one LLM provider (Gemini/OpenAI/Anthropic)

### Step 1: Clone or Download

```bash
cd pdf-chatbot
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- LangChain & LangGraph
- Streamlit for UI
- PDF processing libraries
- Vector stores (FAISS/Chroma)
- Embedding models

### Step 4: Set Up Environment Variables

1. Copy the example environment file:
```bash
copy .env.example .env  # Windows
cp .env.example .env    # Mac/Linux
```

2. Edit `.env` and add your API key:

**For Gemini (Recommended - Free tier available):**
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
```

**For OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

**For Anthropic (Claude):**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎯 How to Use

### 1. Upload PDFs
- Click **"Browse files"** in the sidebar
- Select one or more PDF files
- Click **"🚀 Process Documents"**

### 2. Ask Questions
- Type your question in the chat input
- Press Enter or click Send
- Get AI-powered answers with source citations

### 3. Example Questions

**For a storybook:**
- "What is the main character's name?"
- "Summarize the story in 3 sentences"
- "What lesson does this story teach?"

**For a technical document:**
- "What are the key requirements?"
- "Explain the architecture described"
- "What are the main features?"

**For multiple documents:**
- "Compare the approaches in different documents"
- "What topics are covered across all documents?"

---

## 🔑 Getting API Keys

### Google Gemini (Free & Recommended)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy and paste into `.env`

### OpenAI
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Create a new API key
4. Copy and paste into `.env`

### Anthropic (Claude)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Generate an API key
4. Copy and paste into `.env`

---

## ⚙️ Configuration Options

Edit `.env` to customize:

```env
# Choose LLM provider
LLM_PROVIDER=gemini  # or openai, anthropic

# Model selection
GEMINI_MODEL=gemini-1.5-flash
OPENAI_MODEL=gpt-3.5-turbo
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector store
VECTOR_STORE=faiss  # or chroma

# Text chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## 🧪 Demo Scenario

### Example: Analyzing a Storybook

1. **Upload PDF**: Upload "The Little Prince" or any storybook
2. **Process**: Click "Process Documents"
3. **Ask Questions**:
   - "Who is the main character?"
   - "What planets does he visit?"
   - "What is the moral of the story?"
4. **Review Sources**: See which pages the answers came from

---

## 🎨 Architecture Overview

### Data Flow

```
PDFs → Text Extraction → Chunking → Embeddings → Vector Store
                                                      ↓
User Question → Retrieval → Context Formation → LLM → Answer
```

### LangGraph Workflow

```
1. Validate Input
   ↓
2. Retrieve Documents (Vector Similarity Search)
   ↓
3. Format Context (Combine Retrieved Chunks)
   ↓
4. Generate Answer (LLM Processing)
   ↓
5. Format Response (Add Source Citations)
```

---

## 💡 Optional Improvements for Demo

### 1. Multi-User Support
- Add user authentication
- Separate vector stores per user
- Session management

### 2. Caching
- Cache embeddings for faster reprocessing
- Store vector stores persistently
- Cache frequent queries

### 3. Advanced Features
- PDF preview within the app
- Highlight relevant passages
- Export chat history
- Audio input/output
- Multi-language support

### 4. Performance Optimization
- Batch processing for large PDFs
- Async operations
- GPU acceleration for embeddings
- CDN for faster loading

### 5. Analytics Dashboard
- Query statistics
- Response quality metrics
- Usage tracking

---

## 🐛 Troubleshooting

### Issue: "No module named 'X'"
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "API key not found"
**Solution**: Check your `.env` file
- Ensure the file is named `.env` (not `.env.txt`)
- Verify the API key is correct
- Restart the application

### Issue: "Out of memory"
**Solution**: Reduce chunk size
```env
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Issue: "Slow processing"
**Solution**: Use FAISS instead of Chroma
```env
VECTOR_STORE=faiss
```

---

## 📦 Dependencies

### Core Libraries
- **LangChain**: Orchestration framework
- **LangGraph**: Workflow visualization
- **Streamlit**: Web interface

### AI Models
- **Google Gemini**: Free & powerful
- **OpenAI GPT**: Industry standard
- **Anthropic Claude**: High quality responses

### Vector Stores
- **FAISS**: Fast similarity search
- **Chroma**: Persistent storage

### Utilities
- **pypdf**: PDF text extraction
- **sentence-transformers**: Text embeddings

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM providers
- More vector store options
- Enhanced UI features
- Better error handling
- Unit tests

---

## 📄 License

MIT License - Feel free to use for personal or commercial projects.

---

## 🙏 Acknowledgments

- **LangChain**: For the amazing framework
- **Streamlit**: For the beautiful UI library
- **HuggingFace**: For free embedding models
- **Google/OpenAI/Anthropic**: For powerful LLMs

---

## 📞 Support

Having issues? Check:
1. Environment variables are set correctly
2. All dependencies are installed
3. Python version is 3.8+
4. API keys are valid

---

## 🎓 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**Built with ❤️ using Python, LangChain, and AI**

*Ready for your Thursday demo! 🚀*
