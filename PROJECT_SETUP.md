# Project Setup Guide - QSS PDF Chatbot

Complete guide to set up and run the PDF Chatbot project locally.

---

## Prerequisites

1. **Python 3.8+** - Download from https://www.python.org/downloads/
2. **Git** - Download from https://git-scm.com/downloads
3. **Google Gemini API Key (FREE)** - Get from https://makersuite.google.com/app/apikey

---

## Setup Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/harsh-qss/Langchain-QSS-Chatbot.git
cd Langchain-QSS-Chatbot
```

### Step 2: Create Virtual Environment (Optional but Recommended)

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

Wait 5-10 minutes for installation to complete.

### Step 4: Configure Environment Variables

1. Create `.env` file:

**Windows:**
```bash
copy .env.example .env
```

**Mac/Linux:**
```bash
cp .env.example .env
```

2. Open `.env` file and add your API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash
```

Replace `your_actual_api_key_here` with your actual Google Gemini API key.

### Step 5: Add PDF Files

Place your PDF files in the `backend/pdfs/` folder:

```
backend/
  pdfs/
    your_document1.pdf
    your_document2.pdf
    ...
```

### Step 6: Run the Application

**Windows (Easiest):**
```bash
RUN_APP.bat
```

**All Platforms:**
```bash
streamlit run app.py
```

The app will open automatically at http://localhost:8501

---

## How to Use

### General Questions:
- "What is 2+2?"
- "Explain Python functions"

### PDF-Specific Questions:
- "Where is the company headquarters?"
- "What is the maternity policy?"
- "What is the dress code?"

---

## Troubleshooting

### ModuleNotFoundError

```bash
pip install -r requirements.txt --force-reinstall
```

### API Key Not Found

- Check `.env` file exists
- Verify API key is correct
- No extra spaces around `=`

### No PDFs Found

- Add PDFs to `backend/pdfs/` folder
- Restart the application

### Port Already in Use

**Windows:**
```bash
taskkill /F /IM python.exe
```

**Mac/Linux:**
```bash
pkill -9 python
```

---

## Project Structure

```
├── app.py              # Streamlit interface
├── chatbot.py          # Main chatbot logic
├── utils.py            # PDF processing
├── requirements.txt    # Dependencies
├── .env                # API keys (YOU CREATE THIS)
├── RUN_APP.bat         # Startup script (Windows)
└── backend/pdfs/       # Put PDF files here
```

---

## Updating Project

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## Security Note

**NEVER** commit `.env` file to Git - it contains your API key!

---

**Setup Complete! Start chatting at http://localhost:8501** 🚀
