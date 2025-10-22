@echo off
echo ========================================
echo   PDF CHATBOT LAUNCHER
echo ========================================
echo.
echo Killing all existing Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
timeout /t 2 >nul

echo.
echo Navigating to project folder...
cd /d "C:\Users\QSS\Downloads\pdf-chatbot\pdf-chatbot"

echo.
echo Checking LangChain version...
python -c "import langchain; print('LangChain:', langchain.__version__)"

echo.
echo Testing imports...
python -c "from langchain.schema import Document; print('✓ Imports working!')"

echo.
echo Starting PDF Chatbot...
echo.
echo ========================================
echo   READY! Browser will open automatically
echo   URL: http://localhost:8501
echo ========================================
echo.
E:\miniconda\python.exe -m streamlit run app.py

pause
