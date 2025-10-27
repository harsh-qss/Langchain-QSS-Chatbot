"""
Simple PDF Chatbot - Streamlit Interface
PDFs are automatically loaded from backend/pdfs folder
Features live PDF monitoring with real-time updates
"""

import streamlit as st
from chatbot import create_chatbot

# Page configuration
st.set_page_config(
    page_title="QSS Chatbot",
    page_icon="💬",
    layout="centered"
)

# Define callback function for PDF updates
def on_pdf_update(event_type: str, filename: str):
    """Handle PDF file updates with Streamlit notifications."""
    if event_type == "created":
        st.toast(f"📄 New PDF added: {filename}", icon="✅")
    elif event_type == "deleted":
        st.toast(f"🗑️ PDF removed: {filename}", icon="⏹️")
    elif event_type == "modified":
        st.toast(f"🔄 PDF updated: {filename}", icon="🔄")
    elif event_type == "error":
        st.toast(f"❌ Error: {filename}", icon="⚠️")

# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    with st.spinner("Loading PDFs and initializing chatbot..."):
        try:
            st.session_state.chatbot = create_chatbot(
                enable_watcher=True,
                on_update_callback=on_pdf_update
            )
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()

# Title
st.title("💬 QSS Chatbot")

# Check if chatbot is ready
if not st.session_state.chatbot.is_ready():
    st.warning("⚠️ No PDF files found in backend/pdfs folder")
    st.info("Please add PDF files to the backend/pdfs folder and restart the application")
    st.stop()

# Welcome message when chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("""
    ### 👋 Welcome to QSS Chatbot!

    I can help you with:
    - 📚 Questions about the PDF documents loaded in the system
    - 🔍 Finding specific information from your documents

    **Try asking:**
    - "What is the dress code policy?"
    - "Tell me about the leave policy"
    - "Where is the QSS office located?"
    """)

    st.divider()

    # Sample question buttons
    st.subheader(" Quick Start - Click to ask:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Dress Code"):
            st.session_state.sample_question = "What is the dress code policy?"

    with col2:
        if st.button("🏖️ Leave Policy"):
            st.session_state.sample_question = "What is the leave policy?"

    with col3:
        if st.button("🏢 Office Info"):
            st.session_state.sample_question = "Tell me about QSS office details"

    # Handle sample question clicks
    if "sample_question" in st.session_state and st.session_state.sample_question:
        prompt = st.session_state.sample_question
        st.session_state.sample_question = None

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response
        try:
            response = st.session_state.chatbot.ask(prompt)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"]
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Sources"):
                st.text(message["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chatbot.ask(prompt)
                answer = response["answer"]
                sources = response["sources"]

                # Display answer
                st.write(answer)

                # Display sources if available
                if sources:
                    with st.expander("📄 Sources"):
                        st.text(sources)

                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sidebar with enhanced controls
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot answers questions about PDFs in the backend/pdfs folder.")

    st.divider()

    # Model Information
    st.subheader("🤖 AI Model")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    st.info(f"**Provider:** {provider.upper()}\n\n**Model:** {model}")

    st.divider()

    # Chat Statistics
    st.subheader("📊 Statistics")
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions Asked", user_messages)
    st.metric("Total Messages", len(st.session_state.messages))

    st.divider()

    # PDF Files List
    st.subheader("📚 Loaded Documents")
    try:
        import glob
        pdf_files = glob.glob("backend/pdfs/*.pdf")
        if pdf_files:
            st.success(f"**{len(pdf_files)} PDFs loaded:**")
            for pdf in pdf_files:
                filename = pdf.split("\\")[-1].split("/")[-1]
                st.text(f"• {filename}")
        else:
            st.warning("No PDFs found")
    except Exception as e:
        st.error("Could not load PDF list")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chatbot.clear_memory()
        st.rerun()

    st.divider()

    # Display info
    st.caption("Powered by LangChain + Google Gemini")
    st.caption("Made with ❤️ by QSS")
