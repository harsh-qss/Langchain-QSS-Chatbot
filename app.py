"""
Simple PDF Chatbot - Streamlit Interface
PDFs are automatically loaded from backend/pdfs folder
"""

import streamlit as st
from chatbot import create_chatbot

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="💬",
    layout="centered"
)

# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    with st.spinner("Loading PDFs and initializing chatbot..."):
        try:
            st.session_state.chatbot = create_chatbot()
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()

# Title
st.title("💬 PDF Chatbot")

# Check if chatbot is ready
if not st.session_state.chatbot.is_ready():
    st.warning("⚠️ No PDF files found in backend/pdfs folder")
    st.info("Please add PDF files to the backend/pdfs folder and restart the application")
    st.stop()

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

# Sidebar with minimal controls
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot answers questions about PDFs in the backend/pdfs folder.")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chatbot.clear_memory()
        st.rerun()

    st.divider()

    # Display info
    st.caption("Powered by LangChain + Google Gemini")
