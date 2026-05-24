import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from qa import get_ollama_response_stream, process_pdf

st.set_page_config(page_title="KIIT GPT", layout="wide")

st.title("🤖 KIIT GPT")
st.caption("Ask anything about KIIT University - Local Privacy Mode")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
    st.session_state.pdf_chunks = None

# Sidebar
with st.sidebar:
    st.header("📂 Context")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        st.success("Processing PDF...")
        index, chunks = process_pdf(uploaded_file)
        st.session_state.pdf_index = index
        st.session_state.pdf_chunks = chunks
        st.success("PDF ready for questions!")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = st.write_stream(
                get_ollama_response_stream(
                    prompt,
                    st.session_state.messages,
                    st.session_state.pdf_index,
                    st.session_state.pdf_chunks
                )
            )

    st.session_state.messages.append({"role": "assistant", "content": full_response})