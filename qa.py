import requests
import json
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from backend import search

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "llama3
from groq import Groq
import os


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- PDF PROCESSING ---------------- #

def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def create_pdf_index(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

def process_pdf(uploaded_file):
    text = extract_pdf_text(uploaded_file)
    # chu.2:3b"nks = chunk_text(text)
    chunks = chunk_text(text)
    index, chunks = create_pdf_index(chunks)
    return index, chunks

def search_pdf(query, index, chunks, top_k=2):
    if index is None:
        return []

    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), top_k)
    return [chunks[i] for i in I[0]]

# ---------------- MAIN RESPONSE ---------------- #

# def get_ollama_response_stream(query, chat_history, pdf_index=None, pdf_chunks=None):

#     # 🔹 1. KIIT dataset context
#     db_results = search(query, top_k=2)
#     db_context = "\n".join(db_results)

#     # 🔹 2. PDF context
#     pdf_context = ""
#     if pdf_index is not None:
#         pdf_results = search_pdf(query, pdf_index, pdf_chunks)
#         pdf_context = "\n".join(pdf_results)

#     # 🔹 3. Chat memory (last 5 messages)
#     history_text = ""
#     for msg in chat_history[-5:]:
#         history_text += f"{msg['role']}: {msg['content']}\n"

#     # 🔹 4. Final prompt
#     prompt = f"""
# You are KIIT GPT, an intelligent assistant for KIIT University.

# Conversation History:
# {history_text}

# KIIT Database Context:
# {db_context}

# PDF Context:
# {pdf_context}

# User: {query}
# Assistant:
# """

#     payload = {
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": True
#     }

#     try:
#         response = requests.post(OLLAMA_URL, json=payload, stream=True)

#         for line in response.iter_lines():
#             if line:
#                 chunk = json.loads(line)
#                 yield chunk.get("response", "")
#                 if chunk.get("done"):
#                     break

#     except Exception as e:
#         yield f"Error: {str(e)}. Make sure Ollama is running."

def get_ollama_response_stream(query, chat_history, pdf_index=None, pdf_chunks=None):

    # KIIT dataset context
    db_results = search(query, top_k=2)
    db_context = "\n".join(db_results)

    # PDF context
    pdf_context = ""
    if pdf_index is not None:
        pdf_results = search_pdf(query, pdf_index, pdf_chunks)
        pdf_context = "\n".join(pdf_results)

    # Chat memory
    history_text = ""
    for msg in chat_history[-5:]:
        history_text += f"{msg['role']}: {msg['content']}\n"

    # Final prompt
    prompt = f"""
You are KIIT GPT, an intelligent assistant for KIIT University.

Conversation History:
{history_text}

KIIT Database Context:
{db_context}

PDF Context:
{pdf_context}

User: {query}
Assistant:
"""

    try:

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True
        )

        for chunk in response:
            content = chunk.choices[0].delta.content

            if content:
                yield content

    except Exception as e:
        yield f"Error: {str(e)}"
        
        
        