import streamlit as st
import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"

# Initialize embedder
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set Tesseract path (adjust if necessary)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def initialize_ollama():
    try:
        requests.get("http://localhost:11434").raise_for_status()
    except requests.RequestException:
        st.error("Please start Ollama and pull the model first.")
        st.stop()


def generate_response(prompt, context=None):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "context": context,
        "options": {"temperature": 0.7, "top_p": 0.95, "num_predict": 512}
    }
    res = requests.post(OLLAMA_API_URL, json=payload)
    res.raise_for_status()
    data = res.json()
    return data["response"], data.get("context")


def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []

    for page in doc:
        # Extract text using PyMuPDF
        text = page.get_text().strip()

        # If text is minimal, use OCR
        if len(text) < 50:
            # Render page at high DPI for better OCR results
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(img)

        full_text.append(text)

    return "\n".join(full_text)


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks):
    return embed_model.encode(chunks, convert_to_tensor=True)


def retrieve_context(question, chunks, embeddings, top_k=3):
    if embeddings is None or not chunks:
        return ""
    chunk_np = embeddings.cpu().numpy()
    if chunk_np.ndim == 1:
        chunk_np = chunk_np.reshape(1, -1)
    q_np = np.array(embed_model.encode([question], convert_to_tensor=False))
    if q_np.ndim == 1:
        q_np = q_np.reshape(1, -1)
    sims = cosine_similarity(q_np, chunk_np)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]
    return "\n\n".join(chunks[i] for i in top_idxs)


# Initialize Ollama
initialize_ollama()

st.title("💎 Chat with Any PDF (Gemma 3B + Streamlit)")
st.markdown("Upload any PDF (text or image-based) to begin.")

# Session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# File uploader
pdf_file = st.file_uploader("Upload your PDF", type="pdf")
if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    st.session_state.chunks = chunk_text(text)
    st.session_state.embeddings = embed_chunks(st.session_state.chunks)
    st.success("Document processed! Ask away.")

# Chat form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:")
    submitted = st.form_submit_button("Send")

if submitted:
    if not st.session_state.chunks:
        st.warning("Upload and process a PDF before asking questions.")
    else:
        st.session_state.conversation.append(
            {"role": "user", "content": user_input})
        context = retrieve_context(
            user_input,
            st.session_state.chunks,
            st.session_state.embeddings
        )
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        response, _ = generate_response(prompt, None)
        st.session_state.conversation.append(
            {"role": "assistant", "content": response})

# Display conversation
for msg in st.session_state.conversation:
    sender = "You" if msg["role"] == "user" else "Gemma"
    st.markdown(f"**{sender}:** {msg['content']}")
