#python -m streamlit run day3_rag.py             
import os
import numpy as np
import streamlit as st

from pypdf import PdfReader
from google import genai


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SelfAI RAG",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "gemini-3-flash-preview"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


# --------------------------------------------------
# API key
# --------------------------------------------------

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GEMINI_API_KEY")


api_key = get_api_key()

if not api_key:
    st.error(
        "GEMINI_API_KEY not found. "
        "Add it to .streamlit/secrets.toml."
    )
    st.stop()


client = genai.Client(api_key=api_key)


# --------------------------------------------------
# Read PDF
# --------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF."""

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# --------------------------------------------------
# Split text into chunks
# --------------------------------------------------

def create_chunks(text):
    """Split document text into overlapping chunks."""

    text = text.replace("\n", " ")

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

def create_embedding(text):
    """Convert text into a numerical vector."""

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32,
    )


# --------------------------------------------------
# Build vector store
# --------------------------------------------------

def build_vector_store(chunks):

    vectors = []

    for chunk in chunks:

        vector = create_embedding(chunk)

        vectors.append(vector)

    return np.vstack(vectors)


# --------------------------------------------------
# Cosine similarity
# --------------------------------------------------

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------

def retrieve_chunks(question, chunks, vectors):

    question_vector = create_embedding(question)

    scores = []

    for index, vector in enumerate(vectors):

        score = cosine_similarity(
            question_vector,
            vector,
        )

        scores.append((score, index))

    scores.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    results = []

    for score, index in scores[:TOP_K]:

        results.append(
            {
                "text": chunks[index],
                "score": float(score),
            }
        )

    return results


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

def answer_question(question, retrieved_chunks):

    context = "\n\n".join(
        [
            f"DOCUMENT CHUNK {i + 1}:\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are SelfAI, a document-aware AI assistant.

Answer the user's question using the provided document context.

Rules:
- Use the document context as the primary source.
- Do not invent information.
- If the answer is not present in the document,
  clearly say that the document does not contain
  enough information.
- Give a concise and useful answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📚 SelfAI — Document RAG")

st.write(
    "Upload a document and ask questions about its contents."
)


# --------------------------------------------------
# Upload document
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # Extract text
    with st.spinner("Reading document..."):

        document_text = extract_text_from_pdf(
            uploaded_file
        )

    if not document_text.strip():

        st.error(
            "Could not extract text from this PDF."
        )

        st.stop()

    # Create chunks
    chunks = create_chunks(
        document_text
    )

    st.write(
        f"Created {len(chunks)} document chunks."
    )

    # Create embeddings
    with st.spinner(
        "Creating document embeddings..."
    ):

        vectors = build_vector_store(
            chunks
        )

    st.success(
        "Document indexed successfully!"
    )

    st.divider()

    # --------------------------------------------------
    # Question
    # --------------------------------------------------

    question = st.chat_input(
        "Ask something about your document..."
    )

    if question:

        st.chat_message(
            "user"
        ).markdown(question)

        # Retrieve
        with st.spinner(
            "Searching the document..."
        ):

            retrieved_chunks = retrieve_chunks(
                question,
                chunks,
                vectors,
            )

        # Generate
        with st.spinner(
            "Generating answer..."
        ):

            answer = answer_question(
                question,
                retrieved_chunks,
            )

        # Display answer
        with st.chat_message(
            "assistant"
        ):

            st.markdown(answer)

        # Show retrieved context
        with st.expander(
            "🔎 See retrieved document chunks"
        ):

            for i, item in enumerate(
                retrieved_chunks
            ):

                st.markdown(
                    f"### Chunk {i + 1}"
                )

                st.write(
                    f"Similarity: "
                    f"{item['score']:.4f}"
                )

                st.write(
                    item["text"]
                )