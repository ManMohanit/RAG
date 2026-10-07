# ============================================================
# Day 4: Advanced RAG
# Phase 1  -> Source/Page Citations
# Phase 2  -> Persistent ChromaDB Vector Database
# Phase 3  -> Hybrid Retrieval: TF-IDF + Gemini Embeddings
# ============================================================

import os
import hashlib

import numpy as np
import streamlit as st
import chromadb

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SelfAI — Advanced RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "gemini-3-flash-preview"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4
RETRIEVAL_CANDIDATES = 8

# Weight given to semantic embedding retrieval.
# 0.7 means:
#   70% semantic similarity
#   30% lexical TF-IDF similarity
SEMANTIC_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "selfai_documents"


# ============================================================
# API Key
# ============================================================

def get_api_key():
    """Read Gemini API key from Streamlit secrets or environment."""

    try:
        return st.secrets["GEMINI_API_KEY"]

    except (KeyError, FileNotFoundError):
        return os.getenv("GEMINI_API_KEY")


api_key = get_api_key()

if not api_key:
    st.error(
        "GEMINI_API_KEY not found. "
        "Add it to .streamlit/secrets.toml "
        "or set it as an environment variable."
    )
    st.stop()


# ============================================================
# Gemini Client
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# Persistent ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


# ============================================================
# Data Preparation
# ============================================================

def create_chunks(text, source, page_number):
    """
    Split text into overlapping chunks while preserving
    source and page metadata.
    """

    text = text.replace("\n", " ").strip()

    chunks = []

    start = 0

    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "page": page_number,
                }
            )

        start += step

    return chunks


# ============================================================
# PDF Processing
# ============================================================

def extract_pdf_chunks(uploaded_file):
    """
    Extract text page-by-page from a PDF and create chunks.

    Each chunk keeps:
        - source filename
        - page number
        - chunk index
    """

    reader = PdfReader(uploaded_file)

    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        page_chunks = create_chunks(
            text=text,
            source=uploaded_file.name,
            page_number=page_number,
        )

        all_chunks.extend(page_chunks)

    return all_chunks


# ============================================================
# Embeddings
# ============================================================

def create_embedding(text):
    """
    Convert text into a numerical vector using Gemini embeddings.
    """

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32,
    )


# ============================================================
# Persistent Vector Database
# ============================================================

def create_chunk_id(chunk):
    """
    Create a deterministic ID for a chunk.

    The same document + page + chunk text will generate
    the same ID, preventing unnecessary duplicate storage.
    """

    raw_id = (
        f"{chunk['source']}"
        f"|{chunk['page']}"
        f"|{chunk['text']}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def store_chunks_in_chroma(chunks):
    """
    Generate embeddings and store chunks in persistent ChromaDB.

    Existing chunks are skipped using deterministic IDs.
    """

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in chunks:

        chunk_id = create_chunk_id(chunk)

        # Check whether this chunk already exists.
        existing = collection.get(
            ids=[chunk_id],
            include=[]
        )

        if existing["ids"]:
            continue

        vector = create_embedding(
            chunk["text"]
        )

        ids.append(chunk_id)

        documents.append(
            chunk["text"]
        )

        metadatas.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
            }
        )

        embeddings.append(
            vector.tolist()
        )

    if not ids:
        return 0

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(ids)


# ============================================================
# Get All Stored Documents
# ============================================================

def get_all_documents():
    """
    Retrieve all stored document chunks from ChromaDB.

    Used for the TF-IDF lexical retrieval branch.
    """

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []

    results = []

    for i in range(len(documents)):

        results.append(
            {
                "id": ids[i],
                "text": documents[i],
                "source": metadatas[i]["source"],
                "page": metadatas[i]["page"],
            }
        )

    return results


# ============================================================
# TF-IDF Retrieval
# ============================================================

def lexical_retrieval(question, documents):
    """
    Retrieve documents using TF-IDF + cosine similarity.

    This is the lexical retrieval branch of hybrid search.
    """

    if not documents:
        return {}

    corpus = [
        item["text"]
        for item in documents
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        corpus + [question]
    )

    question_vector = matrix[-1]

    document_vectors = matrix[:-1]

    scores = cosine_similarity(
        question_vector,
        document_vectors
    ).flatten()

    ranked_indices = np.argsort(
        scores
    )[::-1][:RETRIEVAL_CANDIDATES]

    results = {}

    for index in ranked_indices:

        score = float(scores[index])

        if score <= 0:
            continue

        item = documents[index]

        results[item["id"]] = {
            "id": item["id"],
            "text": item["text"],
            "source": item["source"],
            "page": item["page"],
            "lexical_score": score,
            "semantic_score": 0.0,
        }

    return results


# ============================================================
# Semantic Retrieval
# ============================================================

def semantic_retrieval(question):
    """
    Retrieve documents using Gemini embeddings + ChromaDB.
    """

    question_vector = create_embedding(
        question
    )

    result = collection.query(
        query_embeddings=[
            question_vector.tolist()
        ],
        n_results=min(
            RETRIEVAL_CANDIDATES,
            collection.count()
        ),
    )

    results = {}

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for i in range(len(ids)):

        chunk_id = ids[i]

        distance = float(
            distances[i]
        )

        # Chroma cosine distance:
        # smaller distance = greater similarity.
        semantic_score = 1.0 - distance

        results[chunk_id] = {
            "id": chunk_id,
            "text": documents[i],
            "source": metadatas[i]["source"],
            "page": metadatas[i]["page"],
            "lexical_score": 0.0,
            "semantic_score": semantic_score,
        }

    return results


# ============================================================
# Score Normalization
# ============================================================

def normalize_scores(results, score_key):
    """
    Normalize scores between 0 and 1 for combining
    lexical and semantic retrieval scores.
    """

    if not results:
        return

    scores = [
        item[score_key]
        for item in results.values()
    ]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:

        for item in results.values():
            item[score_key] = 1.0

        return

    for item in results.values():

        item[score_key] = (
            (item[score_key] - min_score)
            / (max_score - min_score)
        )


# ============================================================
# Hybrid Retrieval
# ============================================================

def hybrid_retrieval(question):
    """
    Combine lexical TF-IDF retrieval and semantic embedding
    retrieval.

    Final score:

        semantic_weight * semantic_score
        +
        lexical_weight * lexical_score
    """

    documents = get_all_documents()

    if not documents:
        return []

    lexical_results = lexical_retrieval(
        question,
        documents
    )

    semantic_results = semantic_retrieval(
        question
    )

    # Combine candidates from both retrieval methods.
    combined = {}

    for chunk_id, item in lexical_results.items():

        combined[chunk_id] = item

    for chunk_id, item in semantic_results.items():

        if chunk_id not in combined:

            combined[chunk_id] = item

        else:

            combined[chunk_id]["semantic_score"] = (
                item["semantic_score"]
            )

    # Normalize both score types before combining.
    normalize_scores(
        combined,
        "lexical_score"
    )

    normalize_scores(
        combined,
        "semantic_score"
    )

    # Calculate final hybrid score.
    for item in combined.values():

        item["hybrid_score"] = (
            SEMANTIC_WEIGHT
            * item["semantic_score"]
            +
            LEXICAL_WEIGHT
            * item["lexical_score"]
        )

    # Rank by final hybrid score.
    ranked = sorted(
        combined.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return ranked[:TOP_K]


# ============================================================
# Generate Answer
# ============================================================

def answer_question(
    question,
    retrieved_chunks,
):
    """
    Generate an answer using only the retrieved document context.
    """

    if not retrieved_chunks:

        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    context_parts = []

    for index, item in enumerate(
        retrieved_chunks,
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE [{index}]
File: {item['source']}
Page: {item['page']}

Content:
{item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are SelfAI, a document-aware AI assistant.

Answer the user's question using ONLY the retrieved
document context provided below.

Rules:
- Use the retrieved document context as the primary source.
- Do not invent information.
- If the answer is not present in the retrieved context,
  clearly say that the documents do not contain enough information.
- Keep the answer concise and useful.
- Cite supporting sources using [1], [2], etc.
- Do not create citations that are not present in the context.

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


# ============================================================
# Streamlit UI
# ============================================================

st.title("📚 SelfAI — Advanced RAG")

st.caption(
    "Source citations + Persistent ChromaDB + Hybrid Retrieval"
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write(
        f"Embedding Model: `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"Generation Model: `{MODEL_NAME}`"
    )

    st.write(
        f"Chunk Size: `{CHUNK_SIZE}`"
    )

    st.write(
        f"Chunk Overlap: `{CHUNK_OVERLAP}`"
    )

    st.write(
        f"Top-K: `{TOP_K}`"
    )

    st.write(
        f"Semantic Weight: `{SEMANTIC_WEIGHT}`"
    )

    st.write(
        f"Lexical Weight: `{LEXICAL_WEIGHT}`"
    )

    st.divider()

    st.subheader("Persistent Database")

    st.write(
        f"Stored chunks: `{collection.count()}`"
    )

    st.write(
        f"Database: `ChromaDB`"
    )


# ============================================================
# Upload PDF
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


# ============================================================
# Process Document
# ============================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    with st.spinner(
        "Extracting text and creating chunks..."
    ):

        chunks = extract_pdf_chunks(
            uploaded_file
        )

    if not chunks:

        st.error(
            "No text could be extracted from this PDF."
        )

        st.stop()

    st.info(
        f"Created {len(chunks)} chunks "
        f"from the uploaded document."
    )

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    with st.spinner(
        "Creating embeddings and updating ChromaDB..."
    ):

        new_chunks = store_chunks_in_chroma(
            chunks
        )

    st.success(
        f"Database updated. "
        f"{new_chunks} new chunks added."
    )

    st.write(
        f"Total persistent chunks: "
        f"**{collection.count()}**"
    )

    st.divider()

    # ========================================================
    # Question
    # ========================================================

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:

        # ----------------------------------------------------
        # Display User Question
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------------------------
        # Hybrid Retrieval
        # ----------------------------------------------------

        with st.spinner(
            "Running hybrid retrieval..."
        ):

            retrieved_chunks = hybrid_retrieval(
                question
            )

        # ----------------------------------------------------
        # Generate Answer
        # ----------------------------------------------------

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = answer_question(
                question,
                retrieved_chunks,
            )

        # ----------------------------------------------------
        # Display Answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)

        # ----------------------------------------------------
        # Source Citations
        # ----------------------------------------------------

        st.subheader(
            "📚 Retrieved Sources"
        )

        if retrieved_chunks:

            for index, item in enumerate(
                retrieved_chunks,
                start=1,
            ):

                with st.expander(
                    f"[{index}] "
                    f"{item['source']} "
                    f"— Page {item['page']}"
                ):

                    st.write(
                        f"**Hybrid Score:** "
                        f"{item['hybrid_score']:.4f}"
                    )

                    st.write(
                        f"**Semantic Score:** "
                        f"{item['semantic_score']:.4f}"
                    )

                    st.write(
                        f"**TF-IDF Score:** "
                        f"{item['lexical_score']:.4f}"
                    )

                    st.write(
                        item["text"]
                    )

        else:

            st.warning(
                "No relevant document chunks were found."
            )


# ============================================================
# Explanation
# ============================================================

else:

    st.info(
        "Upload a PDF to start the RAG pipeline."
    )

#     st.markdown(
#         """
# ### Day 4 Architecture

# ```text
# PDF
#  ↓
# Text Extraction + Page Metadata
#  ↓
# Chunking
#  ↓
# Gemini Embeddings
#  ↓
# Persistent ChromaDB
#  ↓
#         User Question
#               ↓
#        ┌──────┴──────┐
#        ↓             ↓
#      TF-IDF      Embeddings
#        ↓             ↓
#        └──────┬──────┘
#               ↓
#        Hybrid Retrieval
#               ↓
#           Top-K Chunks
#               ↓
#         Gemini 3 Flash
#               ↓
#       Grounded Answer
#               ↓
#        Source Citations
# """
#     )