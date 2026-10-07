# Document Intelligence MVP

A deliberately small project demonstrating **Retrieval-Augmented Generation (RAG), Generative AI, semantic search, and a lightweight AI-agent decision layer**.

The project was developed progressively in three stages:

- **Day 1:** Streamlit UI and chat interface
- **Day 2:** Gemini-powered conversational AI
- **Day 3:** Document-based RAG using Gemini embeddings and cosine similarity

---

## Project Overview

The application allows users to upload documents and ask questions about their content.

The RAG pipeline works as follows:

```text
Document
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embedding Generation
   ↓
Vector Representation
   ↓
Cosine Similarity Retrieval
   ↓
Top-K Relevant Chunks
   ↓
Gemini LLM
   ↓
Grounded Answer