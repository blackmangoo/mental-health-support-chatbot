"""
rag.py — Retrieval-Augmented Generation (RAG) Pipeline.

This module:
1. Loads the local knowledge base (data/knowledge_base.json)
2. Embeds all entries using sentence-transformers (runs locally, no API needed)
3. Builds a FAISS index for fast similarity search
4. At inference time: embeds the user query → retrieves top-K relevant entries

The retrieved context is then injected into the model's prompt to ground
responses in evidence-based wellness information.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import RAG_EMBEDDING_MODEL, RAG_TOP_K
import streamlit as st


KB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.json'))


@st.cache_resource(show_spinner=False)
def build_rag_index():
    """
    Load the knowledge base, embed all entries, and build a FAISS index.
    This runs once and is cached for the entire Streamlit session.
    
    Returns:
        tuple: (faiss_index, embedder, knowledge_base_list)
    """
    # Load knowledge base
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    # Load sentence transformer model (downloads once, then cached locally)
    embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)
    
    # Create embeddings for all knowledge base entries
    texts = [f"{entry['title']}: {entry['content']}" for entry in kb]
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    # Build FAISS index (Inner Product = cosine similarity when normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    
    return index, embedder, kb


def retrieve_context(query: str) -> tuple[list[dict], str]:
    """
    Retrieve the top-K most relevant knowledge base entries for a user query.
    
    Args:
        query: The user's message text.
        
    Returns:
        tuple:
            - list of matched knowledge base entry dicts
            - formatted context string ready to inject into the prompt
    """
    index, embedder, kb = build_rag_index()
    
    # Embed the query
    query_embedding = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    
    # Search FAISS index
    scores, indices = index.search(query_embedding, RAG_TOP_K)
    
    matched_entries = []
    context_parts = []
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if score > 0.3:  # Relevance threshold — only include if meaningfully similar
            entry = kb[idx]
            matched_entries.append(entry)
            context_parts.append(f"• {entry['title']}: {entry['content']}")
    
    context_str = "\n".join(context_parts) if context_parts else ""
    return matched_entries, context_str
