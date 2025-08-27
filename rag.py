import os
from typing import List, Dict, Any
from pathlib import Path
import json
import re

import chromadb
from chromadb.config import Settings
from openai import OpenAI

DATA_PATH = Path(__file__).parent / "data" / "book_summaries.json"
DB_DIR = Path(__file__).parent / ".chroma"
# ---Tuning knobs--- for RAG system
# These can be adjusted based on performance and retrieval quality
# MIN_SIMILARITY: Minimum cosine similarity for a hit to be considered relevant
# MIN_KEYWORD_OVERLAP: Minimum overlap of keywords between query and document themes
# MAX_CANDIDATES: Maximum number of candidates to return from the search
MIN_SIMILARITY = 0.20
MIN_KEYWORD_OVERLAP = 0.0
MAX_CANDIDATES = 15

_WORD_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)
STOP = {"and","the","for","with","from","that","este","sunt","care","din","și","sau","în","pe","după","despre"}

def _keywords(s:str):
    toks = [t.lower() for t in _WORD_RE.findall(s)]
    return {t for t in toks if t not in STOP}

def _overlap_fraction(q:str, d:str) -> float:
    qk = _keywords(q)
    if not qk: return 0.0
    dk = _keywords(d)
    return len(qk & dk) / max(1,len(qk))

def get_client() -> OpenAI:
    # Needs OPENAI_API_KEY in env
    return OpenAI()

def load_items() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_collection(collection_name: str = "book_summaries") -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    try:
        col = client.get_collection(collection_name)
    except Exception:
        col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    return col

def embed_texts(texts: List[str], client: OpenAI) -> List[List[float]]:
    # Batches are fine for small demo sizes
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

def index_if_empty():
    items = load_items()
    col = ensure_collection()
    if col.count() > 0:
        return
    client = get_client()
    docs = []
    metadatas = []
    ids = []
    for i, it in enumerate(items):
        # Concatenate title + themes + short summary -> good retrieval
        themes = ", ".join(it.get("themes", []))  # Convert list to comma-separated string
        doc = f"Title: {it['title']}\nThemes: {themes}\nSummary: {it['summary_short']}"
        docs.append(doc)
        metadatas.append({"title": it["title"], "themes": themes})  # Store themes as a string
        ids.append(f"book-{i}")
    embeddings = embed_texts(docs, client)
    col.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

def extract_keywords(text: str) -> set:
    """Extract keywords from the text by removing stopwords and normalizing."""
    stopwords = {"vreau", "o", "carte", "despre", "si", "in", "pe", "cu", "la", "de", "un", "o", "este", "sunt"}
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return set(words) - stopwords

def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Enhanced search function:
    - Prioritize matches based on themes and keywords.
    - Return top-k, sorted by relevance.
    """
    col = ensure_collection()
    client = get_client()

    # Embed the query with your current OpenAI path
    q_emb = embed_texts([query], client)[0]

    # Ask Chroma for more than k so we can filter and still keep enough
    res = col.query(
        query_embeddings=[q_emb],
        n_results=max(k, MAX_CANDIDATES),
        include=["documents", "metadatas", "distances"],  # distances needed for similarity gate
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # Convert Chroma cosine *distance* -> cosine *similarity*
        try:
            sim = 1.0 - float(dist) if dist is not None else 0.0
        except Exception:
            sim = 0.0

        # Extract themes and keywords from metadata
        themes = set(meta.get("themes", "").split(", "))
        query_keywords = extract_keywords(query)

        # Compute overlap with themes and document content
        overlap = len(query_keywords & themes)

        if sim >= MIN_SIMILARITY or overlap > 0:  # Prioritize theme overlap
            hits.append({
                "similarity": sim,
                "overlap": overlap,
                "document": doc,
                "metadata": meta,  # keep original shape; UI can still read ["title"]
            })

    # Sort best-first by overlap and similarity, then keep top-k
    hits.sort(key=lambda h: (h["overlap"], h["similarity"]), reverse=True)
    return hits[:k]

def generate_response_with_rag(query: str, k: int = 3) -> str:
    """Integrate retrieval and generation to create a RAG engine."""
    # Step 1: Retrieve relevant documents
    hits = search(query, k=k)
    if not hits:
        return "I'm sorry, I couldn't find any relevant information."

    # Step 2: Combine retrieved documents into a context
    context = "\n\n".join([hit["document"] for hit in hits])

    # Step 3: Use OpenAI GPT model to generate a response
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information based on the following context."},
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content