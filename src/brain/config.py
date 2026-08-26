"""All environment configuration, read once at import time."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_ROOT / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL") or OPENAI_BASE_URL
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY") or OPENAI_API_KEY

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "brain_chunks")

RAW_DIR = str(_ROOT / os.getenv("RAW_DIR", "data/raw"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def describe() -> str:
    return (
        f"chat model      {CHAT_MODEL}  ({OPENAI_BASE_URL})\n"
        f"embedding model {EMBEDDING_MODEL}  ({EMBEDDING_DIM} dims, {EMBEDDINGS_BASE_URL})\n"
        f"neo4j           {NEO4J_URI}  db={NEO4J_DATABASE}\n"
        f"qdrant          {QDRANT_URL}  collection={QDRANT_COLLECTION}\n"
        f"raw dir         {RAW_DIR}"
    )
