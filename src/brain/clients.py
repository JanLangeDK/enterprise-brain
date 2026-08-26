"""Neo4j driver, Qdrant client, OpenAI chat, embeddings — one singleton each."""

from __future__ import annotations

from functools import lru_cache

from neo4j import GraphDatabase
from openai import OpenAI
from qdrant_client import QdrantClient

from . import config


@lru_cache(maxsize=1)
def neo4j_driver() -> GraphDatabase.driver:
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


@lru_cache(maxsize=1)
def qdrant() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


@lru_cache(maxsize=1)
def _chat_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


@lru_cache(maxsize=1)
def _embeddings_client() -> OpenAI:
    return OpenAI(api_key=config.EMBEDDINGS_API_KEY, base_url=config.EMBEDDINGS_BASE_URL)


def complete(prompt: str, system: str | None = None, json_mode: bool = False) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    response = _chat_client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"} if json_mode else None,
    )
    return response.choices[0].message.content or ""


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _embeddings_client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]


def health() -> dict[str, str]:
    state: dict[str, str] = {}

    try:
        neo4j_driver().verify_connectivity()
        state["neo4j"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["neo4j"] = f"failed: {exc}"

    try:
        qdrant().get_collections()
        state["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["qdrant"] = f"failed: {exc}"

    try:
        embed_one("health check")
        state["embeddings"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["embeddings"] = f"failed: {exc}"

    try:
        complete("Reply with exactly: ok")
        state["chat"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["chat"] = f"failed: {exc}"

    return state
