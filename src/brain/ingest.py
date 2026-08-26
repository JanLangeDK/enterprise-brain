"""Ingestion pipeline: raw markdown -> chunks -> Qdrant vectors + Neo4j graph.

Two things come out of every chunk:

1. a vector in Qdrant, for "what does this sound like" retrieval
2. entities and relationships in Neo4j, for "how does this connect" retrieval

The two are joined by the chunk id, which is stored on both sides. That join is
what makes the hybrid GraphRAG retrieval in retrieval.py possible.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config
from .clients import complete, embed_texts, neo4j_driver, qdrant

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


# --------------------------------------------------------------------------
# 1. Load markdown
# --------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: str
    path: str
    title: str
    meta: dict
    body: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    heading: str
    text: str
    entities: list[str] = field(default_factory=list)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Minimal `key: value` frontmatter parser. No YAML dependency."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta: dict[str, str] = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw[end + 4 :].lstrip("\n")


def load_documents(raw_dir: Path | None = None) -> list[Document]:
    raw_dir = Path(raw_dir or config.RAW_DIR)
    documents: list[Document] = []
    for path in sorted(raw_dir.rglob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        heading = re.search(r"^#\s+(.+)$", body, flags=re.M)
        documents.append(
            Document(
                doc_id=path.stem,
                path=str(path.relative_to(raw_dir)),
                title=meta.get("title") or (heading.group(1) if heading else path.stem),
                meta=meta,
                body=body,
            )
        )
    return documents


# --------------------------------------------------------------------------
# 2. Chunk
# --------------------------------------------------------------------------
def chunk_document(doc: Document) -> list[Chunk]:
    """Split on `##` headings first, then hard split anything still too long."""
    sections: list[tuple[str, str]] = []
    current_heading, buffer = doc.title, []
    for line in doc.body.splitlines():
        if line.startswith("## "):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
            current_heading, buffer = line[3:].strip(), []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    for heading, text in sections:
        if not text:
            continue
        for piece in _split_long(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{index}",
                    doc_id=doc.doc_id,
                    index=index,
                    heading=heading,
                    # keep the document title in the embedded text: it carries the entity name
                    text=f"{doc.title} — {heading}\n\n{piece}",
                )
            )
    return chunks


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return pieces


# --------------------------------------------------------------------------
# 3. Vectors in Qdrant
# --------------------------------------------------------------------------
def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def ensure_collection(reset: bool = False) -> None:
    client = qdrant()
    exists = client.collection_exists(config.QDRANT_COLLECTION)
    if exists and reset:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )


def index_chunks(chunks: list[Chunk], docs_by_id: dict[str, Document], batch: int = 64) -> int:
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        vectors = embed_texts([c.text for c in window])
        qdrant().upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id(c.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "path": docs_by_id[c.doc_id].path,
                        "title": docs_by_id[c.doc_id].title,
                        "source_system": docs_by_id[c.doc_id].meta.get("source_system", "unknown"),
                        "heading": c.heading,
                        "text": c.text,
                    },
                )
                for c, vector in zip(window, vectors)
            ],
        )
    return len(chunks)


# --------------------------------------------------------------------------
# 4. Open schema extraction
# --------------------------------------------------------------------------
EXTRACTION_SYSTEM = """You build a knowledge graph for a company's internal brain.

You decide the schema yourself. Do not force the text into a fixed ontology, but stay
consistent: reuse a type you have already used when it fits, keep types short, singular
and in PascalCase (Company, Person, Site, Equipment, Contract, Opportunity, Issue, Event,
Competitor, System, Amount...). Relationship types are SCREAMING_SNAKE_CASE verbs
(WORKS_FOR, LOCATED_AT, COVERED_BY, COMPETES_FOR, REPORTED_BY, DEPENDS_ON...).

Rules:
- Only extract what the text actually states. No world knowledge, no guesses.
- Use the entity's full name as it appears in the text ("Nordvind Energi A/S", not "the customer").
- Every relationship must reference entity names you also return in "entities".
- "evidence" is a short quote or close paraphrase of the sentence that supports the edge.

Return JSON only, shaped exactly like:
{"entities":[{"name":"...","type":"...","description":"..."}],
 "relationships":[{"source":"...","target":"...","type":"...","evidence":"..."}]}"""


def extract_from_chunk(chunk: Chunk, doc: Document) -> dict:
    prompt = (
        f"Document: {doc.title}\n"
        f"Source system: {doc.meta.get('source_system', 'unknown')}\n"
        f"Section: {chunk.heading}\n\n"
        f"Text:\n{chunk.text}"
    )
    raw = complete(prompt, system=EXTRACTION_SYSTEM, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0)) if match else {"entities": [], "relationships": []}
    data.setdefault("entities", [])
    data.setdefault("relationships", [])
    return data


def normalise_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def clean_label(value: str, default: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]", " ", value or "").title().replace(" ", "")
    return value or default


def clean_rel_type(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", (value or "RELATED_TO").upper()).strip("_")
    return value or "RELATED_TO"


# --------------------------------------------------------------------------
# 5. Write to Neo4j
# --------------------------------------------------------------------------
SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
]


def ensure_schema() -> None:
    with neo4j_driver().session(database=config.NEO4J_DATABASE) as session:
        for statement in SCHEMA_STATEMENTS:
            session.run(statement)


def reset_graph() -> None:
    with neo4j_driver().session(database=config.NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")


WRITE_DOC = """
MERGE (d:Document {id: $doc_id})
SET d.title = $title, d.path = $path, d.source_system = $source_system, d.updated = $updated
WITH d
UNWIND $chunks AS chunk
MERGE (c:Chunk {id: chunk.chunk_id})
SET c.text = chunk.text, c.heading = chunk.heading, c.index = chunk.index, c.doc_id = $doc_id
MERGE (d)-[:HAS_CHUNK]->(c)
"""

WRITE_ENTITIES = """
UNWIND $entities AS entity
MERGE (e:Entity {key: entity.key})
ON CREATE SET e.name = entity.name, e.description = entity.description, e.type = entity.label
ON MATCH SET e.name = entity.name
WITH e, entity
CALL apoc.create.addLabels(e, [entity.label]) YIELD node
WITH node AS e, entity
MATCH (c:Chunk {id: $chunk_id})
MERGE (c)-[m:MENTIONS]->(e)
SET m.name_in_text = entity.name
"""

WRITE_RELATIONSHIPS = """
UNWIND $relationships AS rel
MATCH (a:Entity {key: rel.source_key})
MATCH (b:Entity {key: rel.target_key})
CALL apoc.merge.relationship(a, rel.type, {}, {evidence: rel.evidence, chunk_id: $chunk_id}, b, {})
YIELD rel AS r
RETURN count(r) AS written
"""


def write_chunks(doc: Document, chunks: list[Chunk]) -> None:
    with neo4j_driver().session(database=config.NEO4J_DATABASE) as session:
        session.run(
            WRITE_DOC,
            doc_id=doc.doc_id,
            title=doc.title,
            path=doc.path,
            source_system=doc.meta.get("source_system", "unknown"),
            updated=doc.meta.get("updated") or doc.meta.get("date", ""),
            chunks=[
                {"chunk_id": c.chunk_id, "text": c.text, "heading": c.heading, "index": c.index}
                for c in chunks
            ],
        )


def write_extraction(chunk: Chunk, extraction: dict) -> tuple[int, int]:
    entities = []
    seen: set[str] = set()
    for item in extraction.get("entities", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        key = normalise_key(name)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "key": key,
                "name": name,
                "label": clean_label(item.get("type", ""), "Thing"),
                "description": (item.get("description") or "")[:600],
            }
        )

    relationships = []
    for item in extraction.get("relationships", []):
        source_key = normalise_key(item.get("source", ""))
        target_key = normalise_key(item.get("target", ""))
        if not source_key or not target_key or source_key == target_key:
            continue
        if source_key not in seen or target_key not in seen:
            continue  # only edges between entities we actually created
        relationships.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "type": clean_rel_type(item.get("type", "")),
                "evidence": (item.get("evidence") or "")[:600],
            }
        )

    with neo4j_driver().session(database=config.NEO4J_DATABASE) as session:
        if entities:
            session.run(WRITE_ENTITIES, entities=entities, chunk_id=chunk.chunk_id)
        if relationships:
            session.run(WRITE_RELATIONSHIPS, relationships=relationships, chunk_id=chunk.chunk_id)

    chunk.entities = [e["name"] for e in entities]
    return len(entities), len(relationships)


# --------------------------------------------------------------------------
# 6. One call to run the lot
# --------------------------------------------------------------------------
def ingest_all(reset: bool = False, verbose: bool = True) -> dict:
    docs = load_documents()
    docs_by_id = {d.doc_id: d for d in docs}

    ensure_collection(reset=reset)
    ensure_schema()
    if reset:
        reset_graph()
        ensure_schema()

    stats = {"documents": len(docs), "chunks": 0, "entities": 0, "relationships": 0}

    for doc in docs:
        chunks = chunk_document(doc)
        write_chunks(doc, chunks)
        index_chunks(chunks, docs_by_id)
        stats["chunks"] += len(chunks)

        for chunk in chunks:
            extraction = extract_from_chunk(chunk, doc)
            entity_count, rel_count = write_extraction(chunk, extraction)
            stats["entities"] += entity_count
            stats["relationships"] += rel_count

        if verbose:
            print(f"  {doc.path:38s} {len(chunks):2d} chunks")

    return stats
