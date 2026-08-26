"""Three ways to get knowledge out, and one that combines them.

    vector_search      "what does this sound like"      Qdrant
    run_cypher         "what connects to what"          Neo4j
    hybrid_context     vector hit -> graph expansion    both

hybrid_context is the actual GraphRAG move: find chunks by meaning, follow the
MENTIONS edge into the graph, walk one or two hops, and hand the model both the
prose and the structure — with the source path on every piece so answers stay checkable.
"""

from __future__ import annotations

import re
from typing import Any

from . import config
from .clients import embed_one, neo4j_driver, qdrant

WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+\{[^}]*\bCREATE)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Vector side
# --------------------------------------------------------------------------
def vector_search(query: str, k: int = 6, source_system: str | None = None) -> list[dict[str, Any]]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    query_filter = None
    if source_system:
        query_filter = Filter(
            must=[FieldCondition(key="source_system", match=MatchValue(value=source_system))]
        )

    hits = qdrant().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embed_one(query),
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    return [
        {
            "chunk_id": hit.payload["chunk_id"],
            "score": round(hit.score, 4),
            "path": hit.payload["path"],
            "title": hit.payload["title"],
            "heading": hit.payload["heading"],
            "source_system": hit.payload.get("source_system", "unknown"),
            "text": hit.payload["text"],
        }
        for hit in hits
    ]


# --------------------------------------------------------------------------
# Graph side
# --------------------------------------------------------------------------
def run_cypher(cypher: str, params: dict | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Read-only Cypher. Writes are refused rather than sanitised."""
    if WRITE_KEYWORDS.search(cypher):
        raise ValueError("This helper is read only. Remove write clauses from the query.")
    with neo4j_driver().session(database=config.NEO4J_DATABASE) as session:
        result = session.run(cypher, **(params or {}))
        return [record.data() for record in result][:limit]


def graph_schema() -> str:
    """What labels and relationship types the open schema extraction actually produced."""
    labels = run_cypher(
        "MATCH (e:Entity) UNWIND labels(e) AS label "
        "WITH label WHERE label <> 'Entity' "
        "RETURN label, count(*) AS n ORDER BY n DESC"
    )
    rels = run_cypher(
        "MATCH (:Entity)-[r]->(:Entity) RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"
    )
    return (
        "Entity labels: "
        + ", ".join(f"{row['label']} ({row['n']})" for row in labels)
        + "\nRelationship types: "
        + ", ".join(f"{row['type']} ({row['n']})" for row in rels)
        + "\nAlso present: (:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)"
        + "\nEntity properties: key (normalised), name, type, description"
    )


FIND_ENTITY = """
CALL db.index.fulltext.queryNodes('entity_name', $term) YIELD node, score
RETURN node.name AS name, node.type AS type, node.description AS description, score
ORDER BY score DESC LIMIT $limit
"""


def find_entity(term: str, limit: int = 5) -> list[dict[str, Any]]:
    return run_cypher(FIND_ENTITY, {"term": term, "limit": limit})


NEIGHBOURHOOD = """
MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name)
MATCH path = (e)-[r*1..%d]-(other:Entity)
UNWIND relationships(path) AS edge
WITH DISTINCT startNode(edge) AS a, edge, endNode(edge) AS b
RETURN a.name AS source, a.type AS source_type, type(edge) AS relationship,
       b.name AS target, b.type AS target_type,
       edge.evidence AS evidence, edge.chunk_id AS chunk_id
LIMIT $limit
"""


def neighbourhood(name: str, hops: int = 2, limit: int = 40) -> list[dict[str, Any]]:
    hops = max(1, min(hops, 3))
    return run_cypher(NEIGHBOURHOOD % hops, {"name": name, "limit": limit}, limit=limit)


# --------------------------------------------------------------------------
# The join: vector hits -> graph expansion
# --------------------------------------------------------------------------
EXPAND_FROM_CHUNKS = """
MATCH (c:Chunk)-[:MENTIONS]->(seed:Entity)
WHERE c.id IN $chunk_ids
WITH collect(DISTINCT seed) AS seeds
UNWIND seeds AS seed
MATCH path = (seed)-[r*1..%d]-(other:Entity)
UNWIND relationships(path) AS edge
WITH DISTINCT startNode(edge) AS a, edge, endNode(edge) AS b
MATCH (evidence_chunk:Chunk {id: edge.chunk_id})<-[:HAS_CHUNK]-(d:Document)
RETURN a.name AS source, a.type AS source_type, type(edge) AS relationship,
       b.name AS target, b.type AS target_type,
       edge.evidence AS evidence, d.path AS path
LIMIT $limit
"""


def hybrid_context(question: str, k: int = 5, hops: int = 2, edge_limit: int = 35) -> dict[str, Any]:
    """Return the passages, the facts, and a ready made context string."""
    chunks = vector_search(question, k=k)
    chunk_ids = [c["chunk_id"] for c in chunks]

    facts: list[dict[str, Any]] = []
    if chunk_ids:
        facts = run_cypher(
            EXPAND_FROM_CHUNKS % max(1, min(hops, 3)),
            {"chunk_ids": chunk_ids, "limit": edge_limit},
            limit=edge_limit,
        )

    passage_block = "\n\n".join(
        f"[passage {i + 1}] source: {c['path']} ({c['source_system']}) — {c['heading']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    fact_block = "\n".join(
        f"- ({f['source_type']}) {f['source']} -[{f['relationship']}]-> ({f['target_type']}) {f['target']}"
        f"   | evidence: {f['evidence']} | source: {f['path']}"
        for f in facts
    )

    context = (
        "GRAPH FACTS (structure, from Neo4j)\n"
        + (fact_block or "none")
        + "\n\nPASSAGES (prose, from Qdrant)\n"
        + (passage_block or "none")
    )
    return {"chunks": chunks, "facts": facts, "context": context}
