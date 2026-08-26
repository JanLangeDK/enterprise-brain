"""The agent: Microsoft Agent Framework ChatAgent, pointed at OpenAI,
with four tools over the brain.

The tools deliberately mirror the retrieval module, so the model can choose:
descriptive question -> vectors, structural question -> Cypher, most real
questions -> the hybrid tool.
"""

from __future__ import annotations

import json
from typing import Annotated

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

from . import config, retrieval

INSTRUCTIONS = """You are the enterprise brain. You answer questions about customers,
sites, equipment, contracts, people and opportunities using only the tools.

How to choose a tool:
- ask_the_brain: your default. Combines meaning based passage retrieval with graph expansion.
- search_documents: when the user wants wording, quotes or "what does the note say".
- query_graph: when the question is structural, counting, or spans entities that no single
  document mentions together ("which customers have X", "who signed what", "what is affected if Y").
  Call describe_graph first if you are unsure which labels or relationship types exist.
- describe_graph: the current labels, relationship types and properties.

Rules:
- Never invent facts. If the tools return nothing, say so and say what you searched for.
- Cite the source file path for every claim, like [visit-note-2026-08-14.md].
- Prefer specific answers over hedged ones. Short paragraphs or tight bullets.
- When facts from different sources disagree, say which sources disagree and how.
- Write Cypher against the schema from describe_graph. Read only: no CREATE, MERGE, SET or DELETE."""


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def search_documents(
    query: Annotated[str, Field(description="What to look for, in natural language.")],
    k: Annotated[int, Field(description="How many passages to return, 1-10.")] = 5,
) -> str:
    """Find passages in the ingested documents by meaning (vector search over Qdrant)."""
    hits = retrieval.vector_search(query, k=max(1, min(k, 10)))
    if not hits:
        return "No passages found."
    return "\n\n".join(
        f"[{h['path']}] {h['heading']} (score {h['score']})\n{h['text']}" for h in hits
    )


def query_graph(
    cypher: Annotated[str, Field(description="A read only Cypher query against the knowledge graph.")],
) -> str:
    """Run a read only Cypher query against Neo4j. Use describe_graph first if unsure of the schema."""
    try:
        rows = retrieval.run_cypher(cypher)
    except Exception as exc:  # noqa: BLE001
        return f"Query failed: {exc}. Check the schema with describe_graph and try again."
    if not rows:
        return "Query ran and returned no rows."
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str)


def describe_graph() -> str:
    """List the entity labels, relationship types and properties currently in the graph."""
    return retrieval.graph_schema()


def ask_the_brain(
    question: Annotated[str, Field(description="The user's question, rephrased if helpful.")],
) -> str:
    """Hybrid GraphRAG: retrieve passages by meaning, then expand into the graph around them."""
    result = retrieval.hybrid_context(question)
    return result["context"]


TOOLS = [ask_the_brain, search_documents, query_graph, describe_graph]


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
def build_agent(name: str = "enterprise-brain", instructions: str = INSTRUCTIONS) -> Agent:
    client = OpenAIChatClient(
        model=config.CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )
    return Agent(client, name=name, instructions=instructions, tools=TOOLS)


async def ask(question: str, agent: Agent | None = None) -> str:
    agent = agent or build_agent()
    result = await agent.run(question)
    return result.text
