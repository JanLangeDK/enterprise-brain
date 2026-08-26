"""Gradio front end. Launch from the notebook with `ui.launch_ui()`,
or run the container: `docker compose --profile app up -d rag`.
"""

from __future__ import annotations

import os

import gradio as gr

from .agent import build_agent

EXAMPLES = [
    "What is the risk on the Nordvind Energi account before the September meeting?",
    "Does the Service SLA 2026 cover the alignment work Mette thinks P-40 needs?",
    "Have we seen the same vibration symptom at another customer, and what fixed it?",
    "Who decides on the Q4 retrofit, and who influences that decision?",
    "Which customers have equipment without a variable frequency drive?",
    "List every entity connected to Sydpumpe A/S and how.",
]

DESCRIPTION = """Ask about customers, sites, equipment, contracts and people.
The agent picks its own route: vector search for wording, Cypher for structure,
hybrid GraphRAG for most real questions. Answers cite the source file."""


def build_ui(agent=None) -> gr.Blocks:
    agent = agent or build_agent()

    async def respond(message: str, history: list) -> str:
        if not message.strip():
            return "Ask me something about the accounts."
        try:
            result = await agent.run(message)
            return result.text
        except Exception as exc:  # noqa: BLE001
            return f"That failed: {exc}\n\nCheck that Neo4j and Qdrant are up and the API keys are set."

    with gr.Blocks(title="Enterprise Brain", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Enterprise Brain\n" + DESCRIPTION)
        gr.ChatInterface(fn=respond, examples=EXAMPLES, save_history=False)
    return demo


def launch_ui(share: bool = False, inline: bool = False, **kwargs):
    """Convenience for the notebook."""
    return build_ui().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=share,
        inline=inline,
        **kwargs,
    )


if __name__ == "__main__":
    launch_ui()
