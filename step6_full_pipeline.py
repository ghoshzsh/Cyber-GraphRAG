"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Full End-to-End Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script runs the complete pipeline:

  CSV Logs
     │
     ▼
  [Step 1] Entity Extraction (Pydantic + LangChain structured output)
     │  → Extracts typed ThreatEvent objects from raw log text
     │
     ▼
  [Step 2] Graph Builder (NetworkX)
     │  → Turns entities into a knowledge graph
     │  → Creates typed edges between IPs, protocols, attack types
     │
     ▼
  [Step 3] GraphRAG Index (NIM Embed + Reranker)
     │  → Embeds each node's graph context
     │  → Enables semantic search over graph nodes
     │
     ▼
  [Step 5] ReAct Agent (LangGraph + MCP)
     │  → Investigates analyst questions
     │  → Calls MCP tools to query the graph
     │  → Returns a human-readable threat report

RUN THIS FILE ONCE to see the full loop.
It's the best way to understand how all pieces connect.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()
console = Console()


async def run_full_pipeline():
    console.print(Panel(
        "[bold white]HAL-O Threat Intel — Full Educational Pipeline[/]\n"
        "[dim]LangChain + LangGraph + GraphRAG + MCP + NVIDIA NIM[/]",
        title="🛡  Threat Intel EDU v1.0",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    csv_path = os.getenv("DATA_PATH", "cybersecurity_threat_detection_logs.csv")

    # ──────────────────────────────────────────
    # PHASE 1 — Entity Extraction
    # ──────────────────────────────────────────
    console.print(Rule("[bold cyan]PHASE 1 — Entity Extraction[/]"))
    console.print("[dim]Concept: Pydantic schemas + LangChain with_structured_output()[/]\n")

    from step1_entity_extraction import extract_entities, save_events

    events_path = "data/extracted_events.json"
    if os.path.exists(events_path):
        console.print("[yellow]⚡ Loading cached extracted events (delete to re-run LLM)[/]")
        with open(events_path) as f:
            events_raw = json.load(f)
    else:
        console.print("[cyan]🔍 Calling NIM to extract entities from 3 log rows...[/]")
        events = extract_entities(csv_path, max_rows=3)
        save_events(events)
        events_raw = [e.model_dump() for e in events]

    console.print(f"[green]✓ Phase 1 complete: {len(events_raw)} structured events[/]\n")

    # ──────────────────────────────────────────
    # PHASE 2 — Build Knowledge Graph
    # ──────────────────────────────────────────
    console.print(Rule("[bold cyan]PHASE 2 — Knowledge Graph[/]"))
    console.print("[dim]Concept: Entities as nodes, relationships as typed edges[/]\n")

    from step2_graph_builder import ThreatGraph

    graph_path = "data/threat_graph.json"
    if os.path.exists(graph_path):
        console.print("[yellow]⚡ Loading cached graph[/]")
        graph = ThreatGraph.load(graph_path)
    else:
        console.print("[cyan]🕸  Building graph from full CSV...[/]")
        graph = ThreatGraph()
        graph.ingest_from_csv(csv_path)
        graph.save()

    console.print(f"[green]✓ Phase 2 complete: {graph.G.number_of_nodes()} nodes, {graph.G.number_of_edges()} edges[/]\n")

    # ──────────────────────────────────────────
    # PHASE 3 — GraphRAG
    # ──────────────────────────────────────────
    console.print(Rule("[bold cyan]PHASE 3 — GraphRAG Index[/]"))
    console.print("[dim]Concept: Embed graph nodes → cosine search → graph expansion → rerank[/]\n")

    from step3_graph_rag import GraphRAGEngine

    rag = GraphRAGEngine(graph)
    index_path = "data/graph_rag_index.json"

    if os.path.exists(index_path):
        console.print("[yellow]⚡ Loading cached GraphRAG index[/]")
        rag.load_index(index_path)
    else:
        console.print("[cyan]📐 Building embedding index over graph nodes...[/]")
        rag.build_index()
        rag.save_index(index_path)

    # One demo GraphRAG query
    console.print("\n[bold]GraphRAG demo query:[/]")
    rag.answer("Which IPs show signs of coordinated attack behaviour?")
    console.print(f"\n[green]✓ Phase 3 complete[/]\n")

    # ──────────────────────────────────────────
    # PHASE 4 — ReAct Agent via MCP
    # ──────────────────────────────────────────
    console.print(Rule("[bold cyan]PHASE 4 — ReAct Agent + MCP[/]"))
    console.print("[dim]Concept: LangGraph state machine + MCP tool calling[/]\n")

    from step5_react_agent import run_agent

    analyst_question = (
        "I need a full threat report. "
        "Which IPs are malicious? Did any succeed in bypassing our firewall? "
        "What attack types were used and what is the overall risk level?"
    )

    await run_agent(analyst_question)

    # ──────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────
    console.print(Rule("[bold green]Pipeline Complete[/]"))
    console.print(Panel(
        "[bold]What each component contributed:[/]\n\n"
        "[cyan]LangChain[/]\n"
        "  • ChatOpenAI → connected to NIM's OpenAI-compatible API\n"
        "  • with_structured_output() → forced LLM to return Pydantic-validated JSON\n"
        "  • ChatPromptTemplate → clean, reusable prompt templates\n\n"
        "[cyan]Pydantic[/]\n"
        "  • IPEntity, ThreatEvent → typed schemas = no hallucinated fields\n"
        "  • model_dump() → serialize to JSON for persistence\n\n"
        "[cyan]NetworkX (Knowledge Graph)[/]\n"
        "  • DiGraph → nodes = IPs/protocols/attacks, edges = relationships\n"
        "  • Multi-hop traversal → richer context than flat tables\n\n"
        "[cyan]GraphRAG[/]\n"
        "  • NIM Embed → semantic vectors over graph node contexts\n"
        "  • Cosine similarity → find relevant IPs by meaning, not keyword\n"
        "  • Graph expansion → follow edges for richer multi-hop evidence\n"
        "  • NIM Reranker → cross-encoder scores each context piece precisely\n\n"
        "[cyan]MCP (Model Context Protocol)[/]\n"
        "  • FastMCP server → tools as decorated Python functions\n"
        "  • MultiServerMCPClient → agent discovers tools at runtime\n"
        "  • stdio transport → subprocess lifecycle managed automatically\n\n"
        "[cyan]LangGraph ReAct[/]\n"
        "  • create_react_agent() → wires agent_node ↔ tools_node loop\n"
        "  • State = messages list → full history flows through every node\n"
        "  • Conditional edge → tool_call present? → tools; else → END\n"
        "  • astream() → observe each ReAct step live\n",
        title="📚 Learning Summary",
        border_style="bright_green",
    ))


if __name__ == "__main__":
    asyncio.run(run_full_pipeline())