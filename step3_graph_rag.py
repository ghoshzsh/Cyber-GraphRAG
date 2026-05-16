"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 (v2) — GraphRAG via Graphiti's Hybrid Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ Graphiti's search_() method — retrieves BOTH nodes and edges
  ✦ Three search strategies used together: BM25 + cosine + BFS
  ✦ How node.name_embedding and edge.fact_embedding power retrieval
  ✦ How to configure SearchConfig for different retrieval strategies
  ✦ How to build a RAG answer from graph search results

REPLACES: step3_graph_rag.py (manual cosine search + custom reranker)

SEARCH ARCHITECTURE IN GRAPHITI:

  Your Question
       │
       ├── BM25 (keyword)  ──────────┐
       │   searches node names        │
       │   and edge facts             │   Results merged
       │                             │   by reranker
       ├── Cosine Similarity ────────┤   (RRF or cross-encoder)
       │   uses name_embedding        │        │
       │   and fact_embedding         │        ▼
       │   stored in FalkorDB         │   Top-N Nodes + Edges
       │                             │        │
       └── BFS (graph walk) ─────────┘        ▼
           expands from matched               LLM Answer
           nodes to neighbours

  All three happen in ONE search_() call against FalkorDB.
  This replaces the manual: embed → cosine → expand → rerank pipeline
  from Step 3 v1.

NODE vs EDGE RETRIEVAL:
  ┌─────────────────────────────────────────────────┐
  │  EntityNode                                     │
  │    .name           "10.0.0.45"                  │
  │    .summary        "External IP performing..."  │
  │    .name_embedding [0.12, -0.34, ...]  ← vector │
  │    .attributes     {custom fields}              │
  └─────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────┐
  │  EntityEdge                                     │
  │    .name           "ATTACKED"                   │
  │    .fact           "10.0.0.45 attacked..."      │
  │    .fact_embedding [0.55, 0.12, ...]  ← vector  │
  │    .valid_at       2024-06-12T00:00:00          │
  └─────────────────────────────────────────────────┘

  Step 3 v1 only searched nodes (IP contexts).
  Step 3 v2 searches BOTH — nodes give you entity info,
  edges give you specific facts/relationships.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import (
    SearchConfig,
    NodeSearchConfig,   EdgeSearchConfig,
    NodeSearchMethod,   EdgeSearchMethod,
    NodeReranker,       EdgeReranker,
    SearchResults,
)

# LangChain for the final answer generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
console = Console()


# ──────────────────────────────────────────────
# 1.  REUSE THE GRAPHITI FACTORY FROM STEP 2
# ──────────────────────────────────────────────

def build_graphiti() -> Graphiti:
    """Same setup as step2_graphiti_builder.py."""
    falkor_driver = FalkorDriver(
        host     = os.getenv("FALKORDB_HOST",     "localhost"),
        port     = int(os.getenv("FALKORDB_PORT", "6379")),
        username = os.getenv("FALKORDB_USER",     None),
        password = os.getenv("FALKORDB_PASSWORD", None),
        database = "threat_intel_edu",
    )

    llm_client = OpenAIClient(
        config=LLMConfig(
            api_key  = os.getenv("NVIDIA_API_KEY"),
            model    = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.1-8b-instruct"),
            base_url = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
            temperature = 0,
        )
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key         = os.getenv("NVIDIA_API_KEY"),
            embedding_model = os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
            base_url        = os.getenv("NIM_BASE_URL",    "https://integrate.api.nvidia.com/v1"),
            embedding_dim   = 1024,
        )
    )

    return Graphiti(
        llm_client   = llm_client,
        embedder     = embedder,
        graph_driver = falkor_driver,
    )


# ──────────────────────────────────────────────
# 2.  SEARCH CONFIGURATIONS
#     Educational: showing different search modes
# ──────────────────────────────────────────────

def make_search_config(strategy: str = "hybrid") -> SearchConfig:
    """
    Build a SearchConfig that controls HOW Graphiti searches FalkorDB.

    STRATEGY OPTIONS (for learning):
      "cosine"  → pure vector search on name_embedding / fact_embedding
      "bm25"    → pure keyword search on node names and edge facts
      "hybrid"  → all three: BM25 + cosine + BFS graph walk (default)
    """

    if strategy == "cosine":
        # Only semantic similarity — good for conceptual questions
        node_methods = [NodeSearchMethod.cosine_similarity]
        edge_methods = [EdgeSearchMethod.cosine_similarity]
        node_reranker = NodeReranker.rrf
        edge_reranker = EdgeReranker.rrf

    elif strategy == "bm25":
        # Only keyword matching — good for exact IP/protocol lookups
        node_methods = [NodeSearchMethod.bm25]
        edge_methods = [EdgeSearchMethod.bm25]
        node_reranker = NodeReranker.rrf
        edge_reranker = EdgeReranker.rrf

    else:  # "hybrid" — use everything
        # BM25 + cosine + BFS graph expansion (best for RAG)
        # RRF (Reciprocal Rank Fusion) merges the three result lists
        node_methods = [
            NodeSearchMethod.bm25,
            NodeSearchMethod.cosine_similarity,
            NodeSearchMethod.bfs,          # walk to neighbours
        ]
        edge_methods = [
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity,
            EdgeSearchMethod.bfs,
        ]
        node_reranker = NodeReranker.rrf   # merge BM25 + cosine + BFS lists
        edge_reranker = EdgeReranker.rrf

    return SearchConfig(
        node_config = NodeSearchConfig(
            search_methods = node_methods,
            reranker       = node_reranker,
            sim_min_score  = 0.4,   # minimum cosine similarity threshold
        ),
        edge_config = EdgeSearchConfig(
            search_methods = edge_methods,
            reranker       = edge_reranker,
            sim_min_score  = 0.4,
        ),
        limit = 8,  # max results per search type
    )


# ──────────────────────────────────────────────
# 3.  SEARCH AND DISPLAY
#     Retrieve both nodes and edges, show embeddings
# ──────────────────────────────────────────────

async def search_and_display(
    graphiti: Graphiti,
    query: str,
    strategy: str = "hybrid",
) -> SearchResults:
    """
    Run Graphiti's hybrid search and display BOTH node and edge results.

    The key point: we retrieve EntityNode (with name_embedding) AND
    EntityEdge (with fact_embedding) in a single search call.
    """
    console.print(f"\n[bold cyan]🔍 Query:[/] {query}")
    console.print(f"[dim]   Strategy: {strategy}[/]")

    config  = make_search_config(strategy)
    results: SearchResults = await graphiti.search_(
        query  = query,
        config = config,
    )

    # ── Display retrieved NODES ───────────────
    if results.nodes:
        node_table = Table(
            title   = f"Retrieved EntityNodes ({len(results.nodes)})",
            border_style = "green"
        )
        node_table.add_column("Entity Name",    style="bold white")
        node_table.add_column("Summary",        style="dim",   max_width=50)
        node_table.add_column("Has Embedding",  style="green")
        node_table.add_column("Reranker Score", style="yellow")

        for i, node in enumerate(results.nodes):
            # name_embedding is stored as a property in FalkorDB
            # It's a list of 1024 floats — the semantic vector of the node name
            has_emb = f"✓ ({len(node.name_embedding)}d)" if node.name_embedding else "✗"

            score = ""
            if results.node_reranker_scores and i < len(results.node_reranker_scores):
                score = f"{results.node_reranker_scores[i]:.3f}"

            node_table.add_row(
                node.name,
                (node.summary or "")[:50],
                has_emb,
                score,
            )
        console.print(node_table)
    else:
        console.print("[dim]  No nodes retrieved[/]")

    # ── Display retrieved EDGES ───────────────
    if results.edges:
        edge_table = Table(
            title   = f"Retrieved EntityEdges ({len(results.edges)})",
            border_style = "cyan"
        )
        edge_table.add_column("Fact",           style="white", max_width=55)
        edge_table.add_column("Has Embedding",  style="green")
        edge_table.add_column("Valid At",       style="dim")
        edge_table.add_column("Reranker Score", style="yellow")

        for i, edge in enumerate(results.edges):
            # fact_embedding is stored as a property in FalkorDB
            # It's the semantic vector of the relationship fact string
            has_emb = f"✓ ({len(edge.fact_embedding)}d)" if edge.fact_embedding else "✗"

            score = ""
            if results.edge_reranker_scores and i < len(results.edge_reranker_scores):
                score = f"{results.edge_reranker_scores[i]:.3f}"

            valid_at = str(edge.valid_at)[:10] if edge.valid_at else "N/A"

            edge_table.add_row(
                edge.fact[:55],
                has_emb,
                valid_at,
                score,
            )
        console.print(edge_table)
    else:
        console.print("[dim]  No edges retrieved[/]")

    return results


# ──────────────────────────────────────────────
# 4.  CONTEXT BUILDER
#     Turn graph search results into LLM-ready text
# ──────────────────────────────────────────────

def build_rag_context(results: SearchResults) -> str:
    """
    Combine node summaries and edge facts into a single context block.

    NODES give us: "what is this entity?"
    EDGES give us: "what relationships/facts involve this entity?"

    Together they give the LLM richer context than either alone.
    """
    lines = []

    if results.nodes:
        lines.append("=== ENTITIES ===")
        for node in results.nodes:
            lines.append(f"• {node.name}: {node.summary or 'No summary'}")

    if results.edges:
        lines.append("\n=== FACTS (Relationships) ===")
        for edge in results.edges:
            ts = str(edge.valid_at)[:10] if edge.valid_at else "unknown"
            lines.append(f"• [{ts}] {edge.fact}")

    return "\n".join(lines) if lines else "No relevant context found."


# ──────────────────────────────────────────────
# 5.  GENERATE ANSWER FROM GRAPH CONTEXT
# ──────────────────────────────────────────────

def build_answer_chain():
    llm = ChatOpenAI(
        model    = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.1-8b-instruct"),
        base_url = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
        api_key  = os.getenv("NVIDIA_API_KEY"),
        temperature = 0,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a cybersecurity analyst. Answer the analyst's question using "
         "ONLY the provided graph context (entities and relationship facts). "
         "Be specific and cite which facts support your answer."),
        ("human",
         "GRAPH CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ])
    return prompt | llm


async def answer_question(graphiti: Graphiti, question: str, strategy: str = "hybrid"):
    """Full RAG pipeline: search graph → build context → generate answer."""

    # Step 1: Hybrid graph search (vector + BM25 + BFS)
    results = await search_and_display(graphiti, question, strategy)

    # Step 2: Convert graph results to text context
    context = build_rag_context(results)
    console.print(Panel(
        f"[dim]{context}[/]",
        title="[bold]Context passed to LLM[/]",
        border_style="dim"
    ))

    # Step 3: Generate answer
    chain    = build_answer_chain()
    response = chain.invoke({"context": context, "question": question})

    console.print(Panel(
        response.content,
        title="[bold green]✅ Answer[/]",
        border_style="green"
    ))
    return response.content


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

async def main():
    console.print(Panel(
        "[bold]STEP 3 (v2) — Graphiti Hybrid Search[/]\n"
        "Concept: BM25 + cosine_similarity + BFS in one query\n"
        "Storage: FalkorDB (vectors live ON nodes/edges as properties)\n"
        "Retrieve: EntityNode (name_embedding) + EntityEdge (fact_embedding)",
        title="🔎 Threat Intel EDU", border_style="blue"
    ))

    graphiti = build_graphiti()

    # ── Demo 1: Compare search strategies ─────
    console.print("\n[bold yellow]DEMO 1 — Same query, different strategies[/]")
    query = "SQL injection attack blocked"

    console.print("\n[underline]Strategy: cosine (pure vector)[/]")
    await search_and_display(graphiti, query, strategy="cosine")

    console.print("\n[underline]Strategy: bm25 (pure keyword)[/]")
    await search_and_display(graphiti, query, strategy="bm25")

    console.print("\n[underline]Strategy: hybrid (BM25 + cosine + BFS)[/]")
    await search_and_display(graphiti, query, strategy="hybrid")

    # ── Demo 2: Full RAG answers ───────────────
    console.print("\n[bold yellow]DEMO 2 — Full RAG answers from graph[/]")

    questions = [
        "Which IP addresses performed port scanning?",
        "Were there SQL injection attempts and did they succeed?",
        "Which external IPs made malicious connections?",
    ]

    for q in questions:
        await answer_question(graphiti, q, strategy="hybrid")
        console.print()

    console.print("\n[bold]📚 What just happened?[/]")
    console.print("  1. search_() hit FalkorDB with 3 strategies simultaneously")
    console.print("  2. BM25 matched keyword hits in node names and edge facts")
    console.print("  3. Cosine compared query embedding vs name_embedding/fact_embedding")
    console.print("  4. BFS walked graph edges to include connected nodes")
    console.print("  5. RRF merged all three ranked lists into one")
    console.print("  6. SearchResults gave us both nodes (entities) and edges (facts)")
    console.print("  7. LLM answered from the combined entity+fact context")

    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())