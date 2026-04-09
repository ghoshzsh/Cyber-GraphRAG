"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — GraphRAG (Graph-Augmented Retrieval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ What RAG is and why it exists
  ✦ What GraphRAG adds on top of plain RAG
  ✦ How to embed text using NIM's embedding model
  ✦ Cosine similarity search (no vector DB needed)
  ✦ Reranking: using a 3rd model to re-score results
  ✦ The full Retrieve → Rerank → Generate pipeline

CONCEPT — RAG vs GraphRAG:

  Plain RAG:
    Query  →  embed query  →  find similar CHUNKS  →  LLM answers

  GraphRAG:
    Query  →  embed query  →  find similar GRAPH NODES
                           →  walk graph neighbourhood
                           →  collect richer multi-hop context
                           →  Reranker scores each context piece
                           →  LLM answers with best-scored context

  The graph gives you RELATIONSHIPS between evidence pieces.
  e.g., "10.0.0.45 attacked 3 hosts, all via sqlmap, all blocked"
  Plain RAG can't see that — it finds individual chunks in isolation.
"""

import os
import json
import numpy as np
from typing import Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Re-use the graph from Step 2
from step2_graph_builder import ThreatGraph

load_dotenv()
console = Console()


# ──────────────────────────────────────────────
# 1.  EMBEDDING CLIENT  (NIM embedding model)
# ──────────────────────────────────────────────
# Embeddings = dense vector representations of text.
# Similar meaning → similar vectors → high cosine similarity.

def get_embedder() -> OllamaEmbeddings:
    """
    NIM's embedding endpoint is OpenAI-compatible.
    We use langchain_openai.Embeddings with a custom base_url.
    """
    return OllamaEmbeddings(
    model="qwen3-embedding:4b"  # or any embedding model you pulled
)  # or any embedding model you pulled



# ──────────────────────────────────────────────
# 2.  RERANKER CLIENT  (NIM reranker model)
# ──────────────────────────────────────────────
# A reranker is a cross-encoder: it takes (query, passage) pairs
# and scores each pair for relevance — more accurate than embedding
# similarity alone, but too slow to run on the whole corpus.
# Strategy: embed → top-K → rerank top-K → use top-N.

class NIMReranker:
    """
    Thin wrapper around NIM's reranking endpoint.
    NIM exposes /v1/ranking for cross-encoder reranking.
    """
    def __init__(self):
        import httpx
        self.client   = httpx.Client()
        self.base_url = os.getenv("NIM_BASE_URL", "https://ai.api.nvidia.com/v1")
        self.api_key  = os.getenv("NVIDIA_API_KEY")
        self.model    = os.getenv("NIM_RERANK_MODEL", "nvidia/llama-nemotron-rerank-vl-1b-v2")

    def rerank(self, query: str, passages: list[str], top_n: int = 3) -> list[tuple[float, str]]:
        """
        Returns list of (score, passage) sorted best-first.
        Falls back to order-preserved list if API fails (for offline testing).
        """
        try:
            response = self.client.post(
                f"{self.base_url}/retrieval/{self.model}/reranking",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": {"text": query},
                    "passages": [{"text": p} for p in passages],
                },
                timeout=30,
            )
            response.raise_for_status()
            rankings = response.json()["rankings"]
            # rankings: [{"index": int, "logit": float}, ...]
            scored = [(r["logit"], passages[r["index"]]) for r in rankings[:top_n]]
            return sorted(scored, key=lambda x: x[0], reverse=True)
        except Exception as e:
            console.print(f"[yellow]⚠ Reranker unavailable ({e}), using embedding order[/]")
            # Graceful fallback: return passages as-is with dummy scores
            return [(1.0 - i * 0.1, p) for i, p in enumerate(passages[:top_n])]


# ──────────────────────────────────────────────
# 3.  GRAPH-RAG ENGINE
# ──────────────────────────────────────────────

class GraphRAGEngine:
    """
    The core GraphRAG implementation.

    Index:  For every IP node in the graph, we generate a text context
            (using get_ip_context from Step 2) and embed it.
            This creates a "semantic index over the graph".

    Query:  Embed the question → find most similar IP contexts →
            expand via graph neighbours → rerank → answer with LLM.
    """

    def __init__(self, graph: ThreatGraph):
        self.graph    = graph
        self.embedder = get_embedder()
        self.reranker = NIMReranker()
        self.llm      = ChatOpenAI(
            model    = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.3-70b-instruct"),
            base_url = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
            api_key  = os.getenv("NVIDIA_API_KEY"),
            temperature = 0,
        )

        # The "vector store" — stored in memory as plain numpy arrays
        # In production you'd use Qdrant, Weaviate, pgvector, etc.
        self.node_ids:   list[str]        = []   # IP addresses
        self.node_texts: list[str]        = []   # text contexts
        self.embeddings: np.ndarray | None = None  # shape: (N, embed_dim)

    # ── INDEXING ──────────────────────────────

    def build_index(self):
        """
        Embed every IP node's graph context.
        This is the "offline" phase — run once, cache results.
        """
        from step2_graph_builder import NODE_IP

        console.print("[cyan]Building GraphRAG index...[/]")

        ip_nodes = [
            n for n, d in self.graph.G.nodes(data=True)
            if d.get("type") == NODE_IP
        ]

        # Generate text context for each IP (from the graph)
        texts = [self.graph.get_ip_context(ip) for ip in ip_nodes]

        # for i, t in enumerate(texts):
        #     print(i, type(t), t)    

        # Embed all texts in one batch API call
        console.print(f"  Embedding {len(texts)} IP contexts via NIM...")
        raw_embeddings = self.embedder.embed_documents(texts)

        self.node_ids   = ip_nodes
        self.node_texts = texts
        self.embeddings = np.array(raw_embeddings)  # shape: (N, 1024) for e5-v5

        console.print(f"[green]✓ Index ready: {len(ip_nodes)} nodes, embedding dim={self.embeddings.shape[1]}[/]")

    def save_index(self, path: str = "data/graph_rag_index.json"):
        """Persist the index so we don't re-embed on every run."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "node_ids"  : self.node_ids,
                "node_texts": self.node_texts,
                "embeddings": self.embeddings.tolist(),
            }, f)
        console.print(f"[green]✓ Index saved → {path}[/]")

    def load_index(self, path: str = "data/graph_rag_index.json"):
        with open(path) as f:
            data = json.load(f)
        self.node_ids   = data["node_ids"]
        self.node_texts = data["node_texts"]
        self.embeddings = np.array(data["embeddings"])
        console.print(f"[green]✓ Index loaded: {len(self.node_ids)} nodes[/]")

    # ── RETRIEVAL ─────────────────────────────

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between query vector and all indexed vectors.
        sim = (A · B) / (|A| |B|)
        Result: 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite
        """
        # Normalize vectors to unit length
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        M = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
        return M @ q   # dot product = cosine sim after normalization

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        """
        Step 1 of GraphRAG: semantic retrieval.
        Returns (node_id, context_text, similarity_score) triples.
        """
        # Embed the query
        q_vec = np.array(self.embedder.embed_query(query))

        # Cosine similarity against all node embeddings
        scores = self._cosine_similarity(q_vec)

        # Rank by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self.node_ids[idx],
                self.node_texts[idx],
                float(scores[idx]),
            ))
        return results

    def expand_with_graph(self, retrieved_nodes: list[str]) -> list[str]:
        """
        Step 2 of GraphRAG: graph expansion.
        For each retrieved IP, also pull in its direct neighbours.
        This is what plain RAG CANNOT do — follow relationships.
        """
        expanded_contexts = []
        seen = set()

        for node_id in retrieved_nodes:
            if node_id in seen:
                continue
            seen.add(node_id)
            expanded_contexts.append(self.graph.get_ip_context(node_id))

            # Walk one hop: add neighbour contexts too
            for neighbour in self.graph.neighbors_of(node_id):
                if neighbour not in seen and self.graph.G.nodes[neighbour].get("type") == "IP":
                    seen.add(neighbour)
                    expanded_contexts.append(self.graph.get_ip_context(neighbour))

        return expanded_contexts

    # ── FULL PIPELINE: Retrieve → Rerank → Generate ──

    def answer(self, question: str) -> str:
        """
        The full GraphRAG pipeline:
          1. Embed question
          2. Retrieve top-K similar IP contexts from the index
          3. Expand via graph (get neighbours too)
          4. Rerank all contexts using the cross-encoder
          5. Feed top-N to the LLM for final answer
        """
        console.print(f"\n[bold cyan]Question:[/] {question}")

        # ── Step 1: Semantic retrieval ─────────
        console.print("[dim]  Step 1: Embedding-based retrieval...[/]")
        retrieved = self.retrieve(question, top_k=5)
        retrieved_node_ids = [r[0] for r in retrieved]

        self._show_retrieval_table(retrieved)

        # ── Step 2: Graph expansion ────────────
        console.print("[dim]  Step 2: Graph neighbourhood expansion...[/]")
        contexts = self.expand_with_graph(retrieved_node_ids)
        console.print(f"  → {len(contexts)} context pieces after graph expansion")

        # ── Step 3: Reranking ──────────────────
        console.print("[dim]  Step 3: Reranking with cross-encoder...[/]")
        ranked = self.reranker.rerank(question, contexts, top_n=3)
        top_context = "\n\n---\n\n".join(text for _, text in ranked)

        # ── Step 4: Generate ───────────────────
        console.print("[dim]  Step 4: Generating answer with LLM...[/]")
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a cybersecurity analyst. Answer the analyst's question "
             "using ONLY the provided graph context. Be concise and specific."),
            ("human",
             "GRAPH CONTEXT:\n{context}\n\nQUESTION: {question}"),
        ])
        chain    = prompt | self.llm
        response = chain.invoke({"context": top_context, "question": question})
        answer   = response.content

        console.print(Panel(answer, title="[bold green]Answer[/]", border_style="green"))
        return answer

    # ── Display helpers ───────────────────────

    def _show_retrieval_table(self, results: list[tuple[str, str, float]]):
        table = Table(title="Top Retrieved Nodes", border_style="dim")
        table.add_column("IP",    style="cyan")
        table.add_column("Score", style="yellow")
        table.add_column("Preview", style="dim")
        for node_id, text, score in results:
            preview = text.split("\n")[0][:60]
            table.add_row(node_id, f"{score:.3f}", preview)
        console.print(table)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()

    console.print(Panel(
        "[bold]STEP 3 — GraphRAG[/]\n"
        "Concepts: Embeddings, Cosine Similarity, Graph Expansion, Reranking\n"
        "Models:   NIM Chat + NIM Embed + NIM Reranker",
        title="🔎 Threat Intel EDU", border_style="blue"
    ))

    # Load the graph built in Step 2
    graph = ThreatGraph.load("data/threat_graph.json")

    rag = GraphRAGEngine(graph)

    # Build index (embeds all IP contexts) — or load cached
    index_path = "data/graph_rag_index.json"
    if os.path.exists(index_path):
        rag.load_index(index_path)
    else:
        rag.build_index()
        rag.save_index(index_path)

    # ── Demo questions ────────────────────────
    QUESTIONS = [
        "Which IP addresses performed port scanning?",
        "Were there any SQL injection attempts? What were the targets?",
        "Which malicious IPs successfully bypassed the firewall?",
    ]

    for q in QUESTIONS:
        rag.answer(q)
        console.print()

    console.print("\n[bold]📚 What just happened?[/]")
    console.print("  1. Each IP's graph context was converted to an embedding vector")
    console.print("  2. The question was embedded the same way")
    console.print("  3. Cosine similarity found the most relevant IPs")
    console.print("  4. Graph expansion pulled in connected neighbours (multi-hop)")
    console.print("  5. The reranker re-scored each context piece for precision")
    console.print("  6. The LLM answered using only the top-ranked context")
    console.print("\n[dim]→ Run step4_mcp_server.py next[/]")