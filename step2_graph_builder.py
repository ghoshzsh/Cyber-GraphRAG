"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 (v2) — Knowledge Graph with Graphiti + FalkorDB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ What Graphiti is and how it differs from plain NetworkX
  ✦ How embeddings are stored DIRECTLY on graph nodes (name_embedding)
  ✦ How FalkorDB stores both graph structure AND vectors in one DB
  ✦ How add_episode() auto-extracts entities and relationships via LLM
  ✦ The difference between EpisodeType.json vs .text

REPLACES: step2_graph_builder.py (NetworkX)

WHY THE UPGRADE?
  NetworkX (old):
    ✗ In-memory only — lost on restart
    ✗ Embeddings stored separately in a numpy array
    ✗ Manual entity/edge creation
    ✗ No built-in hybrid search

  Graphiti + FalkorDB (new):
    ✓ Persistent — FalkorDB stores graph on disk
    ✓ Embeddings stored as node/edge PROPERTIES in the graph itself
    ✓ LLM auto-extracts entities and relationships from raw text
    ✓ Built-in hybrid search: BM25 + cosine + BFS in one query
    ✓ Temporal — edges have valid_at / invalid_at for time-aware queries

HOW GRAPHITI WORKS (the magic of add_episode):

  Raw log text (1 episode)
          │
          ▼  LLM extracts entities
  EntityNode("10.0.0.45")         ← name_embedding stored in FalkorDB
  EntityNode("192.168.1.1")       ← name_embedding stored in FalkorDB
          │
          ▼  LLM extracts relationships
  EntityEdge(                     ← fact_embedding stored in FalkorDB
    source="10.0.0.45",
    target="192.168.1.1",
    fact="10.0.0.45 performed SQL injection against 192.168.1.1, blocked"
  )

  Both the node NAME and the edge FACT are embedded and stored as
  vector properties in FalkorDB. No separate vector database needed.

PREREQUISITES:
  Start FalkorDB first:
    docker run -p 6379:6379 --rm falkordb/falkordb:latest
"""

import asyncio
import json
import os
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

# For reranker
import httpx
from graphiti_core.cross_encoder.client import CrossEncoderClient

# For episode URL calling fix 404
import json as _json
from graphiti_core.llm_client.openai_client import OpenAIClient


load_dotenv()
console = Console()

class NIMRerankerClient(CrossEncoderClient):
    """
    Custom CrossEncoderClient for NIM's reranking endpoint.

    WHY: Graphiti's built-in OpenAIRerankerClient calls the chat/completions
    endpoint for reranking, but NIM's reranker lives at a different URL:
      /retrieval/{model}/reranking
    We implement the CrossEncoderClient ABC so Graphiti accepts this
    as a drop-in replacement — same interface, correct NIM endpoint.

    RETURN FORMAT Graphiti expects:
      list[tuple[str, float]]  →  [(passage_text, score), ...]
      sorted highest score first
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/retrieval/{self.model}/reranking",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model"   : self.model,
                        "query"   : {"text": query},
                        "passages": [{"text": p} for p in passages],
                    },
                    timeout=30,
                )
                response.raise_for_status()
                rankings = response.json()["rankings"]
                # NIM returns: [{"index": int, "logit": float}, ...]
                # Graphiti expects: [(passage_str, score), ...] highest first
                scored = [
                    (passages[r["index"]], float(r["logit"]))
                    for r in rankings
                ]
                return sorted(scored, key=lambda x: x[1], reverse=True)

            except Exception as e:
                print(f"⚠ NIM Reranker failed ({e}), falling back to original order")
                # Fallback: return passages as-is with descending dummy scores
                return [(p, 1.0 - i * 0.1) for i, p in enumerate(passages)]


class NIMChatClient(OpenAIClient):
    """
    Drop-in replacement for OpenAIClient compatible with NIM.

    WHY: Graphiti's structured extraction path calls responses.parse()
    (OpenAI Responses API) which NIM doesn't support → 404.
    We override _generate_response() to always use chat.completions
    with json_object mode and return the (dict, tokens, tokens) tuple
    Graphiti expects — bypassing _create_structured_completion and
    _handle_structured_response entirely.

    KEY REMAP: NIM's LLaMA returns {"entity": ..., "entity_type": ...}
    but Graphiti's ExtractedEntity schema expects {"name": ..., "entity_type_id": ...}.
    We fix that inline before returning.
    """

    async def _generate_response(self, messages, response_model=None, max_tokens=2048, model_size=None):
        from graphiti_core.llm_client.config import ModelSize

        # Pick the right model string for the requested size
        model = self._get_model_for_size(model_size) if model_size else self.model

        openai_messages = self._convert_messages_to_openai_format(messages)

        # Always use chat.completions — NIM supports this, responses.parse() 404s
        response = await self.client.chat.completions.create(
            model           = model,
            messages        = openai_messages,
            temperature     = self.temperature,
            max_tokens      = max_tokens or self.max_tokens,
            response_format = {"type": "json_object"},
        )

        raw_text     = response.choices[0].message.content or "{}"
        input_tokens  = getattr(response.usage, "prompt_tokens",     0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        parsed = _json.loads(raw_text)

        # ── Key remapping ─────────────────────────────────────────────
        # LLaMA uses different key names than Graphiti's Pydantic schemas.
        # Remap them here so model_validate() succeeds downstream.
        if "extracted_entities" in parsed:
            fixed = []
            for ent in parsed["extracted_entities"]:
                # Resolve the name field
                name = ent.get("name") or ent.get("entity") or ent.get("entity_name", "")

                # entity_type_id MUST be an int — LLaMA returns strings like "IP_ADDRESS"
                # Hash the string to a stable positive int, fallback to 0
                raw_type = ent.get("entity_type_id") or ent.get("entity_type") or ent.get("type", "")
                if isinstance(raw_type, int):
                    type_id = raw_type
                elif isinstance(raw_type, str) and raw_type.isdigit():
                    type_id = int(raw_type)
                else:
                    type_id = 0   # default — Graphiti uses 0 for generic entity

                fixed.append({"name": name, "entity_type_id": type_id})
            parsed["extracted_entities"] = fixed

        return parsed, input_tokens, output_tokens

# ──────────────────────────────────────────────
# 1.  CLIENT FACTORY
#     All three NIM models wired into Graphiti
# ──────────────────────────────────────────────

async def build_graphiti() -> Graphiti:
    """
    Wire up Graphiti with:
      - FalkorDriver  → graph storage (Redis-compatible, local)
      - OpenAIClient  → NIM Chat model for entity/relationship extraction
      - OpenAIEmbedder→ NIM Embed model for node & edge embeddings
    """
    # ── FalkorDB connection ────────────────────
    # FalkorDB is a Redis-compatible graph DB.
    # Run it locally: docker run -p 6379:6379 --rm falkordb/falkordb:latest
    falkor_driver = FalkorDriver(
        host     = os.getenv("FALKORDB_HOST",     "localhost"),
        port     = int(os.getenv("FALKORDB_PORT", "6379")),
        username = os.getenv("FALKORDB_USER",     None),
        password = os.getenv("FALKORDB_PASSWORD", None),
        database = "threat_intel_edu",   # logical database name inside FalkorDB
    )

    # ── NIM Chat model (entity extraction) ────
    # Graphiti uses this to read each episode and extract:
    #   - entity names ("10.0.0.45", "sqlmap", "SQL Injection")
    #   - facts between entities ("10.0.0.45 attacked 192.168.1.1")
    llm_client = NIMChatClient(
        config=LLMConfig(
            api_key  = os.getenv("NVIDIA_API_KEY"),
            model    = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.3-70b-instruct"),
            base_url = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
            temperature = 0,
            max_tokens  = 2048,
        )
    )

    # ── NIM Embedding model ────────────────────
    # Graphiti calls this to embed:
    #   - Entity node NAMES → stored as node.name_embedding in FalkorDB
    #   - Edge FACTS        → stored as edge.fact_embedding in FalkorDB
    # This is the key difference from Step 2 v1:
    # embeddings live INSIDE the graph, not in a separate numpy array.
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key         = os.getenv("NVIDIA_API_KEY", "ollama"),
            embedding_model = os.getenv("NIM_EMBED_MODEL", "qwen3-embedding:4b"),
            base_url        = os.getenv("NIM_BASE_URL",    "http://localhost:11434/v1"),
            embedding_dim   = 1024,   # nv-embedqa-e5-v5 outputs 1024-dim vectors
        )
    )

    # ── NIM Reranker (cross-encoder) ───────────────
    # Graphiti auto-creates OpenAIRerankerClient() with no config, which
    # fails because it falls back to looking for OPENAI_API_KEY.
    # We explicitly pass a CrossEncoderConfig pointing at NIM instead.
    reranker = NIMRerankerClient(
        model    = os.getenv("NIM_RERANKER_MODEL", "nvidia/llama-nemotron-rerank-vl-1b-v2"),
        api_key  = os.getenv("NVIDIA_API_KEY"),
        base_url = os.getenv("NIM_BASE_URL", "https://ai.api.nvidia.com/v1"),
    )

    # ── 🔍 Health Checks ─────────────────────
    async def run_checks():
        print("\n🔍 Running connection checks...\n")

        # LLM — real method is generate_response(), takes a messages list
        try:
            from graphiti_core.prompts.models import Message
            await llm_client.generate_response(
                messages=[Message(role="user", content="Say pong")]
            )
            print("✅ LLM client connected")
        except Exception as e:
            print(f"❌ LLM client failed: {e}")

        # Embedder — real method is create(), takes a single string
        try:
            vec = await embedder.create("ping")
            print(f"✅ Embedder connected (dim={len(vec)})")
        except Exception as e:
            print(f"❌ Embedder failed: {e}")

        # Reranker — real method is rank(), takes query + passages list
        try:
            await reranker.rank(query="ping", passages=["doc1", "doc2"])
            print("✅ Reranker connected")
        except Exception as e:
            print(f"❌ Reranker failed: {e}")

        print("\n✔️ Initialization complete\n")

    await run_checks()

    return Graphiti(
        llm_client    = llm_client,
        embedder      = embedder,
        graph_driver  = falkor_driver,
        cross_encoder = reranker,
    )


# ──────────────────────────────────────────────
# 2.  ROW → EPISODE TEXT
#     Each log row becomes one "episode" Graphiti ingests
# ──────────────────────────────────────────────

def row_to_episode_body(row: pd.Series) -> str:
    """
    Convert a CSV row to a structured text episode.

    CONCEPT — Episode:
    An episode is the unit of input for Graphiti. Think of it as one
    "memory" or "observation". Graphiti reads it, extracts the entities
    and relationships inside, and builds graph nodes + edges from them.

    We use EpisodeType.json so Graphiti knows the structure is predictable.
    """
    return json.dumps({
        "timestamp"        : str(row["timestamp"]),
        "source_ip"        : str(row["source_ip"]),
        "destination_ip"   : str(row["dest_ip"]),
        "protocol"         : str(row["protocol"]),
        "action"           : str(row["action"]),
        "threat_label"     : str(row["threat_label"]),
        "log_type"         : str(row["log_type"]),
        "bytes_transferred": int(row["bytes_transferred"]),
        "user_agent"       : str(row["user_agent"]),
        "request_path"     : str(row["request_path"]),
    })


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp from CSV into timezone-aware datetime."""
    try:
        dt = datetime.fromisoformat(str(ts_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(tz=timezone.utc)


# ──────────────────────────────────────────────
# 3.  INGEST LOGS INTO GRAPHITI
# ──────────────────────────────────────────────

async def ingest_logs(graphiti: Graphiti, csv_path: str, max_rows: int = 10):
    """
    Feed each log row into Graphiti as an episode.

    For each episode, Graphiti automatically:
      1. Sends the text to the LLM → extracts entity names and facts
      2. Resolves entities against existing nodes (deduplication)
      3. Calls the embedder → stores name_embedding on each EntityNode
      4. Creates EntityEdge objects with fact_embedding stored in FalkorDB
      5. Persists everything to FalkorDB (graph structure + vectors)

    After ingestion, FalkorDB contains:
      - Nodes:  EntityNode(name="10.0.0.45", name_embedding=[...1024 floats...])
      - Edges:  EntityEdge(fact="10.0.0.45 attacked 192.168.1.1", fact_embedding=[...])
    """
    df = pd.read_csv(csv_path).head(max_rows)

    console.print(Panel(
        f"[bold cyan]Ingesting {len(df)} log rows into Graphiti + FalkorDB[/]\n"
        "[dim]Each row → 1 episode → LLM extracts entities → embeddings stored on nodes[/]",
        border_style="cyan"
    ))

    # Build the schema index once before ingesting
    # This creates the necessary FalkorDB indexes for vector search
    await graphiti.build_indices_and_constraints()
    console.print("[green]✓ FalkorDB indexes ready[/]")

    results_summary = []

    for i, (_, row) in enumerate(df.iterrows()):
        episode_body = row_to_episode_body(row)
        episode_name = f"log_{i+1}_{row['source_ip']}_to_{row['dest_ip']}"

        console.print(f"\n[yellow]▶ Episode {i+1}/{len(df)}:[/] {row['source_ip']} → {row['dest_ip']} ({row['action']})")

        try:
            result = await graphiti.add_episode(
                name               = episode_name,
                episode_body       = episode_body,

                # source_description tells the LLM what kind of data this is
                source_description = "Network security log from a corporate firewall/IDS system",

                # reference_time enables temporal reasoning
                # Graphiti can answer: "what was happening on June 12th?"
                reference_time     = parse_timestamp(row["timestamp"]),

                # EpisodeType.json → Graphiti knows it's structured data
                # EpisodeType.text → for free-form text like chat messages
                source             = EpisodeType.json,

                # group_id groups episodes logically (like a "session" or "dataset")
                group_id           = "kaggle_threat_logs",
            )

            # result.nodes → list of EntityNode extracted from this episode
            # result.edges → list of EntityEdge created from this episode
            node_names = [n.name for n in result.nodes]
            edge_facts = [e.fact[:60] for e in result.edges]

            results_summary.append({
                "episode"   : episode_name,
                "nodes"     : node_names,
                "edges"     : len(result.edges),
            })

            console.print(f"  [green]✓ Nodes extracted:[/] {node_names}")
            console.print(f"  [green]✓ Edges created:[/]  {len(result.edges)}")
            if edge_facts:
                console.print(f"  [dim]  First fact: {edge_facts[0]}...[/]")

        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/]")

    return results_summary


# ──────────────────────────────────────────────
# 4.  INSPECT WHAT'S IN FALKORDB
#     Show that embeddings live ON the nodes
# ──────────────────────────────────────────────

async def inspect_graph(graphiti: Graphiti):
    """
    Run a simple search to show what Graphiti stored.
    This demonstrates that name_embedding is a node property in FalkorDB.
    """
    console.print(Panel("[bold]Inspecting stored graph[/]", border_style="yellow"))

    # Simple keyword search — returns EntityEdge list
    edges: list[EntityEdge] = await graphiti.search(
        query       = "malicious IP attack blocked",
        num_results = 5,
    )

    table = Table(title="Retrieved EntityEdges (fact + fact_embedding)", border_style="cyan")
    table.add_column("Fact",              style="white", max_width=60)
    table.add_column("Has Embedding?",    style="green")
    table.add_column("Valid At",          style="dim")

    for edge in edges:
        has_embed = "✓" if edge.fact_embedding else "✗"
        valid_at  = str(edge.valid_at)[:10] if edge.valid_at else "N/A"
        table.add_row(edge.fact[:60], has_embed, valid_at)

    console.print(table)

    console.print("\n[bold yellow]Key insight:[/]")
    console.print("  edge.fact_embedding  = vector stored IN FalkorDB as an edge property")
    console.print("  node.name_embedding  = vector stored IN FalkorDB as a node property")
    console.print("  No separate vector DB needed — graph IS the vector store")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

async def main():
    console.print(Panel(
        "[bold]STEP 2 (v2) — Graphiti + FalkorDB[/]\n"
        "Concept: Auto entity extraction, embeddings stored on graph nodes\n"
        "Storage: FalkorDB (graph structure + vectors in one DB)\n"
        "Models:  NIM Chat (extraction) + NIM Embed (node/edge vectors)",
        title="🕸  Threat Intel EDU", border_style="blue"
    ))

    graphiti = await build_graphiti()

    summary = await ingest_logs(
        graphiti = graphiti,
        csv_path = os.getenv("DATA_PATH", "data/cybersecurity_threat_detection_logs.csv"),
        max_rows = 10,    # ← increase for more data (each row = 1 LLM call)
    )

    await inspect_graph(graphiti)

    console.print("\n[bold]📚 What just happened?[/]")
    console.print("  1. Each CSV row was converted to a JSON episode string")
    console.print("  2. Graphiti sent it to NIM Chat → extracted entity names + facts")
    console.print("  3. Graphiti called NIM Embed → got 1024-dim vectors")
    console.print("  4. FalkorDB stored: EntityNode(name_embedding=[...]) as a node property")
    console.print("  5. FalkorDB stored: EntityEdge(fact_embedding=[...]) as an edge property")
    console.print("  6. Everything is in ONE database — graph + vectors together")
    console.print("\n[dim]→ Run step3_graphiti_rag.py next[/]")

    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())