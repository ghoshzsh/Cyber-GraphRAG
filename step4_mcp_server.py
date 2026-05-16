"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — MCP: Model Context Protocol Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ What MCP is and why it exists
  ✦ How to build an MCP server with FastMCP
  ✦ Exposing tools that an LLM agent can call
  ✦ How the agent (Step 5) discovers and uses these tools

CONCEPT — MCP (Model Context Protocol):
  MCP is Anthropic's open standard for connecting LLMs to external tools.
  Think of it as "USB-C for AI tools" — any MCP client can talk to
  any MCP server using the same protocol.

  Without MCP:  you hard-code tools into each agent
  With MCP:     tools are services the agent discovers at runtime

  ┌──────────────┐     MCP      ┌────────────────────┐
  │  LangGraph   │ ──────────── │  ThreatIntel MCP   │
  │  ReAct Agent │  (Step 5)   │  Server  (this file)│
  └──────────────┘             └────────────────────┘
                                    Tools exposed:
                                    • lookup_ip(ip)
                                    • check_port_scan(ip)
                                    • get_attack_summary(ip)
                                    • search_logs(query)

HOW TO RUN:
  In terminal 1:  python step4_mcp_server.py
  In terminal 2:  python step5_react_agent.py

  The agent will automatically discover and call these tools.
"""

import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# FastMCP makes building MCP servers trivial
# The @mcp.tool() decorator does all the protocol heavy lifting
from fastmcp import FastMCP

load_dotenv()

# ──────────────────────────────────────────────
# 1.  CREATE THE MCP SERVER
# ──────────────────────────────────────────────

mcp = FastMCP(
    name        = "ThreatIntelMCP",
    description = "Threat intelligence tools for the security SOC analyst agent",
)


# ──────────────────────────────────────────────
# 2.  GRAPHITI CLIENT (replaces NetworkX graph)
# ──────────────────────────────────────────────
# We lazy-init Graphiti so the server starts fast.
# All queries now go through Graphiti → FalkorDB.

_graphiti = None

def get_graphiti():
    global _graphiti
    if _graphiti is None:
        from graphiti_core import Graphiti
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.llm_client import OpenAIClient, LLMConfig
        from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

        _graphiti = Graphiti(
            llm_client = OpenAIClient(config=LLMConfig(
                api_key  = os.getenv("NVIDIA_API_KEY"),
                model    = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.1-8b-instruct"),
                base_url = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
            )),
            embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
                api_key         = os.getenv("NVIDIA_API_KEY"),
                embedding_model = os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
                base_url        = os.getenv("NIM_BASE_URL",    "https://integrate.api.nvidia.com/v1"),
                embedding_dim   = 1024,
            )),
            graph_driver = FalkorDriver(
                host     = os.getenv("FALKORDB_HOST", "localhost"),
                port     = int(os.getenv("FALKORDB_PORT", "6379")),
                database = "threat_intel_edu",
            ),
        )
    return _graphiti


# ──────────────────────────────────────────────
# 3.  DEFINE TOOLS  (what the agent can call)
# ──────────────────────────────────────────────
# Each tool is a regular Python function decorated with @mcp.tool()
# The docstring becomes the tool description the agent sees.
# Type hints become the tool's parameter schema.

@mcp.tool()
def lookup_ip(ip_address: str) -> str:
    """
    Look up threat intelligence for a specific IP address.
    Searches the Graphiti knowledge graph (FalkorDB) for all facts
    and relationships involving this IP.

    Args:
        ip_address: The IP address to investigate (e.g. '10.0.0.45')
    """
    graphiti = get_graphiti()

    # Run async search in sync context for MCP tool
    async def _search():
        from graphiti_core.search.search_config import (
            SearchConfig, EdgeSearchConfig, EdgeSearchMethod, EdgeReranker,
            NodeSearchConfig, NodeSearchMethod, NodeReranker,
        )
        config = SearchConfig(
            node_config = NodeSearchConfig(
                search_methods = [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                reranker       = NodeReranker.rrf,
            ),
            edge_config = EdgeSearchConfig(
                search_methods = [EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker       = EdgeReranker.rrf,
            ),
            limit = 10,
        )
        return await graphiti.search_(query=ip_address, config=config)

    results = asyncio.get_event_loop().run_until_complete(_search())

    facts = [e.fact for e in results.edges if ip_address in e.fact]
    nodes = [{"name": n.name, "summary": n.summary} for n in results.nodes]

    return json.dumps({
        "ip"            : ip_address,
        "related_facts" : facts[:10],
        "related_nodes" : nodes[:10],
        "fact_count"    : len(facts),
        "status"        : "found" if facts or nodes else "no_data",
    }, indent=2)


@mcp.tool()
def check_port_scan(ip_address: str) -> str:
    """
    Check whether an IP address performed port scanning activity.
    Queries Graphiti for Nmap and port scanning facts related to this IP.

    Args:
        ip_address: The IP to check for port scanning behaviour
    """
    graphiti = get_graphiti()

    async def _search():
        return await graphiti.search(
            query       = f"{ip_address} port scan nmap scanning",
            num_results = 10,
        )

    edges = asyncio.get_event_loop().run_until_complete(_search())

    scan_facts = [
        e.fact for e in edges
        if any(kw in e.fact.lower() for kw in ["nmap", "scan", "port", ip_address])
    ]

    return json.dumps({
        "ip"                 : ip_address,
        "port_scan_detected" : len(scan_facts) > 0,
        "evidence"           : scan_facts[:5],
        "confidence"         : "high" if len(scan_facts) >= 2 else ("medium" if scan_facts else "low"),
    }, indent=2)


@mcp.tool()
def get_attack_summary(attack_type: str) -> str:
    """
    Get a summary of all facts related to a specific attack type.
    Examples: 'sql injection', 'port scan', 'xss', 'data exfiltration'

    Args:
        attack_type: The attack category to summarise
    """
    graphiti = get_graphiti()

    async def _search():
        return await graphiti.search(
            query       = f"{attack_type} attack malicious blocked",
            num_results = 10,
        )

    edges = asyncio.get_event_loop().run_until_complete(_search())

    return json.dumps({
        "attack_type" : attack_type,
        "facts_found" : len(edges),
        "facts"       : [
            {"fact": e.fact, "timestamp": str(e.valid_at)[:10] if e.valid_at else "N/A"}
            for e in edges[:10]
        ],
        "summary": f"Found {len(edges)} graph facts related to {attack_type}",
    }, indent=2)


@mcp.tool()
def search_logs(keyword: str) -> str:
    """
    Semantic search through the threat intelligence graph.
    Uses both keyword (BM25) and vector similarity against FalkorDB.

    Args:
        keyword: Natural language query (e.g. 'blocked malicious SQL', '10.0.0.45 attack')
    """
    graphiti = get_graphiti()

    async def _search():
        return await graphiti.search(query=keyword, num_results=10)

    edges = asyncio.get_event_loop().run_until_complete(_search())

    return json.dumps({
        "keyword"     : keyword,
        "match_count" : len(edges),
        "results"     : [
            {
                "fact"      : e.fact,
                "timestamp" : str(e.valid_at)[:10] if e.valid_at else "N/A",
            }
            for e in edges
        ],
    }, indent=2)


@mcp.tool()
def list_all_malicious_ips() -> str:
    """
    Search for all malicious IP-related facts in the threat graph.
    Uses Graphiti semantic search to find malicious activity.
    """
    graphiti = get_graphiti()

    async def _search():
        return await graphiti.search(
            query       = "malicious IP address attack threat blocked external",
            num_results = 15,
        )

    edges = asyncio.get_event_loop().run_until_complete(_search())

    return json.dumps({
        "total_facts_found" : len(edges),
        "malicious_facts"   : [
            {"fact": e.fact, "timestamp": str(e.valid_at)[:10] if e.valid_at else "N/A"}
            for e in edges
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2)


# ──────────────────────────────────────────────
# ENTRY POINT — Run the MCP Server
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("━" * 50)
    print("  STEP 4 — MCP Threat Intel Server")
    print("━" * 50)
    print()
    print("  Tools exposed:")
    print("    • lookup_ip(ip_address)")
    print("    • check_port_scan(ip_address)")
    print("    • get_attack_summary(attack_type)")
    print("    • search_logs(keyword, field)")
    print("    • list_all_malicious_ips()")
    print()
    print("  ✓ Server starting on stdio transport...")
    print("  ✓ Run step5_react_agent.py in another terminal")
    print()

    # stdio transport = the agent talks to this server over stdin/stdout
    # This is the standard MCP transport for local tools.
    mcp.run(transport="stdio")