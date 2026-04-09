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
# FastMCP wraps your Python functions and exposes them as MCP tools.
# The agent in Step 5 connects to this server and gets the tool list.

mcp = FastMCP(
    name        = "ThreatIntelMCP",
)


# ──────────────────────────────────────────────
# 2.  LOAD THE GRAPH (our "threat intel database")
# ──────────────────────────────────────────────
# We lazy-load the graph so the server starts fast.
# In production this would be a real SIEM or threat feed.

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        from step2_graph_builder import ThreatGraph
        graph_path = "data/threat_graph.json"
        if os.path.exists(graph_path):
            _graph = ThreatGraph.load(graph_path)
        else:
            # Build from CSV if no saved graph
            _graph = ThreatGraph()
            _graph.ingest_from_csv(os.getenv("DATA_PATH", "data/sample_logs.csv"))
    return _graph


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
    Returns the IP's known relationships, attack history, and risk profile
    from our internal threat graph.

    Args:
        ip_address: The IP address to investigate (e.g. '10.0.0.45')
    """
    graph = get_graph()

    if not graph.G.has_node(ip_address):
        return json.dumps({
            "ip": ip_address,
            "status": "unknown",
            "message": "This IP has no recorded activity in our dataset",
        })

    node_data = graph.G.nodes[ip_address]
    attacks   = graph.attack_types_from(ip_address)
    neighbors = graph.neighbors_of(ip_address)

    # Build outbound connection details
    connections = []
    for dst in graph.G.successors(ip_address):
        edge = graph.G[ip_address][dst]
        connections.append({
            "destination" : dst,
            "rel_type"    : edge.get("rel_type", "LINKED"),
            "action"      : edge.get("action", "?"),
            "threat"      : edge.get("threat", "?"),
            "severity"    : edge.get("severity", "?"),
            "bytes"       : edge.get("bytes", 0),
            "timestamp"   : edge.get("timestamp", "?"),
        })

    return json.dumps({
        "ip"           : ip_address,
        "is_internal"  : node_data.get("is_internal", False),
        "is_malicious" : node_data.get("is_malicious", False),
        "attack_types" : attacks,
        "total_connections": len(neighbors),
        "outbound_connections": connections[:10],  # cap at 10 for readability
    }, indent=2)


@mcp.tool()
def check_port_scan(ip_address: str) -> str:
    """
    Check whether an IP address performed port scanning activity.
    Looks for Nmap signatures and sequential multi-host connections.

    Args:
        ip_address: The IP to check for port scanning behaviour
    """
    graph = get_graph()

    if not graph.G.has_node(ip_address):
        return json.dumps({"ip": ip_address, "port_scan_detected": False, "reason": "IP not in dataset"})

    attacks = graph.attack_types_from(ip_address)
    # Count how many distinct IPs this source has connected to
    unique_targets = list(graph.G.successors(ip_address))
    ip_targets = [
        t for t in unique_targets
        if graph.G.nodes[t].get("type") == "IP"
    ]

    is_scanner = "port-scan" in attacks or len(ip_targets) >= 3

    return json.dumps({
        "ip"                 : ip_address,
        "port_scan_detected" : is_scanner,
        "attack_types_found" : attacks,
        "unique_ip_targets"  : len(ip_targets),
        "targets"            : ip_targets[:10],
        "confidence"         : "high" if "port-scan" in attacks else ("medium" if len(ip_targets) >= 3 else "low"),
    }, indent=2)


@mcp.tool()
def get_attack_summary(attack_type: str) -> str:
    """
    Get a summary of all IPs associated with a specific attack type.
    Valid attack types: port-scan, sql-injection, xss, data-exfiltration, c2-beacon, benign-traffic

    Args:
        attack_type: The attack category to summarise
    """
    graph = get_graph()
    from step2_graph_builder import NODE_ATTACK, NODE_IP

    # Find the attack node in the graph
    if not graph.G.has_node(attack_type):
        return json.dumps({
            "attack_type": attack_type,
            "message"    : "No IPs found for this attack type in our dataset",
            "known_types": [n for n, d in graph.G.nodes(data=True) if d.get("type") == NODE_ATTACK],
        })

    # Walk backwards: who has an edge TO this attack node?
    attacker_ips = list(graph.G.predecessors(attack_type))

    details = []
    for ip in attacker_ips:
        node = graph.G.nodes.get(ip, {})
        if node.get("type") == NODE_IP:
            details.append({
                "ip"          : ip,
                "is_internal" : node.get("is_internal", False),
                "is_malicious": node.get("is_malicious", False),
            })

    return json.dumps({
        "attack_type"   : attack_type,
        "attacker_count": len(details),
        "attackers"     : details,
        "summary"       : f"{len(details)} IP(s) performed {attack_type} in our dataset",
    }, indent=2)


@mcp.tool()
def search_logs(keyword: str, field: str = "any") -> str:
    """
    Search through the threat intelligence graph for logs matching a keyword.
    Searches edge attributes (summaries, user agents, paths).

    Args:
        keyword: Text to search for (e.g. 'sqlmap', 'blocked', '192.168.1.1')
        field:   Which field to search: 'summary', 'action', 'threat', or 'any'
    """
    graph = get_graph()
    keyword_lower = keyword.lower()
    matches = []

    for src, dst, edge_data in graph.G.edges(data=True):
        # Only search edges that have summary/event info
        if "summary" not in edge_data:
            continue

        # Determine what to search
        if field == "summary":
            haystack = edge_data.get("summary", "")
        elif field == "action":
            haystack = edge_data.get("action", "")
        elif field == "threat":
            haystack = edge_data.get("threat", "")
        else:  # "any"
            haystack = " ".join(str(v) for v in edge_data.values())

        if keyword_lower in haystack.lower():
            matches.append({
                "source"   : src,
                "dest"     : dst,
                "action"   : edge_data.get("action"),
                "threat"   : edge_data.get("threat"),
                "severity" : edge_data.get("severity"),
                "timestamp": edge_data.get("timestamp"),
                "summary"  : edge_data.get("summary", ""),
            })

    return json.dumps({
        "keyword"     : keyword,
        "match_count" : len(matches),
        "results"     : matches[:20],  # cap at 20
    }, indent=2)


@mcp.tool()
def list_all_malicious_ips() -> str:
    """
    Return a list of all IP addresses flagged as malicious in our threat graph.
    Use this as a starting point for threat hunting.
    """
    graph = get_graph()
    malicious = graph.malicious_ips()

    result = []
    for ip in malicious:
        attacks = graph.attack_types_from(ip)
        result.append({
            "ip"          : ip,
            "attack_types": attacks,
            "is_internal" : graph.G.nodes[ip].get("is_internal", False),
        })

    return json.dumps({
        "total_malicious_ips": len(result),
        "ips"                : result,
        "generated_at"       : datetime.utcnow().isoformat() + "Z",
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