"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Knowledge Graph: Entities & Relationships
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ Turn Pydantic entities into graph nodes
  ✦ Create typed, weighted edges (relationships)
  ✦ Use NetworkX as a lightweight local graph store
  ✦ Why graphs beat flat tables for threat intelligence

CONCEPT — Knowledge Graph:
  Instead of rows in a table, we build:

      [IP: 10.0.0.45] ─── CONNECTED_TO ──→ [IP: 192.168.1.1]
            │                                       │
       (malicious,                           (dest, internal)
        sqlmap, blocked)
            │
       USES_PROTOCOL ──→ [Protocol: TCP]
            │
       TRIGGERED ──→ [AttackType: sql-injection]

  This lets us ask: "Show me ALL IPs connected to 192.168.1.1
  that used sqlmap" — impossible with flat SQL joins.
"""

import json
import os
import pandas as pd
import networkx as nx
from pyvis.network import Network
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

console = Console()

# ──────────────────────────────────────────────
# 1.  NODE & EDGE TYPE CONSTANTS
#     Using strings so the graph is readable as JSON
# ──────────────────────────────────────────────

# Node types
NODE_IP       = "IP"
NODE_PROTOCOL = "Protocol"
NODE_ATTACK   = "AttackType"
NODE_AGENT    = "UserAgent"

# Edge (relationship) types
REL_CONNECTED   = "CONNECTED_TO"     # src_ip → dest_ip
REL_USES_PROTO  = "USES_PROTOCOL"    # ip → protocol
REL_TRIGGERED   = "TRIGGERED"        # ip → attack type
REL_USED_AGENT  = "USED_AGENT"       # ip → user agent


# ──────────────────────────────────────────────
# 2.  GRAPH BUILDER
# ──────────────────────────────────────────────

class ThreatGraph:
    """
    A thin wrapper around NetworkX's DiGraph.
    DiGraph = Directed Graph (edges have direction, like src → dst).
    """

    def __init__(self):
        # DiGraph supports: node attributes, edge attributes, direction
        self.G = nx.DiGraph()
        self._event_count = 0
        self.net = Network(notebook=True)

    # ── Node helpers ──────────────────────────

    def _add_ip_node(self, address: str, **attrs):
        """
        Add or update an IP node.
        If the node already exists, attributes are merged (not replaced).
        This is key — same IP appears in many log rows.
        """
        if self.G.has_node(address):
            # Update existing node — keep accumulated context
            self.G.nodes[address].update(attrs)
        else:
            self.G.add_node(address, type=NODE_IP, **attrs)

    def _add_simple_node(self, node_id: str, node_type: str):
        if not self.G.has_node(node_id):
            self.G.add_node(node_id, type=node_type, label=node_id)

    # ── Ingest from extracted events (Step 1 output) ──

    def ingest_from_events(self, events: list[dict]):
        """Build the graph from the Pydantic-extracted event dicts."""
        for event in events:
            self._ingest_event(event)
        console.print(f"[green]✓ Ingested {len(events)} events[/]")
        self._print_stats()

    def _ingest_event(self, e: dict):
        self._event_count += 1
        src_ip   = e["source"]["address"]
        dst_ip   = e["destination"]["address"]
        protocol = e["protocol"]
        attack   = e.get("attack_type", "unknown")
        agent    = e.get("user_agent", "")   # raw CSV field kept for context

        # ── Add IP nodes ──────────────────────────
        self._add_ip_node(
            src_ip,
            is_internal  = e["source"]["is_internal"],
            is_malicious = e["threat_label"] == "malicious",
            role         = "source",
            suspicion    = e["source"].get("suspicion_reason", ""),
        )
        self._add_ip_node(
            dst_ip,
            is_internal  = e["destination"]["is_internal"],
            role         = "destination",
        )

        # ── Add protocol/attack nodes ─────────────
        self._add_simple_node(protocol, NODE_PROTOCOL)
        self._add_simple_node(attack,   NODE_ATTACK)

        # ── Add relationships (edges) ──────────────
        # Edge attributes carry the full event context.
        # This is what makes graph queries so powerful.
        edge_attrs = {
            "rel_type"   : REL_CONNECTED,
            "action"     : e["action"],
            "threat"     : e["threat_label"],
            "severity"   : e["severity"],
            "bytes"      : e["bytes_transferred"],
            "timestamp"  : e["timestamp"],
            "summary"    : e["summary"],
            "event_id"   : self._event_count,
        }

        # RELATIONSHIP 1: src_ip → dst_ip  (the core connection)
        if self.G.has_edge(src_ip, dst_ip):
            # Edge already exists → increment connection count
            self.G[src_ip][dst_ip]["count"] = self.G[src_ip][dst_ip].get("count", 1) + 1
        else:
            self.G.add_edge(src_ip, dst_ip, count=1, **edge_attrs)

        # RELATIONSHIP 2: src_ip → protocol
        if not self.G.has_edge(src_ip, protocol):
            self.G.add_edge(src_ip, protocol, rel_type=REL_USES_PROTO)

        # RELATIONSHIP 3: src_ip → attack type
        if attack != "unknown" and not self.G.has_edge(src_ip, attack):
            self.G.add_edge(src_ip, attack, rel_type=REL_TRIGGERED)

    # ── Ingest directly from raw CSV (no LLM needed) ──

    def ingest_from_csv(self, csv_path: str):
        """
        Faster path: build the graph directly from the CSV.
        Useful for visualising the full dataset without API costs.
        """
        df = pd.read_csv(csv_path)
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            event = {
                "source"           : {"address": row.source_ip, "is_internal": _is_internal(row.source_ip), "suspicion_reason": ""},
                "destination"      : {"address": row.dest_ip,   "is_internal": _is_internal(row.dest_ip)},
                "protocol"         : row.protocol,
                "action"           : row.action,
                "threat_label"     : row.threat_label,
                "severity"         : _severity_from_threat(row.threat_label, row.action),
                "bytes_transferred": row.bytes_transferred,
                "timestamp"        : row.timestamp,
                "summary"          : f"{row.source_ip} → {row.dest_ip} via {row.protocol} ({row.action})",
                "attack_type"      : _guess_attack(row.user_agent, row.request_path),
                "user_agent"       : row.user_agent,
            }
            self._ingest_event(event)
        console.print(f"[green]✓ Built graph from {len(df)} CSV rows[/]")
        self._print_stats()

    # ── Query helpers ─────────────────────────

    def neighbors_of(self, ip: str) -> list[str]:
        """Return all IPs that have ever communicated with this IP."""
        return list(self.G.predecessors(ip)) + list(self.G.successors(ip))

    def malicious_ips(self) -> list[str]:
        """Return all IP nodes flagged as malicious."""
        return [
            n for n, d in self.G.nodes(data=True)
            if d.get("type") == NODE_IP and d.get("is_malicious", False)
        ]

    def attack_types_from(self, ip: str) -> list[str]:
        """Return attack types triggered by this IP."""
        return [
            dst for dst in self.G.successors(ip)
            if self.G.nodes[dst].get("type") == NODE_ATTACK
        ]

    def connections_between(self, src: str, dst: str) -> list[dict]:
        """All edge attributes for connections from src to dst."""
        if self.G.has_edge(src, dst):
            return [self.G[src][dst]]
        return []

    def get_ip_context(self, ip: str) -> str:
        """
        Build a text summary of an IP's graph neighbourhood.
        This text is what GraphRAG feeds into the LLM as context.
        """
        lines = [f"IP Address: {ip}"]
        node_data = self.G.nodes.get(ip, {})
        lines.append(f"  Internal: {node_data.get('is_internal', '?')}")
        lines.append(f"  Malicious: {node_data.get('is_malicious', '?')}")

        for dst in self.G.successors(ip):
            edge = self.G[ip][dst]
            lines.append(
                f"  → {dst} [{edge.get('rel_type', 'LINKED')}] "
                f"action={edge.get('action','?')} threat={edge.get('threat','?')} "
                f"severity={edge.get('severity','?')}"
            )
        for src in self.G.predecessors(ip):
            edge = self.G[src][ip]
            lines.append(f"  ← {src} connected to this IP")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────

    def save(self, path: str = "data/threat_graph.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = nx.node_link_data(self.G)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[bold green]✓ Graph saved → {path}[/]")

    @classmethod
    def load(cls, path: str = "data/threat_graph.json") -> "ThreatGraph":
        tg = cls()
        with open(path) as f:
            data = json.load(f)
        tg.G = nx.node_link_graph(data)
        return tg

    # ── Display ───────────────────────────────

    def _print_stats(self):
        table = Table(title="Graph Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value",  style="green")
        table.add_row("Total nodes", str(self.G.number_of_nodes()))
        table.add_row("Total edges", str(self.G.number_of_edges()))
        table.add_row("IP nodes",      str(sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == NODE_IP)))
        table.add_row("Protocol nodes",str(sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == NODE_PROTOCOL)))
        table.add_row("Attack nodes",  str(sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == NODE_ATTACK)))
        console.print(table)

    def print_sample_queries(self):
        console.print(Panel("[bold]Sample Graph Queries[/]", border_style="yellow"))

        malicious = self.malicious_ips()
        console.print(f"\n[yellow]Q1: Which IPs are malicious?[/]")
        for ip in malicious[:5]:
            attacks = self.attack_types_from(ip)
            console.print(f"  [red]{ip}[/] → attacks: {attacks}")

        console.print(f"\n[yellow]Q2: Who is talking to 192.168.1.1?[/]")
        neighbors = self.neighbors_of("192.168.1.1")
        console.print(f"  Neighbors: {neighbors}")

        if malicious:
            ip = malicious[0]
            console.print(f"\n[yellow]Q3: Graph context for {ip} (used in GraphRAG)[/]")
            console.print(self.get_ip_context(ip))


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def _is_internal(ip: str) -> bool:
    """Simple RFC-1918 check without extra dependencies."""
    return (
        ip.startswith("192.168.") or
        ip.startswith("10.")      or
        ip.startswith("172.16.")
    )

def _severity_from_threat(threat: str, action: str) -> str:
    if threat == "malicious" and action == "blocked": return "high"
    if threat == "malicious" and action == "allowed": return "critical"
    if threat == "benign"    and action == "blocked": return "low"
    return "low"

def _guess_attack(user_agent: str, path: str) -> str:
    ua   = str(user_agent).lower()
    path = str(path).lower()
    if "sqlmap"  in ua or "id="   in path: return "sql-injection"
    if "nmap"    in ua:                     return "port-scan"
    if "<script" in path:                   return "xss"
    if "export"  in path or "api" in path:  return "data-exfiltration"
    if "curl"    in ua:                     return "c2-beacon"
    return "benign-traffic"


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    console.print(Panel(
        "[bold]STEP 2 — Knowledge Graph Builder[/]\n"
        "Concept: Entities as nodes, relationships as edges\n"
        "Engine:  NetworkX DiGraph (local, no database needed)",
        title="🕸  Threat Intel EDU", border_style="blue"
    ))

    tg = ThreatGraph()

    # Option A: build from raw CSV (fast, no LLM cost)
    #tg.ingest_from_csv(os.getenv("DATA_PATH", "data/cybersecurity_threat_detection_logs.csv"))

    # Option B (commented out): use Step 1's LLM-enriched events
    with open("data/extracted_events.json") as f:
        events = json.load(f)
    tg.ingest_from_events(events)

    tg.net.from_nx(tg.G)
    tg.net.show("graph.html")

    #tg.print_sample_queries()
    tg.save()

    console.print("\n[bold]📚 What just happened?[/]")
    console.print("  1. Each log row became edges + nodes in a directed graph")
    console.print("  2. Same IP appearing in multiple rows → single node with merged attrs")
    console.print("  3. get_ip_context() builds text summaries from the graph")
    console.print("  4. Those text summaries become RAG context in Step 3")
    console.print("\n[dim]→ Run step3_graph_rag.py next[/]")