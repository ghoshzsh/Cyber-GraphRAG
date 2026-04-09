"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Entity Extraction with Pydantic + LangChain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ Define domain models as Pydantic schemas
  ✦ Ask an LLM to extract structured data from raw log text
  ✦ LangChain's  with_structured_output()  pattern
  ✦ How NIM's OpenAI-compatible endpoint plugs into LangChain

CONCEPT — Structured Output:
  Raw log row  →  LLM (with JSON schema)  →  Pydantic object
  The LLM is forced to return JSON that matches your schema.
  No regex parsing. No fragile string splitting.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

# LangChain imports
from langchain_openai import ChatOpenAI          # NIM is OpenAI-compatible
from langchain_core.prompts import ChatPromptTemplate

# Pydantic — our entity schemas
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
console = Console()

# ──────────────────────────────────────────────
# 1.  DEFINE YOUR ENTITIES AS PYDANTIC MODELS
# ──────────────────────────────────────────────
# Pydantic acts as both:
#   (a) runtime validation  — bad data raises an error
#   (b) JSON schema source  — LangChain sends this schema to the LLM

class IPEntity(BaseModel):
    """Represents a single IP address observed in a log."""
    address: str            = Field(description="The IP address string e.g. 192.168.1.1")
    role: Literal["source", "destination"] = Field(description="Is this IP the source or destination?")
    is_internal: bool       = Field(description="True if this is a private/RFC-1918 address")
    suspicion_reason: str   = Field(description="One sentence why this IP is or isn't suspicious", default="")


class ThreatEvent(BaseModel):
    """Structured representation of a single security log entry."""
    # Core identity
    timestamp: str          = Field(description="ISO timestamp from the log")
    protocol: str           = Field(description="Network protocol: TCP, UDP, HTTP, ICMP, etc.")
    action: Literal["allowed", "blocked"] = Field(description="Firewall/app action taken")
    threat_label: Literal["benign", "malicious"] = Field(description="Ground-truth threat label")
    log_type: str           = Field(description="Origin of the log: firewall or application")
    bytes_transferred: int  = Field(description="Number of bytes transferred")

    # Extracted entities (nested Pydantic models)
    source: IPEntity        = Field(description="Source IP entity")
    destination: IPEntity   = Field(description="Destination IP entity")

    # LLM-generated intelligence
    attack_type: str        = Field(
        description="Best guess at attack type: port-scan, sql-injection, xss, data-exfiltration, c2-beacon, benign-traffic etc.",
        default="unknown"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Assessed severity based on the full row context"
    )
    summary: str            = Field(description="One-sentence human-readable summary of this event")


# ──────────────────────────────────────────────
# 2.  CONNECT TO NVIDIA NIM
# ──────────────────────────────────────────────
# NIM exposes an OpenAI-compatible /v1/chat/completions endpoint.
# We just change base_url and api_key — the rest of LangChain stays identical.

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model        = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.3-70b-instruct"),
        base_url     = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
        api_key      = os.getenv("NVIDIA_API_KEY"),
        temperature  = 0,        # 0 = deterministic, good for extraction
        max_tokens   = 1024,
    )


# ──────────────────────────────────────────────
# 3.  BUILD THE EXTRACTION CHAIN
# ──────────────────────────────────────────────
# with_structured_output() does three things automatically:
#   1. Converts ThreatEvent to a JSON schema
#   2. Passes that schema to the LLM as a tool/function definition
#   3. Parses the LLM's JSON response back into a ThreatEvent object

def build_extraction_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a cybersecurity analyst.
Given a raw network log row, extract all entities and classify the event.
Be precise. Use only information present in the log."""),
        ("human", "Analyze this log row:\n\n{log_row}"),
    ])

    # KEY CONCEPT: with_structured_output() forces the LLM to return
    # JSON that matches ThreatEvent's schema. No hallucinated fields.
    chain = prompt | llm.with_structured_output(ThreatEvent)
    return chain


# ──────────────────────────────────────────────
# 4.  PROCESS THE CSV
# ──────────────────────────────────────────────

def row_to_text(row: pd.Series) -> str:
    """Convert a DataFrame row to a natural-language log string for the LLM."""
    return (
        f"Timestamp: {row['timestamp']} | "
        f"Source IP: {row['source_ip']} → Dest IP: {row['dest_ip']} | "
        f"Protocol: {row['protocol']} | Action: {row['action']} | "
        f"Threat: {row['threat_label']} | Log type: {row['log_type']} | "
        f"Bytes: {row['bytes_transferred']} | "
        f"User-Agent: {row['user_agent']} | Path: {row['request_path']}"
    )


def extract_entities(csv_path: str, max_rows: int = 3) -> list[ThreatEvent]:
    """
    Run entity extraction on the first N rows of the dataset.
    We limit to 3 rows for the educational demo to keep API costs low.
    """
    df = pd.read_csv(csv_path)
    chain = build_extraction_chain()
    events: list[ThreatEvent] = []

    console.print(Panel(f"[bold cyan]Extracting entities from {min(max_rows, len(df))} log rows[/]"))

    for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
        log_text = row_to_text(row)
        console.print(f"\n[yellow]▶ Row {i+1}:[/] {log_text[:80]}...")

        # This is the LangChain "invoke" call — one API round-trip
        event: ThreatEvent = chain.invoke({"log_row": log_text})
        events.append(event)

        # Pretty-print the extracted Pydantic object
        console.print(Panel(
            f"[green]attack_type:[/] {event.attack_type}\n"
            f"[green]severity:[/]    {event.severity}\n"
            f"[green]summary:[/]     {event.summary}\n"
            f"[green]source:[/]      {event.source.address} (internal={event.source.is_internal})\n"
            f"[green]dest:[/]        {event.destination.address}",
            title=f"[bold]ThreatEvent #{i+1}[/]",
            border_style="green",
        ))

    return events


# ──────────────────────────────────────────────
# 5.  SAVE FOR THE NEXT STEP
# ──────────────────────────────────────────────

def save_events(events: list[ThreatEvent], path: str = "data/extracted_events.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # Pydantic v2: model.model_dump() → dict
        json.dump([e.model_dump() for e in events], f, indent=2)
    console.print(f"\n[bold green]✓ Saved {len(events)} events → {path}[/]")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel(
        "[bold]STEP 1 — Entity Extraction[/]\n"
        "Concept: Pydantic schemas + LangChain structured output\n"
        "Model:   NIM Chat (LLaMA 3.1)",
        title="🔍 Threat Intel EDU", border_style="blue"
    ))

    events = extract_entities(
        csv_path = os.getenv("DATA_PATH", "data/cybersecurity_threat_detection_logs.csv"),
        max_rows = 25,   # ← increase this to process more rows
    )

    save_events(events)

    console.print("\n[bold]📚 What just happened?[/]")
    console.print("  1. Each CSV row was converted to a text description")
    console.print("  2. LangChain sent it to NIM with a JSON schema (ThreatEvent)")
    console.print("  3. NIM returned structured JSON — automatically validated by Pydantic")
    console.print("  4. You now have typed Python objects — not raw strings!")
    console.print("\n[dim]→ Run step2_graph_builder.py next[/]")