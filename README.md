# 🛡 Threat Intel EDU — Learning Project

An end-to-end **educational** project covering:
- **LangChain** — structured LLM calls, prompt templates
- **LangGraph** — ReAct-style agent loop
- **GraphRAG** — graph-augmented retrieval (basic)
- **Entity Extraction** — Pydantic schemas + `with_structured_output()`
- **Knowledge Graph** — relationships between entities (NetworkX)
- **MCP** — Model Context Protocol for tool serving

**Dataset:** Kaggle network security logs (15 sample rows, 10 columns)
**LLM Provider:** NVIDIA NIM (OpenAI-compatible API)

---

## Architecture

```
CSV Logs (Kaggle dataset)
        │
        ▼
┌───────────────────┐
│  STEP 1           │  LangChain + Pydantic
│  Entity           │  ─────────────────────
│  Extraction       │  ChatOpenAI (NIM Chat)
│                   │  with_structured_output(ThreatEvent)
│  Output:          │
│  extracted_events │
│  .json            │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  STEP 2           │  NetworkX DiGraph
│  Knowledge        │  ─────────────────────
│  Graph Builder    │  Nodes: IP, Protocol, AttackType
│                   │  Edges: CONNECTED_TO, USES_PROTOCOL,
│  Output:          │         TRIGGERED, USED_AGENT
│  threat_graph     │
│  .json            │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  STEP 3           │  Embed + Rerank + Generate
│  GraphRAG         │  ─────────────────────────
│  Engine           │  NIM Embed  → index graph nodes
│                   │  Cosine sim → find relevant nodes
│  Output:          │  Graph walk → expand neighbourhood
│  graph_rag_index  │  NIM Rerank → score each context piece
│  .json            │  NIM Chat   → answer from top context
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     stdio      ┌──────────────────┐
│  STEP 5           │ ◄────────────► │  STEP 4          │
│  LangGraph        │                │  MCP Server      │
│  ReAct Agent      │  Tool calls    │  (FastMCP)       │
│                   │                │                  │
│  NIM Chat model   │                │  Tools:          │
│  create_react_    │                │  • lookup_ip     │
│  agent()          │                │  • check_port_   │
│                   │                │    scan          │
│  ReAct loop:      │                │  • get_attack_   │
│  REASON → ACT     │                │    summary       │
│  OBSERVE → END    │                │  • search_logs   │
└───────────────────┘                │  • list_malicious│
                                     └──────────────────┘
```

## NVIDIA NIM Models Used

| Role | Model | Used In |
|------|--------|---------|
| Chat / Reasoning | `meta/llama-3.3-70b-instruct` | Steps 1, 3, 5 |
| Embedding | `qwen3-embedding:4b` - Ollama| Step 3 (GraphRAG index) |
| Reranker / Prioritization | `nvidia/llama-nemotron-rerank-vl-1b-v2` | Step 3 (context scoring) |

---

## Setup

### 1. Install dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the project
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
# Get it free at: https://build.nvidia.com/
```

### 3. Verify NIM access

```python
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model    = "meta/llama-3.1-8b-instruct",
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key  = os.getenv("NVIDIA_API_KEY"),
)
print(llm.invoke("hello").content)
```

---

## Running the Steps

### Option A — Run each step individually (recommended for learning)

```bash
# Step 1: Entity extraction from log rows
python step1_entity_extraction.py

# Step 2: Build knowledge graph
python step2_graph_builder.py

# Step 3: GraphRAG — embed, retrieve, rerank, generate
python step3_graph_rag.py

# Step 4: Start MCP server (keep running)
python step4_mcp_server.py

# Step 5: ReAct agent (in a new terminal)
python step5_react_agent.py

# Step 6: Full pipeline end-to-end
python step6_full_pipeline.py
```

### Option B — Full pipeline at once

```bash
python step6_full_pipeline.py
```

Results are cached in `data/` — delete the `.json` files to force a re-run.

---

## Key Concepts Explained

### 1. Structured Output (Step 1)
```python
chain = prompt | llm.with_structured_output(ThreatEvent)
event = chain.invoke({"log_row": "..."})  # Returns a ThreatEvent, not a string!
```
The LLM is forced to return JSON matching your Pydantic schema. No parsing, no guessing.

### 2. Knowledge Graph (Step 2)
```python
G = nx.DiGraph()
G.add_node("10.0.0.45", type="IP", is_malicious=True)
G.add_edge("10.0.0.45", "192.168.1.1", rel_type="CONNECTED_TO", action="blocked")
```
Same IP across many log rows → one node with accumulated attributes. Queries become graph traversals.

### 3. GraphRAG (Step 3)
```
Plain RAG:   query → embed → find similar CHUNKS → answer
GraphRAG:    query → embed → find similar NODES → walk graph → expand context → rerank → answer
```
The graph gives you RELATIONSHIPS between evidence. Multi-hop context impossible with flat search.

### 4. MCP (Step 4)
```python
@mcp.tool()
def lookup_ip(ip_address: str) -> str:
    """Look up threat intel for an IP address."""
    ...
```
Decorate a function → it becomes an MCP tool. The agent discovers it at runtime, no hard-coding.

### 5. ReAct Loop (Step 5)
```
LangGraph state machine:
  agent_node → (tool_call?) → tools_node → agent_node → ... → END

State = messages list. Each tool result is appended as ToolMessage.
The loop runs until the LLM produces a response with no tool call.
```

---

## Data Flow: One Log Row Through the Whole System

```
Raw:    "2024-06-12 | 10.0.0.45 → 192.168.1.1 | TCP blocked malicious | sqlmap"
          │
          ▼ Step 1 (LangChain structured output)
Pydantic: ThreatEvent(
            source=IPEntity(address="10.0.0.45", is_internal=False),
            attack_type="sql-injection", severity="high",
            summary="External IP attempted SQL injection, blocked"
          )
          │
          ▼ Step 2 (NetworkX)
Graph:    [10.0.0.45] ──CONNECTED_TO──► [192.168.1.1]
          [10.0.0.45] ──TRIGGERED──────► [sql-injection]
          [10.0.0.45] ──USES_PROTOCOL──► [TCP]
          │
          ▼ Step 3 (GraphRAG)
Embed:    "IP Address: 10.0.0.45 | is_internal: False | ..."  → [0.12, -0.34, ...]
          │
          ▼ Step 5 (ReAct Agent via MCP)
Agent:    "lookup_ip('10.0.0.45')" → tool result → "get_attack_summary('sql-injection')"
          → Final answer: "10.0.0.45 is an external attacker performing SQL injection..."
```

---

## File Structure

```
threat-intel-edu/
├── .env.example                ← copy to .env and add your API key
├── pyproject.toml              ← minimal dependencies
├── data/
│   ├── sample_logs.csv         ← 15 rows of network security logs
│   ├── extracted_events.json   ← Step 1 output (auto-generated)
│   ├── threat_graph.json       ← Step 2 output (auto-generated)
│   └── graph_rag_index.json    ← Step 3 output (auto-generated)
├── step1_entity_extraction.py  ← Pydantic + LangChain structured output
├── step2_graph_builder.py      ← Knowledge graph with NetworkX
├── step3_graph_rag.py          ← Embed + retrieve + rerank + generate
├── step4_mcp_server.py         ← FastMCP tool server
├── step5_react_agent.py        ← LangGraph ReAct agent
└── step6_full_pipeline.py      ← Everything chained together
```