"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — LangGraph ReAct Agent with MCP Tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT YOU WILL LEARN:
  ✦ What the ReAct pattern is
  ✦ How LangGraph implements it as a state machine
  ✦ How to connect MCP tools to a LangGraph agent
  ✦ How the agent decides WHICH tool to call and WHEN
  ✦ How the agent loop terminates

CONCEPT — ReAct (Reason + Act):
  ReAct is a prompting strategy where the LLM alternates between:
    REASON  →  think about what to do next
    ACT     →  call a tool
    OBSERVE →  read the tool result
    REASON  →  think about what to do next (repeat...)
    ANSWER  →  respond when confident

  LangGraph implements this as a graph with two nodes:
    ┌─────────────────────────────────────────┐
    │            START                        │
    │              │                          │
    │              ▼                          │
    │         ┌─────────┐                     │
    │    ┌──  │  agent  │  ──┐ (has tool_call)│
    │    │    └─────────┘    │                │
    │    │ (no tool call =   │                │
    │    │  final answer)    ▼                │
    │    │             ┌──────────┐           │
    │    │             │  tools   │           │
    │    │             └──────────┘           │
    │    │                  │                 │
    │    │    ┌─────────────┘                 │
    │    │    │  (tool result added to state) │
    │    │    ▼                               │
    │    │  back to agent                     │
    │    │                                    │
    │    ▼                                    │
    │   END                                   │
    └─────────────────────────────────────────┘

  STATE: The full conversation history (messages list) flows through
         every node. Each tool result is appended as a ToolMessage.

HOW TO RUN:
  1. First start the MCP server:  python step4_mcp_server.py
  2. Then in another terminal:    python step5_react_agent.py

  OR: the agent can also run in-process (no separate server needed)
  if you use the inline tool mode shown at the bottom.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
# LangGraph imports
from langgraph.prebuilt import create_react_agent   # Pre-built ReAct graph

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# MCP adapter — bridges MCP tools into LangChain tool format
from langchain_mcp_adapters.client import MultiServerMCPClient

# Add this right below it to silence the false deprecation warning:
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

load_dotenv()
console = Console()


# ──────────────────────────────────────────────
# 1.  LLM SETUP
# ──────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model       = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.3-70b-instruct"),
        base_url    = os.getenv("NIM_BASE_URL",   "https://integrate.api.nvidia.com/v1"),
        api_key     = os.getenv("NVIDIA_API_KEY"),
        temperature = 0,
    )


# ──────────────────────────────────────────────
# 2.  SYSTEM PROMPT  (the agent's persona)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are HAL-O, an expert AI Security Operations Center (SOC) analyst.

Your job is to investigate network security events using the threat intelligence tools available to you.

WORKFLOW:
1. Understand the analyst's question
2. Use tools to gather evidence (call multiple tools if needed)
3. Synthesise the evidence into a clear finding
4. Always state your confidence level and reasoning

TOOL STRATEGY:
- Start broad: list_all_malicious_ips() to get the landscape
- Then drill down: lookup_ip() for specific IPs
- Verify: check_port_scan() for scanner IPs
- Cross-reference: get_attack_summary() for attack patterns
- Evidence: search_logs() for specific keywords

Be concise, factual, and always cite which tool gave you which finding."""


# ──────────────────────────────────────────────
# 3.  MCP → LangChain TOOL BRIDGE
# ──────────────────────────────────────────────
# MultiServerMCPClient connects to our MCP server(s) and converts
# the MCP tool definitions into LangChain Tool objects that LangGraph
# can bind to the LLM via tool_calling.

async def get_mcp_tools():
    client = MultiServerMCPClient({
        "threat_intel": {
            "command"  : "python",
            "args"     : ["step4_mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    console.print(f"[green]✓ Loaded {len(tools)} MCP tools:[/]")
    for tool in tools:
        console.print(f"  [cyan]• {tool.name}[/]: {tool.description[:60]}...")
    return tools   # no longer returning client


# ──────────────────────────────────────────────
# 4.  LANGGRAPH REACT AGENT
# ──────────────────────────────────────────────

async def run_agent(question: str):
    console.print(Rule("[bold blue]ReAct Agent[/]"))
    console.print(f"[bold]Question:[/] {question}\n")

    tools = await get_mcp_tools()   # just tools now
    llm   = get_llm()

    agent = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SYSTEM_PROMPT,
    )

    step_num = 0
    final_answer = ""

    async for step in agent.astream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode = "values",
    ):
        messages = step["messages"]
        last_msg = messages[-1]
        msg_type = type(last_msg).__name__

        if msg_type == "AIMessage":
            step_num += 1
            if last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    console.print(Panel(
                        f"[yellow]Tool:[/] {tc['name']}\n"
                        f"[yellow]Args:[/] {tc['args']}",
                        title=f"[bold yellow]🔧 REASON+ACT — Step {step_num}[/]",
                        border_style="yellow",
                    ))
            else:
                final_answer = last_msg.content
                console.print(Panel(
                    final_answer,
                    title="[bold green]✅ Final Answer[/]",
                    border_style="green",
                ))

        elif msg_type == "ToolMessage":
            preview = last_msg.content[:300] + "..." if len(last_msg.content) > 300 else last_msg.content
            console.print(Panel(
                f"[dim]{preview}[/]",
                title=f"[bold cyan]👁 OBSERVE — {last_msg.name}[/]",
                border_style="cyan",
            ))

    return final_answer


# ──────────────────────────────────────────────
# 5.  DEMO SCENARIOS
# ──────────────────────────────────────────────

DEMO_QUESTIONS = [
    # Scenario 1: Broad threat hunt
    "Give me a threat landscape overview. What malicious IPs are in our dataset and what did they do?",

    # Scenario 2: Specific IP investigation
    "Investigate IP 192.168.1.86. Is it a threat? What did it do?",

    # Scenario 3: Attack-type analysis
    "Were there any SQL injection attacks? Who was targeted and was it stopped?",
]


async def main():
    console.print(Panel(
        "[bold]STEP 5 — LangGraph ReAct Agent + MCP[/]\n"
        "Concepts: ReAct loop, State machine, MCP tool calling\n"
        "Models:   NIM Chat (reasoning)\n"
        "Tools:    MCP ThreatIntel Server (Step 4)",
        title="🤖 Threat Intel EDU", border_style="blue"
    ))

    # Run one demo question (change index to try others)
    question = DEMO_QUESTIONS[2]
    await run_agent(question)

    console.print("\n[bold]📚 What just happened?[/]")
    console.print("  1. LangGraph wired up a ReAct state machine")
    console.print("  2. The LLM received the question + tool descriptions")
    console.print("  3. The LLM decided which tools to call (REASON)")
    console.print("  4. LangGraph executed the tool via MCP (ACT)")
    console.print("  5. The tool result was appended to the message state (OBSERVE)")
    console.print("  6. Steps 3-5 repeated until the LLM had enough info")
    console.print("  7. The LLM produced a final answer without any tool call (END)")
    console.print("\n[dim]→ Run step6_full_pipeline.py to see everything together[/]")


if __name__ == "__main__":
    asyncio.run(main())