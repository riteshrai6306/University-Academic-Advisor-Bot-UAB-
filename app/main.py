# ================================================================
# main.py
# ================================================================
# WHAT: LangGraph Master Router — the brain of the UAB system
# WHY:  Connects all 3 agents (RAG, SQL, Web) into one smart graph
#       that automatically routes student questions to the right agent
#
# FLOW:
#   Student question → Router Node → decides agent
#   → RAG / SQL / Web Node → Answer Node → final response
#
# WHY LANGGRAPH?
#   - Built for agentic systems with conditional routing
#   - Manages state across all nodes automatically
#   - Clean graph structure — easy to add new agents later
#   - Each node is independent and testable separately
# ================================================================


# ── IMPORTS ─────────────────────────────────────────────────────

import os

# WHY: loads API keys from .env file
from dotenv import load_dotenv

# WHY: TypedDict defines our State structure
# WHY: State is the shared memory passed between all nodes
from typing import TypedDict

# WHY: StateGraph is the core LangGraph class
# WHY: END marks where the graph stops
from langgraph.graph import StateGraph, END

# WHY: GPT-4o-mini powers our router node
from langchain_openai import ChatOpenAI

# WHY: structures the router prompt
from langchain_core.prompts import PromptTemplate

# WHY: parses LLM output as clean string
from langchain_core.output_parsers import StrOutputParser

# WHY: import all 3 agents we built
from agents.rag_agent import run_rag_agent
from agents.sql_agent import run_sql_agent
from agents.web_agent import run_web_agent


# ── LOAD ENVIRONMENT VARIABLES ──────────────────────────────────

# WHY: must be called before any OpenAI usage
load_dotenv()


# ================================================================
# STEP 1 — DEFINE STATE
# ================================================================

class UABState(TypedDict):
    """
    WHAT: Shared state that flows through every node in the graph.
    WHY:  Every node reads from and writes to this state.
          It's like a baton passed in a relay race.

    FIELDS:
        question          → original student question (never changes)
        resolved_question → standalone question after follow-up rewriting
        history           → prior conversation history for context
        agent             → which agent router picked (sql/rag/web)
        answer            → final answer from the chosen agent
        sources           → source URLs or PDF names (for web/rag agents)
        error             → error message if something goes wrong
    """

    question : str   # WHY: the student's original question
    resolved_question : str  # WHY: rewritten standalone question for agents
    history  : str   # WHY: prior conversation history used for follow-ups
    agent    : str   # WHY: router's decision (sql / rag / web)
    answer   : str   # WHY: final answer to show the student
    sources  : list  # WHY: citations — PDFs used or URLs found
    error    : str   # WHY: error message if agent fails


# ================================================================
# STEP 2 — ROUTER NODE
# ================================================================

# WHY: router prompt tells LLM exactly how to classify questions
# WHY: we give clear examples so LLM makes the right decision
# WHY: output must be exactly "sql", "rag", or "web" — nothing else
ROUTER_PROMPT_TEMPLATE = """
You are a smart query router for a University Academic Advisor Bot.
Your job is to classify the student's question into exactly ONE category.

Categories:
- sql : questions about student data — CGPA, marks, attendance,
        toppers, rankings, department performance, pass/fail rates
        
- rag : questions about university policies, course information,
        prerequisites, syllabus, fees, scholarship eligibility rules,
        program details, attendance rules
        
- web : questions about live external information — scholarships,
        internships, job opportunities, fellowship programs,
        application deadlines, external competitions

Examples:
  "Who has the highest CGPA?" → sql
  "What is the attendance policy?" → rag
  "Any scholarships for CSE students?" → web
  "List students with marks above 90" → sql
  "What are prerequisites for M.Tech?" → rag
  "Internships for 3rd year students?" → web

Rules:
- Reply with ONLY one word: sql, rag, or web
- No explanation, no punctuation, just the word

Student Question: {question}

Category:"""

REWRITE_PROMPT_TEMPLATE = """
You are a conversation assistant that rewrites a student's follow-up question
into a standalone question using the prior dialog context.

History:
{history}

Follow-up Question: {question}

Rewrite this into a clear standalone question that contains all necessary context.
If the question is already standalone, repeat it exactly.
"""

REWRITE_PROMPT = PromptTemplate(
    template=REWRITE_PROMPT_TEMPLATE,
    input_variables=["history", "question"]
)

ROUTER_PROMPT = PromptTemplate(
    template=ROUTER_PROMPT_TEMPLATE,
    input_variables=["question"]
)


def rewrite_node(state: UABState) -> UABState:
    """
    WHAT: Rewrites follow-up questions into standalone questions.
    WHY:  This ensures the router and agents get full context.
    """

    print(f"\n✍️  REWRITE NODE")
    print(f"   Original question: {state['question']}")

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = REWRITE_PROMPT | llm | StrOutputParser()

        rewritten = chain.invoke({
            "history": state.get("history", ""),
            "question": state["question"]
        })

        rewritten = rewritten.strip()
        if not rewritten:
            rewritten = state["question"].strip()

        print(f"   Rewritten question: {rewritten}")

        return {
            **state,
            "resolved_question": rewritten
        }

    except Exception as e:
        print(f"   ❌ Rewrite error: {e}")
        return {
            **state,
            "resolved_question": state["question"]
        }


def router_node(state: UABState) -> UABState:
    """
    WHAT: First node — classifies student question into sql/rag/web.
    WHY:  Without routing, every question goes to every agent —
          wasteful and often wrong. Router sends it to the right one.

    HOW:
        1. Takes question from state
        2. Asks GPT to classify it
        3. Writes agent name back into state
        4. LangGraph reads state → decides which node to go to next

    ARGS:
        state (UABState): current graph state with question

    RETURNS:
        UABState: updated state with agent field set
    """

    print(f"\n🧭 ROUTER NODE")
    print(f"   Resolved question: {state['resolved_question']}")

    try:
        # WHY: temperature=0 = deterministic routing
        # WHY: we want consistent classification every time
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # WHY: chain = prompt → LLM → string parser
        chain = ROUTER_PROMPT | llm | StrOutputParser()

        # WHY: invoke chain with the resolved standalone question
        agent = chain.invoke({"question": state["resolved_question"]})

        # WHY: clean output — remove spaces, newlines, lowercase
        agent = agent.strip().lower()

        # WHY: validate output — if LLM returns something unexpected
        #      default to rag (safest general fallback)
        if agent not in ["sql", "rag", "web"]:
            print(f"⚠️  Unexpected agent '{agent}' — defaulting to rag")
            agent = "rag"

        print(f"   ✅ Routed to: {agent.upper()} Agent")

        # WHY: update state with routing decision
        return {
            **state,       # WHY: keep all existing state fields
            "agent": agent
        }

    except Exception as e:
        print(f"   ❌ Router error: {str(e)} — defaulting to rag")
        return {
            **state,
            "agent": "rag",  # WHY: safe fallback on router failure
            "error": str(e)
        }


# ================================================================
# STEP 3 — AGENT NODES
# ================================================================

def sql_node(state: UABState) -> UABState:
    """
    WHAT: Calls SQL agent with student's question.
    WHY:  Handles structured data questions about student records.

    ARGS:
        state (UABState): state with question

    RETURNS:
        UABState: state updated with answer from SQL agent
    """

    print(f"\n🗄️  SQL NODE")

    # WHY: call our sql_agent with the rewritten standalone question
    result = run_sql_agent(state["resolved_question"])

    return {
        **state,
        "answer":  result["answer"],
        "sources": [],  # WHY: SQL answers come from DB, no URLs
        "error":   result.get("error", "")
    }


def rag_node(state: UABState) -> UABState:
    """
    WHAT: Calls RAG agent with student's question.
    WHY:  Handles unstructured questions about policies and courses.

    ARGS:
        state (UABState): state with question

    RETURNS:
        UABState: state updated with answer + PDF sources
    """

    print(f"\n📄 RAG NODE")

    # WHY: call our rag_agent with the rewritten standalone question
    result = run_rag_agent(state["resolved_question"])

    return {
        **state,
        "answer":  result["answer"],
        "sources": result.get("sources", []),  # WHY: PDF filenames used
        "error":   ""
    }


def web_node(state: UABState) -> UABState:
    """
    WHAT: Calls Web agent with student's question.
    WHY:  Handles live questions about scholarships and internships.

    ARGS:
        state (UABState): state with question

    RETURNS:
        UABState: state updated with answer + source URLs
    """

    print(f"\n🌐 WEB NODE")

    # WHY: call our web_agent with the rewritten standalone question
    result = run_web_agent(state["resolved_question"])

    return {
        **state,
        "answer":  result["answer"],
        "sources": result.get("sources", []),  # WHY: web URLs found
        "error":   result.get("error", "")
    }


# ================================================================
# STEP 4 — CONDITIONAL EDGE (routing decision)
# ================================================================

def route_to_agent(state: UABState) -> str:
    """
    WHAT: Tells LangGraph which node to go to next.
    WHY:  This is the conditional edge — the fork in the road.
          LangGraph calls this function after router_node runs
          and uses the return value to pick the next node.

    ARGS:
        state (UABState): state with agent field set by router

    RETURNS:
        str: name of the next node ("sql_node"/"rag_node"/"web_node")
    """

    agent = state.get("agent", "rag")

    # WHY: map agent name → node name in the graph
    routing_map = {
        "sql": "sql_node",
        "rag": "rag_node",
        "web": "web_node"
    }

    next_node = routing_map.get(agent, "rag_node")
    print(f"\n🔀 Routing to: {next_node}")

    return next_node


# ================================================================
# STEP 5 — BUILD THE LANGGRAPH
# ================================================================

def build_graph() -> StateGraph:
    """
    WHAT: Assembles all nodes and edges into a LangGraph graph.
    WHY:  The graph defines the complete flow of our UAB system.
          Nodes = agents, Edges = connections between them.

    GRAPH STRUCTURE:
        START
          ↓
      rewrite_node
          ↓
      router_node
          ↓ (conditional edge)
        ┌───┴──────────┬─────────────┐
        ↓              ↓             ↓
      sql_node      rag_node      web_node
        ↓              ↓             ↓
        └──────────────┴─────────────┘
                       ↓
                      END

    RETURNS:
        compiled LangGraph graph ready to run
    """

    # WHY: StateGraph takes our state definition
    # WHY: every node in this graph reads/writes UABState
    graph = StateGraph(UABState)

    # ── ADD NODES ───────────────────────────────────────────────
    # WHY: each node is a function that takes state and returns state

    # WHY: rewrite_node — converts follow-up questions into standalone questions
    graph.add_node("rewrite_node", rewrite_node)

    # WHY: router_node — classifies the question
    graph.add_node("router_node", router_node)

    # WHY: agent nodes — each handles one type of question
    graph.add_node("sql_node", sql_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_node", web_node)

    # ── ADD EDGES ───────────────────────────────────────────────
    # WHY: edges define the flow between nodes

    # WHY: graph always starts at rewrite_node
    graph.set_entry_point("rewrite_node")

    # WHY: edge from rewrite_node to router_node
    graph.add_edge("rewrite_node", "router_node")

    # WHY: conditional edge — after router runs, call route_to_agent()
    #      to decide which agent node to go to next
    graph.add_conditional_edges(
        "router_node",      # WHY: from this node
        route_to_agent,     # WHY: call this function to decide next node
        {                   # WHY: map return values to actual node names
            "sql_node": "sql_node",
            "rag_node": "rag_node",
            "web_node": "web_node"
        }
    )

    # WHY: after each agent node runs → go to END
    # WHY: END tells LangGraph the graph is complete
    graph.add_edge("sql_node", END)
    graph.add_edge("rag_node", END)
    graph.add_edge("web_node", END)

    # WHY: compile() validates the graph and prepares it to run
    return graph.compile()


# ================================================================
# STEP 6 — MAIN RUN FUNCTION (called by app.py)
# ================================================================

# WHY: build graph once at module load — reused for every question
# WHY: avoids rebuilding graph on every single query (slow)
uab_graph = build_graph()


def run_uab(question: str, history: str = "") -> dict:
    """
    WHAT: Main entry point — runs the full UAB graph for a question.
    WHY:  app.py calls this function for every student question.
          It handles the complete flow from question to answer.

    ARGS:
        question (str): student's natural language question
        history  (str): previous conversation history for follow-up resolution

    RETURNS:
        dict:
            answer  (str)  → final answer to show student
            agent   (str)  → which agent answered (sql/rag/web)
            sources (list) → PDF names or URLs used
            error   (str)  → error message if something went wrong
    """

    print(f"\n{'='*55}")
    print(f"  UAB SYSTEM — Processing Question")
    print(f"{'='*55}")
    print(f"  Q: {question}")

    # WHY: initial state — question, history, and resolved_question are set
    initial_state: UABState = {
        "question": question,
        "resolved_question": question,
        "history": history or "",
        "agent":    "",    # WHY: router will fill this
        "answer":   "",    # WHY: agent node will fill this
        "sources":  [],    # WHY: agent node will fill this
        "error":    ""     # WHY: filled only if something fails
    }

    try:
        # WHY: invoke() runs the full graph from START to END
        # WHY: returns the final state after all nodes have run
        final_state = uab_graph.invoke(initial_state)

        print(f"\n✅ UAB System — Answer ready")
        print(f"   Agent used: {final_state['agent'].upper()}")

        return {
            "answer":  final_state["answer"],
            "agent":   final_state["agent"],
            "sources": final_state["sources"],
            "error":   final_state["error"]
        }

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ UAB System error: {error_msg}")

        return {
            "answer":  "Something went wrong. Please try again.",
            "agent":   "error",
            "sources": [],
            "error":   error_msg
        }


# ================================================================
# QUICK TEST — run this file directly to test routing
# ================================================================

if __name__ == "__main__":
    # WHY: test all 3 routing paths independently
    # HOW: run → python main.py

    print("=" * 55)
    print("  UAB LANGGRAPH — Routing Test")
    print("=" * 55)

    test_questions = [
        # WHY: should route to SQL agent
        "Who has the highest CGPA in Electronics department?",

        # WHY: should route to RAG agent
        "What is the minimum attendance required to appear in exams?",

        # WHY: should route to Web agent
        "What scholarships are available for engineering students?",
    ]

    for question in test_questions:
        result = run_uab(question)
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        print(f"Agent: {result['agent'].upper()}")
        print(f"A: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {result['sources']}")