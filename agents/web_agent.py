# ================================================================
# agents/web_agent.py
# ================================================================
# WHAT: Web Search Agent for live external information
# WHY:  Scholarships, internships, and deadlines change frequently.
#       No PDF or database can stay this current.
#       We use DuckDuckGo (free, no API key) to fetch live results.
#
# FLOW:
#   Student question → DuckDuckGo search → top results fetched
#   → LLM reads results → summarizes into a clean answer
#
# WHY DUCKDUCKGO?
#   - Completely free — no API key needed
#   - No rate limit issues for development
#   - Returns clean results good enough for LLM processing
# ================================================================


# ── IMPORTS ─────────────────────────────────────────────────────

import os
from functools import lru_cache
# WHY: loads our API keys from .env file
from dotenv import load_dotenv

# WHY: DuckDuckGo search tool — free, no API key needed
# WHY: DDGS = DuckDuckGo Search class (newer API style)
from ddgs import DDGS

# WHY: GPT-4o-mini reads search results and summarizes them
from langchain_openai import ChatOpenAI

# WHY: structures our prompt for the LLM to summarize web results
from langchain_core.prompts import PromptTemplate

# WHY: chains prompt + LLM together into one pipeline
from langchain_core.output_parsers import StrOutputParser


# ── LOAD ENVIRONMENT VARIABLES ──────────────────────────────────

# WHY: reads OPENAI_API_KEY from .env file
load_dotenv()


# ── CONSTANTS ───────────────────────────────────────────────────

# WHY: number of search results to fetch per query
# NOTE: 5 gives enough variety without overwhelming the LLM
MAX_RESULTS = 5

# WHY: max characters to read from each search result
# NOTE: keeps token usage low while retaining useful content
MAX_CONTENT_LENGTH = 300


# ── PROMPT TEMPLATE ─────────────────────────────────────────────

# WHY: custom prompt makes GPT answer like a university advisor
# WHY: {results} = raw search results, {question} = student question
# WHY: "cite your sources" = gives student links to follow up
WEB_PROMPT_TEMPLATE = """
You are a helpful University Academic Advisor searching the web
for the latest opportunities for students.

Based on the search results below, answer the student's question
clearly and concisely. Always mention the source URLs so the
student can get more details.

If the results don't contain relevant information, say:
"I couldn't find specific information about this right now.
Please check official university websites or scholarship portals."

Search Results:
{results}

Student Question: {question}

Your Answer:
"""

# WHY: wraps our string template into LangChain compatible format
WEB_PROMPT = PromptTemplate(
    template=WEB_PROMPT_TEMPLATE,
    input_variables=["results", "question"]
)

# WHY: cache LLM and prompt chain for repeated web queries
_WEB_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
_WEB_CHAIN = WEB_PROMPT | _WEB_LLM | StrOutputParser()


# ================================================================
# STEP 1 — SEARCH THE WEB USING DUCKDUCKGO
# ================================================================

@lru_cache(maxsize=64)
def search_web(query: str) -> list:
    """
    WHAT: Searches DuckDuckGo and returns top results.
    WHY:  Fetches live, current information about scholarships,
          internships, deadlines that PDFs can't provide.

    ARGS:
        query (str): search query built from student's question

    RETURNS:
        list of dicts with keys: title, url, body (snippet)
    """

    print(f"\nSearching web for: {query}")

    try:
        # WHY: DDGS() creates a DuckDuckGo search session
        with DDGS() as ddgs:
            # WHY: ddgs.text() returns text search results
            # WHY: max_results limits how many we fetch
            results = list(
                ddgs.text(
                    query,
                    max_results=MAX_RESULTS
                )
            )

        print(f"Found {len(results)} results")
        return results

    except Exception as e:
        print(f"Web search error: {str(e)}")
        return []


# ================================================================
# STEP 2 — FORMAT SEARCH RESULTS FOR LLM
# ================================================================

def format_results(results: list) -> str:
    """
    WHAT: Converts raw DuckDuckGo results into clean readable text.
    WHY:  LLM needs structured text, not raw JSON objects.
          We format each result with title, URL, and snippet.

    ARGS:
        results (list): raw results from DuckDuckGo

    RETURNS:
        str: formatted results ready to inject into prompt
    """

    if not results:
        return "No search results found."

    formatted = ""

    for i, result in enumerate(results, 1):
        # WHY: extract title, url, body from each result dict
        title   = result.get("title", "No title")
        url     = result.get("href", "No URL")
        snippet = result.get("body", "No description")

        # WHY: truncate snippet to avoid exceeding token limits
        snippet = snippet[:MAX_CONTENT_LENGTH]

        formatted += f"""
Result {i}:
  Title   : {title}
  URL     : {url}
  Summary : {snippet}
---"""

    return formatted


# ================================================================
# STEP 3 — BUILD SMART SEARCH QUERY
# ================================================================

def build_search_query(question: str) -> str:
    """
    WHAT: Converts student question into an optimized search query.
    WHY:  Raw student questions may be too vague for web search.
          We add university-specific context to get better results.

    ARGS:
        question (str): raw student question

    RETURNS:
        str: optimized search query for DuckDuckGo

    EXAMPLES:
        "any scholarships for me?" 
        → "university scholarships for engineering students India 2026"
        
        "internships available?"
        → "internships for computer science students India 2026"
    """

    # WHY: adding "university students India 2026" gets more
    #      relevant results than the raw question alone
    query = f"{question} university students India 2026"

    return query


# ================================================================
# STEP 4 — RUN WEB AGENT (called by LangGraph node)
# ================================================================

def run_web_agent(question: str) -> dict:
    """
    WHAT: Main function — searches web and returns summarized answer.
    WHY:  This is what LangGraph calls when router decides the
          question needs live web data (scholarships, internships).

    ARGS:
        question (str): student's natural language question
                        e.g. "Are there any scholarships for CSE students?"

    RETURNS:
        dict:
            answer  (str)  → LLM summarized answer with sources
            sources (list) → list of URLs found
            success (bool) → True if search ran successfully
            error   (str)  → error message if something went wrong

    EXAMPLE QUESTIONS THIS HANDLES:
        - "What scholarships are available for CS students?"
        - "Any internships for 3rd year engineering students?"
        - "Government scholarships for engineering students 2026"
        - "Internship opportunities at top tech companies India"
        - "Fellowship programs for postgraduate students"
    """

    print(f"\nWeb Agent received: {question}")

    try:
        # STEP 1: Build optimized search query
        search_query = build_search_query(question)

        # STEP 2: Search DuckDuckGo
        raw_results = search_web(search_query)

        if not raw_results:
            # fallback to the raw user question if the optimized query fails
            raw_results = search_web(question)

        if not raw_results:
            return {
                "answer": (
                    "I couldn't find results right now. "
                    "Please check scholarship portals like "
                    "https://scholarships.gov.in or https://internshala.com"
                ),
                "sources": [],
                "success": False,
                "error": "No search results returned"
            }

        # STEP 3: Format results for LLM
        formatted_results = format_results(raw_results)

        # STEP 4: Extract source URLs for citation
        sources = [
            r.get("href", "")
            for r in raw_results
            if r.get("href")
        ]

        # STEP 5: Use the cached LLM and chain for speed
        chain = _WEB_CHAIN

        # STEP 6: Invoke chain with results + question
        answer = chain.invoke({
            "results": formatted_results,
            "question": question
        })

        # STEP 8: Extract answer
        # WHY: StrOutputParser returns a plain string answer
        if not isinstance(answer, str):
            answer = str(answer)

        print("Web Agent answered successfully")
        print(f"   Sources found: {len(sources)}")

        return {
            "answer":  answer,
            "sources": sources,
            "success": True,
            "error":   None
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Web Agent error: {error_msg}")

        return {
            "answer": (
                "I had trouble searching the web right now. "
                "Please try again or visit https://scholarships.gov.in "
                "for scholarship information."
            ),
            "sources": [],
            "success": False,
            "error":   error_msg
        }


# ================================================================
# QUICK TEST — run this file directly to test Web agent
# ================================================================

if __name__ == "__main__":
    # WHY: lets you test web agent independently without full app
    # HOW: run → python agents/web_agent.py

    print("=" * 55)
    print("  WEB AGENT — Quick Test")
    print("=" * 55)

    test_questions = [
        "What scholarships are available for engineering students?",
        "Internship opportunities for computer science students in India",
    ]

    for question in test_questions:
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        result = run_web_agent(question)
        print(f"\nA: {result['answer']}")
        print(f"\nSources:")
        for url in result["sources"]:
            print(f"  → {url}")