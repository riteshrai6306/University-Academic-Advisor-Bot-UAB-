# ================================================================
# agents/sql_agent.py
# ================================================================
# WHAT: Text-to-SQL Agent for structured student data
# WHY:  Student records (StudentID, Name, Course, Marks(/100), Attendance(%), CGPA, Department) are perfect for SQL querying.
#       live in a structured Excel file. We load it into SQLite
#       so the AI can convert natural language → SQL → answer.
#
# FLOW:
#   Excel file → SQLite database → LangChain SQL Agent
#   Student question → LLM generates SQL → executes → returns answer
#
# YOUR EXCEL COLUMNS (from uab_records.xlsx → Student Records sheet):
#   StudentID | Name | Course | Semester | Marks (/100) |
#   Attendance (%) | CGPA | Department
# ================================================================


# ── IMPORTS ─────────────────────────────────────────────────────

import os
import pandas as pd

# WHY: loads our API keys from .env file
from dotenv import load_dotenv

# WHY: SQLAlchemy creates and manages our SQLite database connection
from sqlalchemy import create_engine, text

# WHY: LangChain's built-in SQL database wrapper
# WHY: it understands table schemas and passes them to the LLM
from langchain_community.utilities import SQLDatabase

# WHY: LangChain's ready-made SQL agent — handles the full
#      Text → SQL → Execute → Answer pipeline automatically
from langchain_community.agent_toolkits import create_sql_agent

# WHY: GPT-4o-mini powers our SQL agent — reads schema, writes SQL
from langchain_openai import ChatOpenAI

from functools import lru_cache

# WHY: cached SQL agent reduces per-question LLM setup overhead
_sql_agent = None
_sql_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── LOAD ENVIRONMENT VARIABLES ──────────────────────────────────

# WHY: reads OPENAI_API_KEY from .env file
# NOTE: must be called before any OpenAI usage
load_dotenv()


# ── CONSTANTS ───────────────────────────────────────────────────

# WHY: path to your Excel file inside data/excel/
EXCEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "excel", "uab_records.xlsx"
)

# WHY: SQLite database file — auto-created when we load Excel
# WHY: SQLite needs no server, just a local file — perfect for dev
DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "db", "university.db"
)

# WHY: SQLAlchemy connection string for SQLite
DB_URL = f"sqlite:///{DB_PATH}"

# WHY: the sheet name in your Excel file that has student records
# NOTE: must match exactly — check your Excel sheet tab name
SHEET_NAME = "Student Records"

# WHY: this is the table name inside SQLite after we load Excel
TABLE_NAME = "student_records"


# ================================================================
# STEP 1 — LOAD EXCEL INTO SQLITE
# ================================================================

def load_excel_to_sqlite():
    """
    WHAT: Reads your Excel file and loads it into a SQLite database.
    WHY:  LangChain SQL agent works with databases, not Excel files.
          We convert Excel → SQLite once, then query it forever.

    HOW:
        1. pandas reads the Excel sheet
        2. Column names are cleaned (spaces → underscores)
        3. Data is written into SQLite table
        4. SQLite file saved at db/university.db

    NOTE: Call this once when app starts or when Excel data changes.
    """

    print("\n🔄 Loading Excel into SQLite...")

    # WHY: check if Excel file exists before trying to read
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"❌ Excel file not found at: {EXCEL_PATH}\n"
            f"   Make sure uab_records.xlsx is in data/excel/ folder"
        )

    # STEP 1: Read Excel sheet into pandas DataFrame
    # WHY: pandas makes it easy to clean and transform tabular data
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    print(f"✅ Excel loaded: {len(df)} students found")
    print(f"   Columns: {list(df.columns)}")

    # STEP 2: Clean column names
    # WHY: SQL doesn't like spaces or special chars in column names
    # WHY: "Marks (/100)" → "marks_100" prevents SQL errors
    df.columns = (
        df.columns
        .str.strip()                    # remove leading/trailing spaces
        .str.lower()                    # lowercase everything
        .str.replace(r"[^\w]", "_", regex=True)  # special chars → underscore
        .str.replace(r"_+", "_", regex=True)      # multiple underscores → one
        .str.strip("_")                # remove leading/trailing underscores
    )

    print(f"   Cleaned columns: {list(df.columns)}")

    # STEP 3: Create SQLite engine and write DataFrame
    # WHY: create_engine sets up the connection to our SQLite file
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(DB_URL)

    # WHY: if_exists="replace" → always load fresh data from Excel
    # NOTE: change to "append" if you want to add rows without replacing
    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="replace",  # WHY: replace table every time we reload
        index=False           # WHY: don't write pandas row index to DB
    )

    print(f"✅ SQLite database ready at: {DB_PATH}")
    print(f"   Table created: '{TABLE_NAME}' with {len(df)} rows\n")

    return engine


# ================================================================
# STEP 2 — GET DATABASE SCHEMA (for debugging)
# ================================================================

def get_schema() -> str:
    """
    WHAT: Returns the table schema as a readable string.
    WHY:  Useful for debugging — shows LLM exactly what columns exist.

    RETURNS: string showing table name and all column names
    """

    engine = create_engine(DB_URL)

    # WHY: PRAGMA table_info returns column details for a SQLite table
    with engine.connect() as conn:
        result = conn.execute(text(f"PRAGMA table_info({TABLE_NAME})"))
        columns = result.fetchall()

    schema = f"Table: {TABLE_NAME}\nColumns:\n"
    for col in columns:
        # col = (cid, name, type, notnull, default, pk)
        schema += f"  - {col[1]} ({col[2]})\n"

    return schema


# ================================================================
# STEP 3 — CREATE SQL AGENT
# ================================================================

def create_agent():
    """
    WHAT: Creates and returns a LangChain SQL agent.
    WHY:  The agent combines LLM + DB schema + SQL execution tools
          into one pipeline that handles Text → SQL → Answer.

    HOW IT WORKS INTERNALLY:
        1. Agent reads DB schema automatically
        2. Student asks question in natural language
        3. LLM generates SQL based on schema + question
        4. Agent executes SQL against SQLite
        5. LLM formats the raw result into a natural answer

    RETURNS: LangChain SQL agent ready to answer questions
    """

    global _sql_agent

    if _sql_agent is not None:
        return _sql_agent

    # WHY: make sure DB exists before creating agent
    if not os.path.exists(DB_PATH):
        print("Database not found — loading Excel first...")
        load_excel_to_sqlite()

    # STEP 1: Connect LangChain to our SQLite database
    # WHY: SQLDatabase wrapper gives the agent schema awareness
    # WHY: include_tables = only expose student_records table (safety)
    db = SQLDatabase.from_uri(
        DB_URL,
        include_tables=[TABLE_NAME],  # WHY: limit agent to only our table
        sample_rows_in_table_info=3   # WHY: shows agent 3 sample rows for context
    )

    # STEP 2: Create SQL agent
    # WHY: we reuse a cached LLM instance for speed
    _sql_agent = create_sql_agent(
        llm=_sql_llm,
        db=db,
        agent_type="zero-shot-react-description",
        verbose=False,            # WHY: disable verbose reasoning logs for speed
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=15    # WHY: shorter timeout to avoid long hangs
    )

    return _sql_agent


# ================================================================
# STEP 4 — RUN SQL AGENT (called by LangGraph node)
# ================================================================

def run_sql_agent(question: str) -> dict:
    """
    WHAT: Main function — takes student question, returns SQL answer.
    WHY:  This is what LangGraph calls when router sends a
          structured data question to the SQL node.

    ARGS:
        question (str): student's natural language question
                        e.g. "Who is the topper in CS department?"

    RETURNS:
        dict:
            answer  (str)  → natural language answer from agent
            success (bool) → True if query ran successfully
            error   (str)  → error message if something went wrong

    EXAMPLE QUESTIONS THIS HANDLES:
        - "Who has the highest CGPA?"
        - "List all students in Mechanical department"
        - "Which students have attendance below 75%?"
        - "What is the average CGPA of CS department?"
        - "Who are the toppers in each department?"
        - "How many students scored above 90 marks?"
        - "Show students with CGPA above 9 in Electronics"
    """

    print(f"\n🔍 SQL Agent received: {question}")

    try:
        # STEP 1: Create the agent
        agent = create_agent()

        # STEP 2: Add context to question so agent understands domain
        # WHY: adding context helps LLM write better SQL
        # WHY: explicitly mention table name so agent doesn't guess
        enriched_question = f"""
        You are a university academic advisor assistant.
        Answer this question using the student database.
        
        Table name: {TABLE_NAME}
        
        EXACT column names to use in SQL (do not guess or rename):
        - studentid        → unique student ID (e.g. CS1001, ME1042)
        - name             → full name of the student
        - course           → program enrolled (e.g. B.Tech CSE, MCA, BCA)
        - semester         → current semester number (1 to 8)
        - marks_100        → marks scored out of 100
        - attendance_      → attendance percentage (e.g. 75.5 means 75.5%)
        - cgpa             → cumulative grade point average (scale 0 to 10)
        - department       → department name:
                             Computer Science, Electronics, Mechanical,
                             Civil, Information Technology
        
        Question: {question}
        
        Give a clear, friendly answer with the relevant student details.
        """

        # STEP 3: Run the agent
        # WHY: invoke() runs the full Text → SQL → Execute → Answer chain
        response = agent.invoke({"input": enriched_question})

        # STEP 4: Extract answer from response
        # WHY: LangChain agent returns dict with "output" key
        answer = response.get("output", "No answer generated.")

        print(f"✅ SQL Agent answered successfully")

        return {
            "answer": answer,
            "success": True,
            "error": None
        }

    except Exception as e:
        # WHY: catch all errors and return gracefully
        # WHY: never crash the app — always return something useful
        error_msg = str(e)
        print(f"❌ SQL Agent error: {error_msg}")

        return {
            "answer": (
                "I encountered an issue querying the student database. "
                f"Please try rephrasing your question."
            ),
            "success": False,
            "error": error_msg
        }


# ================================================================
# QUICK TEST — run this file directly to test SQL agent
# ================================================================

if __name__ == "__main__":
    # WHY: lets you test SQL agent independently without running full app
    # HOW: run → python agents/sql_agent.py

    print("=" * 55)
    print("  SQL AGENT — Quick Test")
    print("=" * 55)

    # STEP 1: Load Excel into SQLite first
    load_excel_to_sqlite()

    # STEP 2: Print schema so you can verify columns
    print("\n📋 DATABASE SCHEMA:")
    print(get_schema())

    # STEP 3: Test different question types
    test_questions = [
        "Who has the highest CGPA overall?",
        "List all students in the Mechanical department",
        "Which students have attendance below 75%?",
        "What is the average CGPA of Computer Science department?",
        "How many students scored above 90 marks?",
    ]

    for question in test_questions:
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        result = run_sql_agent(question)
        print(f"A: {result['answer']}")
