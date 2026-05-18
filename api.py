# ================================================================
# api.py
# ================================================================
# WHAT: FastAPI backend for UAB React frontend
# WHY:  Exposes our LangGraph agents as REST API endpoints
# HOW TO RUN: uvicorn api:app --reload --port 8000
# ================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from app.main import run_uab

# ── APP SETUP ───────────────────────────────────────────────────
app = FastAPI(
    title="UAB — University Academic Advisor Bot API",
    version="1.0.0"
)

# WHY: allows React (running on port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ───────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    history: str = ""  # optional conversation history

class AnswerResponse(BaseModel):
    answer: str
    agent: str
    sources: list
    response_time: float  # WHY: needed for evaluation

# ── ROUTES ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "UAB API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    WHAT: Main endpoint — receives student question, returns answer.
    WHY:  React frontend calls this on every message sent.
    """
    start_time = time.time()

    result = run_uab(request.question, history=request.history)

    response_time = round(time.time() - start_time, 2)

    return AnswerResponse(
        answer=result.get("answer", "No answer generated."),
        agent=result.get("agent", "error"),
        sources=result.get("sources", []),
        response_time=response_time
    )