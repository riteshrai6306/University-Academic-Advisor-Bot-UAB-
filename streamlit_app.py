# ================================================================
# streamlit_app.py
# ================================================================
# WHAT: Streamlit UI for the UAB (University Academic Advisor Bot)
# WHY:  Gives students a clean chat interface to ask questions
#       routed automatically to SQL / RAG / Web agents
#
# HOW TO RUN:
#   streamlit run streamlit_app.py
#   Then open: http://localhost:8501
#
# PLACE THIS FILE in your UAB/ root folder (same level as app/)
# ================================================================

import streamlit as st
from app.main import run_uab

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="UAB — University Academic Advisor Bot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── HEADER ── */
.uab-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
    color: white;
}
.uab-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: white;
}
.uab-header p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
}

/* ── BADGE ROW ── */
.badge-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}
.badge {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-sql { background: #dbeafe; color: #1e40af; }
.badge-rag { background: #dcfce7; color: #166534; }
.badge-web { background: #fef9c3; color: #854d0e; }

/* ── CHAT BUBBLES ── */
.msg-user {
    background: #2563eb;
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 60px;
    font-size: 0.95rem;
    line-height: 1.5;
}
.msg-bot {
    background: #f1f5f9;
    color: #0f172a;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 60px 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-agent-badge {
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 6px;
    opacity: 0.65;
}

/* ── SOURCE CHIP ── */
.source-chip {
    background: #e2e8f0;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #475569;
    display: inline-block;
    margin: 3px 3px 0 0;
    word-break: break-all;
}

/* ── EXAMPLES ── */
.example-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 8px;
}

/* ── FOOTER ── */
.uab-footer {
    text-align: center;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
}

/* hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# CONSTANTS
# ================================================================

AGENT_EMOJI = {
    "sql": "🗄️",
    "rag": "📄",
    "web": "🌐",
    "error": "❌"
}

AGENT_LABEL = {
    "sql": "SQL Agent — Student Database",
    "rag": "RAG Agent — University Documents",
    "web": "Web Agent — Live Search",
    "error": "Error"
}

# ================================================================
# SESSION STATE
# ================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, agent, sources}


def build_history_text(messages: list) -> str:
    """Builds the prior conversation text for context-aware follow-up rewriting."""
    history_lines = []
    for msg in messages:
        if msg["role"] == "user":
            history_lines.append(f"User: {msg['content']}")
        else:
            agent   = msg.get("agent", "assistant")
            label   = AGENT_LABEL.get(agent, "Assistant")
            history_lines.append(f"{label}: {msg['content']}")
    return "\n".join(history_lines)

# ================================================================
# HEADER
# ================================================================

st.markdown("""
<div class="uab-header">
    <h1>🎓 UAB — University Academic Advisor Bot</h1>
    <p>Ask me anything about student records, university policies, scholarships & internships.</p>
    <div class="badge-row">
        <span class="badge badge-sql">🗄️ SQL — Student Data</span>
        <span class="badge badge-rag">📄 RAG — University Docs</span>
        <span class="badge badge-web">🌐 Web — Live Search</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# CHAT HISTORY DISPLAY
# ================================================================

# def render_sources(sources: list):
#     """Renders source chips below a bot message."""
#     if not sources:
#         return ""


chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            agent   = msg.get("agent", "")
            emoji   = AGENT_EMOJI.get(agent, "🤖")
            label   = AGENT_LABEL.get(agent, agent.upper())

            badge   = f'<div class="msg-agent-badge">{emoji} {label}</div>'
            content = msg["content"].replace("\n", "<br>")
            # srcs    = render_sources(sources)

            st.markdown(
                f'<div class="msg-bot">{badge}{content}</div>',
                unsafe_allow_html=True
            )

# ================================================================
# INPUT BOX
# ================================================================

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([9, 2])
    with col1:
        user_input = st.text_input(
            label="question",
            placeholder="Ask a question about students, policies, scholarships...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send", use_container_width=True)

col_clear, _ = st.columns([1, 5])
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================================================================
# HANDLE SUBMISSION
# ================================================================

def process_question(question: str):
    """Runs the UAB pipeline and updates session state."""

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    # Build conversation history for follow-up context
    history = build_history_text(st.session_state.messages[:-1])

    # Run UAB graph
    with st.spinner("⏳ Thinking..."):
        result = run_uab(question, history=history)

    answer  = result.get("answer", "No answer generated.")
    agent   = result.get("agent", "error")
    sources = result.get("sources", [])

    # Save bot message
    st.session_state.messages.append({
        "role":    "bot",
        "content": answer,
        "agent":   agent,
        "sources": sources,
    })

    st.rerun()


# Handle form submission
if submitted and user_input.strip():
    process_question(user_input.strip())

# Handle example button clicks
if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    process_question(q)

# ================================================================
# FOOTER
# ================================================================

st.markdown("""
<div class="uab-footer">
    Powered by LangGraph · OpenAI GPT-4o-mini · FAISS · SQLite · DuckDuckGo
</div>
""", unsafe_allow_html=True)