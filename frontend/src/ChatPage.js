// ================================================================
// ChatPage.js
// WHAT: Main chat interface — shown after user picks an agent & logs in
// WHY:  Full professional chat UI with sidebar agent switcher,
//       markdown rendering, and clean message layout
// ================================================================

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./ChatPage.css";

const API_URL = "http://localhost:8000";

// ── AGENT CONFIG ─────────────────────────────────────────────
// WHAT: Maps internal agent keys to UI presentation data
// WHY:  Centralised config means one change updates all UI elements
const AGENTS = {
  rag: {
    key:         "rag",
    icon:        "📋",
    name:        "Academic Compass",
    shortName:   "Compass",
    color:       "#10b981",
    glow:        "rgba(16,185,129,0.2)",
    tag:         "Document Intelligence",
    placeholder: "Ask about policies, syllabi, fees, admission criteria...",
    welcome:     "Hello! I'm **Academic Compass**, your document intelligence advisor. I can help you navigate course syllabi, university policies, fee structures, and official academic documents. What would you like to know?",
    suggestions: [
      "What are the prerequisites for Machine Learning?",
      "Explain the fee structure for postgraduate courses",
      "What is the attendance policy?",
      "List all departments under Faculty of Science",
    ],
  },
  sql: {
    key:         "sql",
    icon:        "🔬",
    name:        "Student Lens",
    shortName:   "Lens",
    color:       "#3b82f6",
    glow:        "rgba(59,130,246,0.2)",
    tag:         "Database Intelligence",
    placeholder: "Ask about student records, grades, enrollment statistics...",
    welcome:     "Hello! I'm **Student Lens**, your database intelligence advisor. I can query live student records, grade distributions, enrollment stats, and departmental analytics. What data would you like to explore?",
    suggestions: [
      "How many students are enrolled in Computer Science?",
      "Show top 10 performers in Semester 3",
      "List students with pending fee dues",
      "Average CGPA by department",
    ],
  },
  web: {
    key:         "web",
    icon:        "🌐",
    name:        "Horizon Scout",
    shortName:   "Scout",
    color:       "#f59e0b",
    glow:        "rgba(245,158,11,0.2)",
    tag:         "Live Web Intelligence",
    placeholder: "Ask about scholarships, notifications, rankings, news...",
    welcome:     "Hello! I'm **Horizon Scout**, your live web intelligence advisor. I can find real-time scholarship opportunities, exam notifications, university rankings, and the latest academic news. What shall I look up?",
    suggestions: [
      "Latest UGC scholarship notifications 2025",
      "Current university rankings in India",
      "Upcoming exam deadlines this semester",
      "Recent changes to admission criteria",
    ],
  },
};

// ── HELPER: build history string for context-aware follow-ups ──
const buildHistory = (msgs) =>
  msgs.map(m => `${m.role === "user" ? "User" : "Bot"}: ${m.content}`).join("\n");

export default function ChatPage({ activeAgent: initialAgent, onBack }) {
  const [activeAgent,  setActiveAgent]  = useState(initialAgent);
  const [sessions,     setSessions]     = useState({
    rag: [], sql: [], web: [],
  });
  const [input,        setInput]        = useState("");
  const [loading,      setLoading]      = useState(false);
  const [sidebarOpen,  setSidebarOpen]  = useState(true);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  const agent    = AGENTS[activeAgent];
  const messages = sessions[activeAgent];

  // WHY: auto-scroll to latest message on any change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [sessions, activeAgent, loading]);

  // WHY: focus input on agent switch
  useEffect(() => {
    inputRef.current?.focus();
  }, [activeAgent]);

  const setMessages = (updater) => {
    setSessions(prev => ({
      ...prev,
      [activeAgent]: typeof updater === "function"
        ? updater(prev[activeAgent])
        : updater,
    }));
  };

  const sendMessage = async (text) => {
    const question = (text || input).trim();
    if (!question || loading) return;
    setInput("");

    const updatedMessages = [...messages, { role: "user", content: question }];
    setMessages(updatedMessages);
    setLoading(true);

    try {
      // WHY: sends question + history to FastAPI /ask endpoint
      const response = await axios.post(`${API_URL}/ask`, {
        question,
        history: buildHistory(messages),
      });

      const { answer, agent: respondingAgent, response_time } = response.data;

      setMessages(prev => [...prev, {
        role:          "bot",
        content:       answer,
        agent:         respondingAgent,
        response_time,
      }]);

    } catch {
      setMessages(prev => [...prev, {
        role:    "bot",
        content: "⚠️ Something went wrong connecting to the server. Please try again.",
        agent:   "error",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="chat-page" style={{ "--agent-color": agent.color, "--agent-glow": agent.glow }}>

      {/* ── SIDEBAR ── */}
      <aside className={`chat-sidebar ${sidebarOpen ? "chat-sidebar--open" : "chat-sidebar--closed"}`}>

        {/* Logo */}
        <div className="sidebar__logo">
          <span className="sidebar__logo-icon">🎓</span>
          {sidebarOpen && (
            <span className="sidebar__logo-text">UAB <span className="sidebar__logo-thin">Portal</span></span>
          )}
        </div>

        {/* Back to home */}
        <button className="sidebar__back-btn" onClick={onBack} title="Back to home">
          <span className="sidebar__back-icon">←</span>
          {sidebarOpen && <span>Back to Home</span>}
        </button>

        {/* Agent switcher */}
        {sidebarOpen && <p className="sidebar__section-label">AI Advisors</p>}

        {Object.values(AGENTS).map(ag => (
          <button
            key={ag.key}
            className={`sidebar__agent-btn ${activeAgent === ag.key ? "sidebar__agent-btn--active" : ""}`}
            style={{ "--btn-color": ag.color }}
            onClick={() => setActiveAgent(ag.key)}
            title={ag.name}
          >
            <span className="sidebar__agent-icon">{ag.icon}</span>
            {sidebarOpen && (
              <span className="sidebar__agent-info">
                <span className="sidebar__agent-name">{ag.name}</span>
                <span className="sidebar__agent-tag">{ag.tag}</span>
              </span>
            )}
            {sidebarOpen && sessions[ag.key].length > 0 && (
              <span className="sidebar__msg-count">
                {sessions[ag.key].filter(m => m.role === "user").length}
              </span>
            )}
          </button>
        ))}

        {/* Divider + utilities */}
        <div className="sidebar__divider" />

        <button className="sidebar__util-btn" onClick={clearChat} title="Clear chat">
          <span>🗑</span>
          {sidebarOpen && <span>Clear Chat</span>}
        </button>

        {/* Toggle */}
        <button
          className="sidebar__toggle"
          onClick={() => setSidebarOpen(o => !o)}
          title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {sidebarOpen ? "◀" : "▶"}
        </button>
      </aside>

      {/* ── MAIN PANEL ── */}
      <main className="chat-main">

        {/* ── TOP BAR ── */}
        <header className="chat-header">
          <div className="chat-header__agent">
            <span className="chat-header__icon">{agent.icon}</span>
            <div>
              <h1 className="chat-header__name">{agent.name}</h1>
              <p className="chat-header__tag">{agent.tag}</p>
            </div>
          </div>
          <div className="chat-header__actions">
            <span className="chat-header__status">
              <span className="chat-header__dot" />
              Online
            </span>
            {messages.length > 0 && (
              <button className="chat-header__clear" onClick={clearChat}>
                Clear
              </button>
            )}
          </div>
        </header>

        {/* ── MESSAGES AREA ── */}
        <div className="chat-messages">

          {/* Welcome / empty state */}
          {messages.length === 0 && (
            <div className="chat-welcome">
              <div className="chat-welcome__icon">{agent.icon}</div>
              <h2 className="chat-welcome__name">{agent.name}</h2>
              <div className="chat-welcome__desc">
                {/* WHY: ReactMarkdown + remarkGfm renders **bold** properly */}
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {agent.welcome}
                </ReactMarkdown>
              </div>

              {/* Suggestion chips */}
              <div className="chat-suggestions">
                <p className="chat-suggestions__label">Try asking:</p>
                <div className="chat-suggestions__chips">
                  {agent.suggestions.map(s => (
                    <button
                      key={s}
                      className="chat-suggestion-chip"
                      onClick={() => sendMessage(s)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, i) => (
            <div key={i} className={`chat-msg chat-msg--${msg.role}`}>
              {msg.role === "bot" && (
                <div className="chat-msg__avatar">{agent.icon}</div>
              )}
              <div className="chat-msg__bubble">
                {msg.role === "bot" && (
                  <div className="chat-msg__meta">
                    <span className="chat-msg__agent-name">{agent.name}</span>
                    {msg.response_time && (
                      <span className="chat-msg__time">⏱ {msg.response_time}s</span>
                    )}
                  </div>
                )}
                {/* WHY: remarkGfm enables table & bold rendering */}
                <div className="chat-msg__content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>
              </div>
              {msg.role === "user" && (
                <div className="chat-msg__avatar chat-msg__avatar--user">👤</div>
              )}
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="chat-msg chat-msg--bot">
              <div className="chat-msg__avatar">{agent.icon}</div>
              <div className="chat-msg__bubble">
                <div className="chat-typing">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* ── INPUT BAR ── */}
        <footer className="chat-input-bar">
          <div className="chat-input-wrap">
            <textarea
              ref={inputRef}
              className="chat-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={agent.placeholder}
              rows={1}
              disabled={loading}
            />
            <button
              className={`chat-send-btn ${input.trim() && !loading ? "chat-send-btn--ready" : ""}`}
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
            >
              {loading ? (
                <span className="chat-send-spinner" />
              ) : (
                <span>↑</span>
              )}
            </button>
          </div>
          <p className="chat-input-hint">
            Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
          </p>
        </footer>

      </main>
    </div>
  );
}