// ================================================================
// LandingPage.js
// WHAT: Landing / Login page — first screen users see
// WHY:  Professional entry point with agent selection cards
//       No real auth yet — just name input + agent pick + launch
// ================================================================

import React, { useState, useEffect } from "react";
import "./LandingPage.css";

// ── AGENT DEFINITIONS ─────────────────────────────────────────
// WHAT: defines each AI agent's public-facing identity
// WHY:  professional naming makes it clear what each agent does
//       without exposing technical terms like "RAG" or "SQL"
const AGENTS = [
  {
    key:         "rag",
    icon:        "📋",
    name:        "Academic Compass",
    subtitle:    "Policies · Syllabi · Documents",
    description: "Explore course syllabi, university policies, fee structures, admission criteria and official academic documents — powered by intelligent document search.",
    tag:         "Document Intelligence",
    color:       "#10b981",
    glow:        "rgba(16,185,129,0.25)",
    questions:   ["What are the prerequisites for ML?", "Explain the fee structure", "What is the leave policy?"],
  },
  {
    key:         "sql",
    icon:        "🔬",
    name:        "Student Lens",
    subtitle:    "Records · Analytics · Database",
    description: "Query live student records, grade distributions, enrollment stats, and departmental analytics from the university's structured database in plain English.",
    tag:         "Database Intelligence",
    color:       "#3b82f6",
    glow:        "rgba(59,130,246,0.25)",
    questions:   ["How many students are in CS?", "Show top performers in Sem 3", "List students with backlog"],
  },
  {
    key:         "web",
    icon:        "🌐",
    name:        "Horizon Scout",
    subtitle:    "News · Scholarships · Live Web",
    description: "Fetch real-time scholarship opportunities, exam notifications, university rankings, and the latest academic news from across the web.",
    tag:         "Live Web Intelligence",
    color:       "#f59e0b",
    glow:        "rgba(245,158,11,0.25)",
    questions:   ["Latest UGC notifications?", "Scholarship deadlines 2025", "DU ranking this year"],
  },
];

export default function LandingPage({ onLaunch }) {
  const [name,     setName]     = useState("");
  const [selected, setSelected] = useState(null);
  const [visible,  setVisible]  = useState(false);

  // WHY: staggered entrance animation on mount
  useEffect(() => {
    setTimeout(() => setVisible(true), 50);
  }, []);

  const canLaunch = name.trim().length > 0 && selected !== null;

  const handleLaunch = () => {
    if (canLaunch) onLaunch(selected);
  };

  return (
    <div className={`landing ${visible ? "landing--visible" : ""}`}>

      {/* ── BACKGROUND GRID ── */}
      <div className="landing__grid" aria-hidden="true" />
      <div className="landing__orb landing__orb--1" aria-hidden="true" />
      <div className="landing__orb landing__orb--2" aria-hidden="true" />
      <div className="landing__orb landing__orb--3" aria-hidden="true" />

      {/* ── TOP NAV ── */}
      <nav className="landing__nav">
        <div className="landing__nav-logo">
          <span className="landing__nav-icon">🎓</span>
          <span className="landing__nav-brand">UAB <span className="landing__nav-thin">Portal</span></span>
        </div>
        <div className="landing__nav-links">
          <a href="#about" className="landing__nav-link">About</a>
          <a href="#agents" className="landing__nav-link">Advisors</a>
          <a href="#contact" className="landing__nav-link">Contact</a>
          <span className="landing__nav-version">v2.1</span>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="landing__hero">
        <div className="landing__hero-eyebrow">
          <span className="landing__pulse" />
          AI-Powered Academic Advisory System
        </div>
        <h1 className="landing__hero-title">
          Your University,<br />
          <span className="landing__hero-gradient">Intelligently Guided</span>
        </h1>
        <p className="landing__hero-sub">
          Three specialised AI advisors — documents, data, and live web —<br />
          unified in one intelligent academic companion.
        </p>

        {/* ── STATS ROW ── */}
        <div className="landing__stats">
          {[
            { value: "3", label: "AI Advisors" },
            { value: "∞", label: "Questions Answered" },
            { value: "<2s", label: "Avg Response" },
            { value: "GPT‑4o", label: "Powered By" },
          ].map(s => (
            <div key={s.label} className="landing__stat">
              <span className="landing__stat-value">{s.value}</span>
              <span className="landing__stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── AGENT CARDS ── */}
      <section className="landing__agents" id="agents">
        <h2 className="landing__section-title">Choose Your Advisor</h2>
        <p className="landing__section-sub">Each advisor is trained for a specific domain. Select one to begin.</p>

        <div className="landing__cards">
          {AGENTS.map((agent, i) => (
            <button
              key={agent.key}
              className={`landing__card ${selected === agent.key ? "landing__card--active" : ""}`}
              style={{
                "--card-color": agent.color,
                "--card-glow":  agent.glow,
                animationDelay: `${i * 0.1}s`,
              }}
              onClick={() => setSelected(agent.key)}
              aria-pressed={selected === agent.key}
            >
              <div className="landing__card-tag">{agent.tag}</div>
              <div className="landing__card-icon">{agent.icon}</div>
              <h3 className="landing__card-name">{agent.name}</h3>
              <p className="landing__card-subtitle">{agent.subtitle}</p>
              <p className="landing__card-desc">{agent.description}</p>

              <div className="landing__card-questions">
                <p className="landing__card-qlabel">Try asking:</p>
                {agent.questions.map(q => (
                  <span key={q} className="landing__card-q">"{q}"</span>
                ))}
              </div>

              {selected === agent.key && (
                <div className="landing__card-check">✓ Selected</div>
              )}
            </button>
          ))}
        </div>
      </section>

      {/* ── LOGIN FORM ── */}
      <section className="landing__login" id="login">
        <div className="landing__login-box">
          <div className="landing__login-header">
            <h2 className="landing__login-title">Begin Your Session</h2>
            <p className="landing__login-sub">No account needed — enter your name to continue</p>
          </div>

          <div className="landing__login-fields">
            <div className="landing__field">
              <label className="landing__label">Your Name</label>
              <input
                className="landing__input"
                type="text"
                placeholder="e.g. Arjun Sharma"
                value={name}
                onChange={e => setName(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleLaunch()}
              />
            </div>

            <div className="landing__field">
              <label className="landing__label">Selected Advisor</label>
              <div className={`landing__selected-display ${!selected ? "landing__selected-display--empty" : ""}`}>
                {selected
                  ? `${AGENTS.find(a => a.key === selected)?.icon}  ${AGENTS.find(a => a.key === selected)?.name}`
                  : "← Pick an advisor above first"}
              </div>
            </div>
          </div>

          <button
            className={`landing__launch-btn ${canLaunch ? "landing__launch-btn--ready" : ""}`}
            onClick={handleLaunch}
            disabled={!canLaunch}
          >
            {canLaunch
              ? `Launch ${AGENTS.find(a => a.key === selected)?.name} →`
              : "Select an advisor & enter your name"}
          </button>

          <p className="landing__disclaimer">
            🔒 Your session is private. No data is stored or shared.
          </p>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="landing__footer">
        <div className="landing__footer-brand">🎓 UAB — University Academic Bot</div>
        <div className="landing__footer-links">
          <span>Built with GPT-4o · LangChain · FAISS · FastAPI · React</span>
        </div>
        <div className="landing__footer-copy">© 2025 Academic Intelligence Platform</div>
      </footer>

    </div>
  );
}