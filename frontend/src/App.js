// ================================================================
// App.js
// WHAT: Root component — manages page routing between Landing and Chat
// WHY:  Single-page app with view-based navigation (no react-router needed)
// ================================================================


import React, { useState } from "react";
import LandingPage from "./LandingPage";
import ChatPage from "./ChatPage";
import "./App.css";

export default function App() {
  // WHAT: tracks which page to show — "landing" or "chat"
  // WHY:  simple state-based routing without installing react-router
  const [page, setPage]           = useState("landing");
  const [activeAgent, setAgent]   = useState("rag");

  // WHY: called from LandingPage when user picks an agent and clicks launch
  const handleLaunch = (agentKey) => {
    setAgent(agentKey);
    setPage("chat");
  };

  // WHY: called from ChatPage header when user wants to go back
  const handleBack = () => setPage("landing");

  return page === "landing"
    ? <LandingPage onLaunch={handleLaunch} />
    : <ChatPage    activeAgent={activeAgent} onBack={handleBack} />;
}