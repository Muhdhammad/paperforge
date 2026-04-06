import { useState, useEffect } from "react";
import Chat from "./Chat";
import KnowledgeBase from "./KnowledgeBase";

export default function App() {
  const [page, setPage] = useState("chat");

  // load messages from localStorage on startup
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("chat_messages");
    return saved ? JSON.parse(saved) : [];
  });

  // save to localStorage whenever messages change
  useEffect(() => {
    localStorage.setItem("chat_messages", JSON.stringify(messages));
  }, [messages]);

  return (
    <div style={{ background: "#0a0a0a", minHeight: "100vh" }}>

      <nav style={styles.nav}>
        <button
          className="nav-btn"
          style={{ ...styles.navBtn, ...(page === "chat" ? styles.active : {}) }}
          onClick={() => setPage("chat")}
        >
          Chat
        </button>
        <button
          className="nav-btn"
          style={{ ...styles.navBtn, ...(page === "kb" ? styles.active : {}) }}
          onClick={() => setPage("kb")}
        >
          Knowledge Base
        </button>

        {messages.length > 0 && (
          <button
            className="nav-btn"
            style={{ ...styles.navBtn, position: "absolute", right: "24px", color: "#555" }}
            onClick={() => {
              setMessages([]);
              localStorage.removeItem("chat_messages");
              setPage("chat");
            }}
          >
            New Chat
          </button>
          )}
      </nav>

      {page === "chat" ? (
        <Chat messages={messages} setMessages={setMessages} />
      ) : (
        <KnowledgeBase />
      )}

    </div>
  );
}

const styles = {
  nav: {
    display: "flex",
    justifyContent: "center",
    position: "relative",
    gap: "8px",
    padding: "16px 24px",
    borderBottom: "1px solid #1a1a1a",
    background: "#0a0a0a",
  },
  navBtn: {
    background: "none",
    border: "none",
    outline: "none",
    color: "#555",
    padding: "6px 16px",
    fontSize: "14px",
    cursor: "pointer",
    fontFamily: "'DM Sans', sans-serif",
  },
  active: {
    color: "#e8e8e8",
    borderBottom: "1px solid #e8e8e8",
    borderRadius: "0",
  },
};