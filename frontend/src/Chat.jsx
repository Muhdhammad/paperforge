import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import styles from "./styles";
import logo from "./assets/paperforge.png";
import docIcon from "./assets/docicon.png";

const API_URL = import.meta.env.VITE_API_URL;  //localhost

export default function Chat({ messages, setMessages }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // paper filter state
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [papers, setPapers] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [loadingPapers, setLoadingPapers] = useState(false);
  const dropdownRef = useRef(null);

  // close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  async function fetchPapers() {
    if (papers.length > 0) {
      setShowDropdown((prev) => !prev);
      return;
    }
    setLoadingPapers(true);
    setShowDropdown(true);
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setPapers(data.Documents || []);
    } catch (err) {
      console.error("Could not fetch papers", err);
    }
    setLoadingPapers(false);
  }

  async function sendMessage() {
    if (!input.trim()) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    const history = messages.slice(-3).map((msg) => ({
      role: msg.role,
      content: msg.text,
    }));

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: input,
          chat_history: history,
          paper_filter: selectedPaper || null,
        }),
      });

      const data = await res.json();
      const assistantMessage = {
        role: "assistant",
        text: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "⚠ Could not reach the server." },
      ]);
    }

    setLoading(false);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const inputBar = (
    <div style={styles.inputWrapper}>

      {selectedPaper && (
        <div style={styles.paperTag}>
          <span style={styles.paperTagText}><img src={docIcon} style={{ width: "14px", height: "14px" }} /> {selectedPaper} {selectedPaper}</span>
          <button style={styles.clearBtn} onClick={() => setSelectedPaper(null)}>✕</button>
        </div>
      )}

      <div style={styles.inputBox}>

        <div style={{ position: "relative" }} ref={dropdownRef}>
          <button
            style={{
              ...styles.iconBtn,
              color: selectedPaper ? "#00c4a0" : "#555",
            }}
            title="Filter by paper"
            onClick={fetchPapers}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
            </svg>
          </button>

          {showDropdown && (
            <div style={styles.dropdown}>
              {loadingPapers ? (
                <p style={styles.dropdownItem}>Loading...</p>
              ) : papers.length === 0 ? (
                <p style={styles.dropdownItem}>No papers found</p>
              ) : (
                <>
                  <button
                    style={{
                      ...styles.dropdownBtn,
                      color: !selectedPaper ? "#e8e8e8" : "#555",
                    }}
                    onClick={() => { setSelectedPaper(null); setShowDropdown(false); }}
                  >
                    All papers
                  </button>
                  {papers.map((paper) => (
                    <button
                      key={paper}
                      style={{
                        ...styles.dropdownBtn,
                        color: selectedPaper === paper ? "#00c4a0" : "#e8e8e8",
                        background: selectedPaper === paper ? "rgba(0,196,160,0.08)" : "transparent",
                      }}
                      onClick={() => { setSelectedPaper(paper); setShowDropdown(false); }}
                    >
                      <img src={docIcon} style={{ width: "14px", height: "14px", marginRight: "8px"}} /> {paper}
                    </button>
                  ))}
                </>
              )}
            </div>
          )}
        </div>

        <textarea
          style={styles.textarea}
          placeholder="Ask me something…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />

        <button
          style={{ ...styles.sendBtn, opacity: input.trim() ? 1 : 0.4 }}
          onClick={sendMessage}
          disabled={!input.trim() || loading}
        >
          ↑
        </button>
      </div>
    </div>
  );

  return (
    <div style={styles.page}>
      {messages.length === 0 ? (
        <div style={styles.centered}>
          <img src={logo} style={{ width: "64px", height: "64px", marginBottom: "4px" }} />
          <h1 style={styles.heading}>PaperForge</h1>
          <p style={styles.subheading}>Hello there, what are we reading today?</p>
          {inputBar}
        </div>
      ) : (
        <>
          <div style={styles.messages} className="messages">
            {messages.map((msg, i) => (
              <div
                key={i}
                style={{
                  ...styles.bubble,
                  alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
                  background: msg.role === "user" ? "#1e1e1e" : "#141414",
                }}
              >
                <span style={styles.roleLabel}>
                  {msg.role === "user" ? "Me" : "<_>"}
                </span>

                <div style={styles.bubbleText} className="bubble-content">
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div style={styles.sourcesBox}>
                    <span style={styles.sourcesLabel}>Sources</span>
                    {msg.sources.slice(0, 3).map((src, j) => (
                      <div key={j} style={styles.sourceRow}>
                        <span style={styles.sourceFile}><img src={docIcon} style={{ width: "11px", height: "11px" }} /> {src.paper}</span>
                        <span style={styles.sourceChunk}>chunk {src.chunk_index}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div style={{ ...styles.bubble, alignSelf: "flex-start", background: "#141414" }}>
                <span style={styles.roleLabel}>PaperForge</span>
                <p style={styles.bubbleText}>...</p>
              </div>
            )}
          </div>
          {inputBar}
        </>
      )}
    </div>
  );
}