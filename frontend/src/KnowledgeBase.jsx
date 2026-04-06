import { useState, useEffect } from "react";
import styles from "./kbStyles";
import docIcon from "./assets/docicon.png";

const API_URL = import.meta.env.VITE_API_URL;

export default function KnowledgeBase() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(null); 

  useEffect(() => {
    fetchDocuments();
  }, []);

  async function fetchDocuments() {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setDocuments(data.Documents || []);
    } catch (err) {
      console.error("Could not fetch documents", err);
      alert("Failed to load documents. Is the server running?");
    }
    setLoading(false);
  }

  async function deleteDocument(filename) {
   
    if (!window.confirm(`Delete all points for "${filename}"?`)) return;

    setDeleting(filename); 
    try {
      await fetch(`${API_URL}/documents/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      
      setDocuments((prev) => prev.filter((d) => d !== filename));
    } catch (err) {
      console.error("Could not delete document", err);
      alert("Failed to delete document. Please try again.");
    }
    setDeleting(null);
  }

  return (
    <div style={styles.page}>

      <h1 style={styles.heading}>Knowledge Base</h1>
      <p style={styles.subheading}>
        All Documents stored in Qdrant: {documents.length > 0 && (
          <strong style={{ color: "#e8e8e8ee" }}>{documents.length} total</strong>
        )}
      </p>

      {loading ? (
        <p style={styles.status}>Loading...</p>
      ) : documents.length === 0 ? (
        <p style={styles.status}>No documents found.</p>
      ) : (
        <div style={styles.list}>
          {documents.map((filename) => (
            <div key={filename} style={styles.row}>

              {/* PDF icon + filename */}
              <div style={styles.fileInfo}>
                <span style={styles.icon}><img src={docIcon} style={{ width: "17px", height: "17px" }} /></span>
                <span style={styles.filename}>{filename}</span>
              </div>

              {/* Delete button */}
              <button
                style={{
                  ...styles.deleteBtn,
                  opacity: deleting === filename ? 0.5 : 1,
                }}
                onClick={() => deleteDocument(filename)}
                disabled={deleting === filename}
              >
                {deleting === filename ? "Deleting..." : "Delete"}
              </button>

            </div>
          ))}
        </div>
      )}

    </div>
  );
}