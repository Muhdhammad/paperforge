const styles = {
  page: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh",
    width: "100%",
    boxSizing: "border-box",
    padding: "48px 24px",
    background: "#0a0a0a",
    color: "#e8e8e8",
    fontFamily: "'DM Sans', sans-serif",
    maxWidth: "760px",
    margin: "0 auto",
  },
  heading: {
    fontSize: "32px",
    fontWeight: "700",
    margin: "0 0 8px 0",
  },
  subheading: {
    fontSize: "15px",
    color: "#555",
    margin: "6 0 32px 0",
  },
  status: {
    color: "#555",
    fontSize: "15px",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  row: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 18px",
    background: "#141414",
    border: "1px solid #262626",
    borderRadius: "6px",
  },
  fileInfo: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    overflow: "hidden",
  },
  icon: {
    fontSize: "18px",
    flexShrink: 0,
  },
  filename: {
    fontSize: "14px",
    color: "#e8e8e8",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  deleteBtn: {
    background: "transparent",
    border: "1px solid #3a1a1a",
    color: "#f87171",
    borderRadius: "8px",
    padding: "6px 14px",
    fontSize: "13px",
    cursor: "pointer",
    flexShrink: 0,
    transition: "background 0.15s",
  },
};

export default styles;