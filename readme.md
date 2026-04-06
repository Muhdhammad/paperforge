# <img src="frontend/src/assets/paperforge.png" alt="Logo" width="30" height="30"> PaperForge

A multimodal **RAG (Retrieval-Augmented Generation)** system for research papers with hybrid (dense + sparse) retrieval, Jina reranking, and conversational querying with precise source attribution.

## Key Features

- **Hybrid Search**: Combines semantic search (dense vectors) with keyword matching (sparse vectors) using Reciprocal Rank Fusion (RRF)
- **Reranking**: Jina Reranker v3 for precision optimization
- **Query Reformulation**: Automatic rewriting of follow-up questions for better retrieval
- **Paper-Specific Filtering**: Search within specific documents or across entire knowledge base
- **Conversational Context**: Maintains chat history for coherent multi-turn conversations
- **Source Attribution**: Every answer includes paper names and chunk indices


## Architecture
```
                			┌─────────────────────────────────────────────────────────────┐
                			│                         User Query                          │
                			└──────────────────────────────┬──────────────────────────────┘
                			                               │
                			                               ▼
                			┌─────────────────────────────────────────────────────────────┐
                			│         Query Reformulation (if chat history exists)        │
                			│    "what about recall?" → "What is recall in BLEU metric?"  │
                			└──────────────────────────────┬──────────────────────────────┘
                			                               │
                			                               ▼
                			┌─────────────────────────────────────────────────────────────┐
                			│                  Hybrid Retrieval (Qdrant)                  │
                			│  ┌─────────────────────────┐     ┌───────────────────────┐  │
                			│  │      Dense Search       │     │     Sparse Search     │  │
                			│  │      (nomic-embed)      │     │        (BM42)         │  │
                			│  │     Top 10 results      │     │    Top 10 results     │  │
                			│  └────────────┬────────────┘     └───────────┬───────────┘  │
                			│               │                              │              │
                			│               └──────────────┬───────────────┘              │
                			│                              │                              │
                			│                    ┌─────────▼─────────┐                    │
                			│                    │    RRF Fusion     │                    │
                			│                    │      Top 10       │                    │
                			│                    └─────────┬─────────┘                    │
                			└──────────────────────────────┼──────────────────────────────┘
                			                               │
                			                               ▼
                			┌─────────────────────────────────────────────────────────────┐
                			│                Jina Reranker v3 (Optional)                  │
                			│                Rerank Top 10 → Final Top 3                  │
                			└──────────────────────────────┬──────────────────────────────┘
                			                               │
                			                               ▼
                			┌─────────────────────────────────────────────────────────────┐
                			│                LLM Generation (Groq/OpenAI)                 │
                			│          Prompt: Context + Chat History + Query             │
                			│              Output: Answer + Source Citations              │
                			└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Backend
- **FastAPI** - High-performance async API framework
- **Qdrant** - Vector database with hybrid search support
- **LangChain** - LLM orchestration and prompt management
- **Groq** - Ultra-fast LLM inference 
- **Jina AI** - State-of-the-art reranking

### Embeddings
- **Dense**: nomic-ai/nomic-embed-text-v1.5 (768-dim semantic vectors)
- **Sparse**: Qdrant/bm42-all-minilm-l6-v2-attentions (BM42 keyword vectors)

### Frontend
- **React** - UI framework
- **Vite** - Build tool
- **Nginx** - Production web server

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

## Quick Start

### Prerequisites
- Docker 
- API Keys:
  - **Groq API Key** ([Get it here](https://console.groq.com))
  - **Jina API Key** ([Get it here](https://jina.ai))
  - (Optional) OpenAI API Key

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/paperforge.git
cd paperforge
```

### 2. Configure Environment
Create a `.env` file in your project root:

```bash
GROQ_API_KEY=gsk_your_key_here
JINA_API=jina_your_key_here
OPENAI_API_KEY=sk_your_key_here  # optional
HF_TOKEN=hf_your_token_here  
```
### 3. Start Services
Build and start all services (Qdrant + Backend + Frontend):
```bash
docker-compose up --build
```
**Services will be available at:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8003
- Qdrant Dashboard: http://localhost:6333/dashboard

### 4. Preprocess & Ingest Research Papers
#### Step 1: Add PDFs  
```bash  
mkdir -p research-papers-arxiv   # Add your .pdf files here
```
#### Step 2: Convert PDFs → Markdown
Place your markdown files in `md-research-papers/`:
```bash
python preprocessing.py
```
Output will be saved in: `md-research-papers-arxiv/`
#### Step 3: Ingest into Vector Database
Run ingestion
```bash
docker-compose exec rag python ingestion.py
```
---
###  Local Development (without Docker Compose)  
  
If you prefer running services locally without Docker Compose:  
  
#### 1. Install Python dependencies  
```bash
pip install -r requirements.txt  
```  
#### 2. Start Qdrant locally  
```bash
docker run -p 6333:6333 qdrant/qdrant:latest  
```
#### 3. Preprocess PDFs to Markdown
```bash
# Place PDFs in input folder  
mkdir -p research-papers-arxiv  
  
# Run preprocessing  
python preprocessing.py
```

#### 4. Ingest Markdown files into Qdrant
```bash
# Output markdown from preprocessing will be in md-research-papers-arxiv/
python ingestion.py
```
#### 5. Run backend  
```bash
python app.py  
```
  
#### 6. Run frontend in another terminal  
```bash
cd frontend  
npm install  
npm run dev
```

## Run Tests
RAGAS retriever evaluation (Context Recall and Context Precision)
```bash
python -m test.ragas_eval 
``` 
Health check of Qdrant collection
```bash  
python -m src.vectordb
```

## 🔬 RAG Pipeline 

### 1. **Preprocessing**  
  
Raw PDFs are converted into structured markdown using **Docling + EasyOCR**, with multimodal extraction and enrichment.
  
**Pipeline Capabilities**  
- Extracts text, tables, and document structure (headings, sections)  
- Handles scanned PDFs via OCR (EasyOCR)  
- Generates **image descriptions using an LLM** and replaces base64 images  

### 2. **Document Ingestion**

**Chunking Strategy:**
- Heading-based splitting (splits on #, ##, ###), also option for semantic based chunking
- Preserves document structure and context
- Metadata: doc_id, file_name, chunk_index, chunk_size

**Hybrid Embeddings**  
- **Dense embeddings** → semantic understanding  
 `nomic-ai/nomic-embed-text-v1.5` (768-dim vectors)

- **Sparse embeddings** → keyword-level   
 `Qdrant/bm42-all-minilm-l6-v2-attentions` (BM42)

### 3. **Hybrid Search**
Combines two complementary search strategies:  
 
- **Dense Search (Semantic)**  
 Retrieves top_k semantically similar chunks using vector embeddings  
  
- **Sparse Search (Keyword)**  
Retrieves top_k keyword-matching chunks using BM42  
  
 
**Reciprocal Rank Fusion (RRF)**  
  
The two ranked lists are merged using RRF to produce a unified ranking.  
  
```python  
# Dense results (top 10): [A (rank 1), B (rank 2), ...]  
# Sparse results (top 10): [X (rank 1), A (rank 4), ...]  
  
# RRF Score = Σ (1 / (rank + k)), where k = 60  
  
# A: appears in both lists  
# → 1/61 (dense rank 1) + 1/64 (sparse rank 4) = 0.0325  
  
# X: appears only in sparse  
# → 1/61 = 0.0164

# Final ranking is sorted by combined RRF score
```

### 4. **Contextual Reranking**

**Jina Reranker**
- Applies cross-attention between the query and each retrieved chunk  
- Reorders top_k chunks → top_n reranked chunks based on relevance

### 5. **Query Reformulation**

Converts follow-up questions into standalone queries:
```
User: "What is BLEU score in the Transformer paper?"
Assistant: "28.4 on WMT 2014 English-to-German translation."

User: "What about recall?"
↓
Rewritten: "What is recall metric in the Transformer paper evaluation?"
```

### 6. **Response Generation**

- The prompt combines chat history, retrieved context, and the user query. Complete prompt can be found in `src/rag_engine.py`  
- **Rules enforced in prompt**:  
  **-** Use only information from context and chat history  
  **-** Cite numbers, methods, and results when relevant  
  **-** Format all mathematical expressions in LaTeX  
  **-** Be concise but precise; use bullet points when needed  
  **-** If information is missing, explicitly say so; do not speculate  
  
- **Fallback**  
**-** If no relevant chunks are found, the system returns:  
`"No relevant information found in the database."`

- **LLM Selection:**
**-** Groq (Llama 3.1)
**-** OpenAI (GPT-4)

---

## 📖 API Documentation

### Health Check
```bash
curl http://localhost:8003/health
```

### List Document
```bash
curl http://localhost:8003/documents
```

**Response:**

```json
{
  "Documents": ["paper1.md", "paper2.md"],
  "Total Documents": 2
}
```

### Query (Simple)

```bash
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What BLEU score did Transformer achieve?",
    "top_k": 10
  }'
```

### Query (with Paper Filter)

```bash
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the model architecture?",
    "paper_filter": "Attention_Is_All_You_Need.md",
    "top_k": 10
  }'
```

### Query (with Chat History)
```bash
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about recall?",
    "chat_history": [
      {"role": "user", "content": "What is BLEU score?"},
      {"role": "assistant", "content": "BLEU measures translation quality..."}
    ]
  }'
```

**Response:**
```json
{
  "answer": "The Transformer achieved 28.4 BLEU on WMT 2014 English-to-German translation.",
  "sources": [
    {"paper": "Attention_Is_All_You_Need.md", "chunk_index": 12},
    {"paper": "Attention_Is_All_You_Need.md", "chunk_index": 45},
    {"paper": "Attention_Is_All_You_Need.md", "chunk_index": 48},
  ]
}
```

### Delete Document

```bash
curl -X DELETE http://localhost:8003/documents/paper_name.md
```

## Project Structure
```
paperforge/
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── Chat.jsx               # Main chat interface
│   │   ├── KnowledgeBase.jsx      # Document management
│   │   └── ...
│   ├── Dockerfile               # Multi-stage build
│   └── package.json
│
├── src/                        # Backend core modules
│   ├── chunking.py                # Markdown chunking (heading/semantic)
│   ├── embedding.py               # Dense + Sparse embedding classes
│   ├── preprocessing.py           # Dense + Sparse embedding classes
│   ├── rag_engine.py              # RAG orchestration & query reformulation
│   ├── retriever.py               # Hybrid search + Jina reranking
│   ├── ingestion.py               # Dense + Sparse embedding classes
│   ├── vectordb.py                # Qdrant client wrapper
│   ├── utils.py                   # Utility functions
│   └── config.py                  # Configuration management
│
├── test/                       # Test files & evaluation
│   ├── ragas_eval.py              # RAGAS evaluation script
│   └── test-json/                 # QA pairs for evaluation
│
├── app.py                      # FastAPI server
├── Dockerfile                  # Backend Docker image
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Full Python dependencies
├── docker-requirement.txt      # Minimum runtime dependencies
```

## Performance Benchmarks

### RAGAS Evaluation Results

**Test Set:** 16 questions across 4 research papers  
**Metrics:** Context Precision & Context Recall

| Configuration                  | Precision | Recall   | Latency   |
| ------------------------------ | --------- | -------- | --------- |
| Dense Only (top-10)            | 0.55      | 0.79     | ~800ms    |
| Hybrid (top-10)                | 0.68      | 0.81     | ~900ms    |
| **Hybrid + Reranking (top-3)** | **0.98**  | **0.86** | **~1.1s** |

## Future Improvements

-  **Conversation Persistence** - PostgreSQL for chat history storage
-  **Multi-user Support** - User authentication & private collections
-  **Streaming Responses** - Server-Sent Events for real-time token streaming

---


Thanks for checking out this project. If you find it useful, please consider giving it a star ⭐
