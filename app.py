from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
import logging
import os

from src.rag_engine import RAG
from src.retriever import Retriever
from src.vectordb import QdrantVDB
from src.embedding import Embedding, SparseEmbedding

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        logger.info(f"Connecting to qdrant {qdrant_url}")

        dense_embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        sparse_embed = SparseEmbedding()
        qdrantvdb = QdrantVDB(collection_name="research-papers-arxiv", url=qdrant_url)

        ret = Retriever(vectordb=qdrantvdb, dense_embed=dense_embed, sparse_embed=sparse_embed)
        rag = RAG(
            retriever=ret,
            llm_name="llama-3.1-8b-instant",
            provider="GROQ",
            top_k=10,
            use_hybrid_search=True,
            reranking=True
        )

        # Store in app state
        app.state.qdrantvdb = qdrantvdb
        app.state.rag = rag
        app.state.retriever = ret
        app.state.dense_embed = dense_embed
        app.state.sparse_embed = sparse_embed

        logger.info("All initializations complete")

    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        raise 

    yield


app = FastAPI(title="PaperForge", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    paper_filter: Optional[str] = None
    chat_history: Optional[list] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.get('/health')
def health():
    return {"status": "ok"}

@app.get('/documents')
def get_documents():
    try:
        file_names = app.state.qdrantvdb.list_documents()
        return {"Documents": file_names, "Total Documents": len(file_names)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Error: {e}")
    
@app.post('/query', response_model=QueryResponse)
def user_query(request: QueryRequest):
    try:
        paper_filter = request.paper_filter if request.paper_filter else None
        result = app.state.rag.generate_response(query=request.query, top_k=request.top_k, paper_filter=paper_filter, chat_history=request.chat_history)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")
    
@app.delete('/documents/{file_name}')
def delete_document(file_name: str):
    try:
        app.state.qdrantvdb.delete_document(file_name)
        return {"message": f"{file_name} Deleted"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)



