from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import logging

from src.rag_engine import RAG
from src.retriever import Retriever
from src.vectordb import QdrantVDB
from src.embedding import Embedding

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        qdrantvdb = QdrantVDB(collection_name="research-pprs")
        ret = Retriever(vectordb=qdrantvdb, embed_text=embed.embed_text)
        rag = RAG(retriever=ret, llm_name="llama-3.1-8b-instant", provider="GROQ")
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        raise 

    app.state.rag = rag
    app.state.retriever = ret
    app.state.embedding = embed

    logger.info("All initializations complete")
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
    top_k: int = 3 

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/query', response_model=QueryResponse)
def user_query(request: QueryRequest):
    try:
        result = app.state.rag.generate_response(query=request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)



