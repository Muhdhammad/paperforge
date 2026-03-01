from qdrant_client import models
from typing import Optional
from src.embedding import Embedding, SparseEmbedding
from src.vectordb import QdrantVDB
from src.config import CONFIG
import time
import requests
import logging

logger = logging.getLogger(__name__)

class RetrievalError(Exception):
  "Raised when retrieval fails."
  pass

class Retriever:

  def  __init__(
      self,
      vectordb,
      dense_embed,
      sparse_embed,
  ):
    self.vectordb = vectordb
    self.dense_embed = dense_embed
    self.sparse_embed = sparse_embed

  def search(self, query: str, top_k: int = 5, paper_filter: Optional[str] = None, use_hybrid_search: bool = False, reranking: bool = False):

    try:
      start_time = time.time()
      search_filter = None
      if paper_filter: 
        search_filter = models.Filter(
          must=[
            models.FieldCondition(
              key="file_name",
              match=models.MatchValue(value=paper_filter)
            )
          ]
        )

      if use_hybrid_search:
        # dense and sparse query vector
        dense_vector = self.dense_embed.embed_text(query)
        sparse_vector = self.sparse_embed.embed_text(query)

        results = self.vectordb.client.query_points(
          collection_name=self.vectordb.collection_name,
          prefetch=[
            models.Prefetch(
              query=dense_vector,
              using=self.vectordb.dense_vector_name,
              limit=top_k * 2
            ),
            models.Prefetch(
              query=sparse_vector,
              using=self.vectordb.sparse_vector_name,
              limit=top_k * 2

            )
          ],
          query = models.FusionQuery(fusion=models.Fusion.RRF),
          limit=top_k,
          query_filter=search_filter
        ).points

      else:
        # dense vector search only
        dense_vector = self.dense_embed.embed_text(query)

        results = self.vectordb.client.query_points(
            collection_name=self.vectordb.collection_name,
            query=dense_vector,
            using=self.vectordb.dense_vector_name,
            limit=top_k,
            query_filter=search_filter,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0
                )
            ),
            timeout=1000,
        ).points

      # 2nd stage reranking with Jina reranker v3
      if reranking:
        try:
          results = self.rerank_with_jina(query=query, results=results, top_n=3)
        except Exception as e:
          logger.warning("Reranking failed: {e}. Falling back to original rankings")


      latency = time.time() - start_time
      return results, latency

    except Exception as e:
      raise RetrievalError(f"Retrieval failed for query: {query}: {e}")
    

  def rerank_with_jina(self, query: str, results: list, top_n: int = 3) -> list:  # results are the retrived points

    if not results:
      raise ValueError("No retrived docs found")

    try:
      url="https://api.jina.ai/v1/rerank"
      headers={
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {CONFIG.JINA_API}"
      }
      data = {
        "model": "jina-reranker-v3",
        "query": query,
        "top_n": top_n,
        "documents": [point.payload["text"] for point in results]
      }

      response = requests.post(url=url, headers=headers, json=data, timeout=30)
      
      if response.status_code != 200:
        raise ValueError(f"Jina reranking failed {response.text}")
      
      reranked = response.json()["results"]

      reranked_result = []
      # Return ScoredPoint in reranked order
      for r in reranked:
        idx = r["index"]
        point = results[idx]
        relevance_score = r["relevance_score"]
        point.payload["jina_score"] = relevance_score # Add relevance score in point payload
        reranked_result.append(point)

      return reranked_result
    
    except Exception as e:
      logger.error(f"Jina reranking error: {e}")
      raise

if __name__ == "__main__":

  query="What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?"
  dense_embed = Embedding()
  sparse_embed = SparseEmbedding()
  qdrantvdb = QdrantVDB(collection_name="research-papers-arxiv")
  ret = Retriever(vectordb=qdrantvdb, dense_embed=dense_embed, sparse_embed=sparse_embed)
  res, latency = ret.search(query=query, top_k=10, use_hybrid_search=True, reranking=True)
  print(res)
  print(latency)