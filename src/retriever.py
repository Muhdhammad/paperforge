from qdrant_client import models
import time
import os
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
      embed_text
  ):

    self.vectordb = vectordb
    self.embed_text = embed_text

  def search(self, query: str, top_k: int = 5):

    try:
      embed_query = self.embed_text(query)

      start_time = time.time()
      results = self.vectordb.client.query_points(
          collection_name=self.vectordb.collection_name,
          query=embed_query,
          limit=top_k,
          search_params=models.SearchParams(
              quantization=models.QuantizationSearchParams(
                  ignore=True,
                  rescore=True,
                  oversampling=2.0
              )
          ),
          timeout=1000,
      ).points

      latency = time.time() - start_time
      return results, latency

    except Exception as e:
      raise RetrievalError(f"Dense retrieval failed for query: {query}: {e}")
    

  def rerank_with_jina(self, query: str, results: list, top_n: int = 3) -> list:  # results are the retrived docs

    if not results:
      raise ValueError("No retrived docs found")

    try:
      url="https://api.jina.ai/v1/rerank"
      headers={
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {os.getenv('JINA_API')}"
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