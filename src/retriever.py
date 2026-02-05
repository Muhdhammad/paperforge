from qdrant_client import models
import time

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
      results = self.vectordb.client.search(
          collection_name=self.vectordb.collection_name,
          query_vector=embed_query,
          limit=top_k,
          search_params=models.SearchParams(
              quantization=models.QuantizationSearchParams(
                  ignore=True,
                  rescore=True,
                  oversampling=2.0
              )
          ),
          timeout=1000,
      )

      latency = time.time() - start_time
      return results, latency

    except Exception as e:
      raise RetrievalError(f"Dense retrieval failed for query: {query}") from e