from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Embedding:

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: int = 4,
        device: str = "cpu", # "cuda" if gpu
        cache_folder: str = "./hf_cache"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.cache_folder = cache_folder
        self.model = self._load_model()
        #self.vector_dim = self.get_vector_dim()


    def _load_model(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device,
                          "trust_remote_code": True},
            encode_kwargs={"batch_size": self.batch_size,
                           "normalize_embeddings": True,
                           "truncate_dim": 768},
            cache_folder=self.cache_folder
        ) 
    
    # def get_vector_dim(self):
        # return self.model.client.get_sentence_embedding_dimension()
    
    def embed_text(self, query: str) -> list:
        return self.model.embed_query(query)
    
    @staticmethod
    def batch_iterate(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i: i + batch_size]

    def embed_batch(self, texts: list[str]) -> list:

        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch in tqdm(self.batch_iterate(texts, self.batch_size), total = total_batches, desc=f"Embedding {total_batches} batches"):

            batch_embeddings = self.model.embed_documents(batch)
            embeddings.extend(batch_embeddings)

        return embeddings
    
class SparseEmbedding:

    def __init__(self, model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"):
        self.model_name = model_name
        self.model = SparseTextEmbedding(model_name=self.model_name)
        logger.info(f"Sparse embedding model loaded")

    def embed_text(self, text: str) -> SparseVector:
        embeddings = next(self.model.embed([text])) # model.embed returns generator
        
        return SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist()
        )
    
    def embed_batch(self, text: list[str]) -> list[SparseVector]:
        embeddings = list(self.model.embed(text))

        return [
            SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist()
            )
            for emb in embeddings
        ]



if __name__ == "__main__":

    test_chunks = [
        Document(
            page_content="I'm Hammad and I have 02 cute cats",
            metadata={"doc_id": "1", "chunk_index": 0, "source": "test.pdf"}
        ),

        Document(
            page_content="# Introduction to Quantum Computing\n\nQuantum computing is an emerging technology that leverages the principles of quantum mechanics to perform computations beyond the capabilities of classical computers.",
            metadata={"doc_id": "2", "chunk_index": 2, "source": "test2.pdf"}
        )

    ]

    texts = [doc.page_content for doc in test_chunks]
    print(texts)

    embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=2)

    result = embed.embed_batch(texts=texts)

    for i in result:
        print(i)