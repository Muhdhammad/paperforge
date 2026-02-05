from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

class Embedding:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
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
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": self.batch_size},
            cache_folder=self.cache_folder
        ) 
    
    # def get_vector_dim(self):
        # return self.model.client.get_sentence_embedding_dimension()
    
    def embed_text(self, query: str):
        return self.model.embed_query(query)
    
    @staticmethod
    def batch_iterate(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i: i + batch_size]

    def batch_embedding(self, docs: list[Document]):

        data = []
        total_batches = (len(docs) + self.batch_size - 1) // self.batch_size

        for batch in tqdm(self.batch_iterate(docs, self.batch_size), total = total_batches, desc=f"Embedding {total_batches} batches"):
            contexts = [doc.page_content for doc in batch]
            vector_embeds = self.model.embed_documents(contexts)

            for doc, vector_embed in zip(batch, vector_embeds):
                data.append({
                    "vector": vector_embed,
                    "payload": {**doc.metadata, "text": doc.page_content}
                })

        return data

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

    embed = Embedding(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=2)

    result = embed.batch_embedding(docs=test_chunks)

    for i in result:
        print(i)