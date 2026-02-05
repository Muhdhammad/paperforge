from qdrant_client import QdrantClient, models
from embedding import Embedding
from tqdm import tqdm

class CollectionAlreadyExists(Exception):
    pass

class CollectionCreationError(Exception):
    pass

class UploadError(Exception):
    pass

class QdrantVDB:
    def __init__(self, collection_name: str, vector_dim: int = 384):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True
        )

    def create_collection(self):
        if self.client.collection_exists(self.collection_name):
            raise CollectionAlreadyExists("Collection already exists")

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                )
            )
        except Exception as e:
            raise CollectionCreationError("Error when creating collection") from e
        
    def upload(self, embed_docs: list[dict], batch_size: int = 32):

        if not embed_docs:
            raise ValueError("No documents found for upload")
        
        try:
            for batch in tqdm(Embedding.batch_iterate(embed_docs, batch_size),
                              total=(len(embed_docs) + batch_size - 1) // batch_size,
                              desc="Uploading batches to Qdrant"):

                points = [models.PointStruct(  # Qdrant expects a list of points
                    id=doc["payload"]["doc_id"],
                    vector=doc["vector"],
                    payload=doc["payload"]
                ) for doc in batch
                ]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

        except Exception as e:
            raise UploadError(f"Failed to upload documents to Qdrant") from e