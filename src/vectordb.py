from qdrant_client import QdrantClient, models
from src.utils import get_uuid, batch_iterate
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class CollectionAlreadyExists(Exception):
    pass

class CollectionDoesntExist(Exception):
    pass

class CollectionCreationError(Exception):
    pass

class UploadError(Exception):
    pass

class QdrantVDB:
    def __init__(self, collection_name: str, vector_dim: int = 768, dense_vector_name: str = "dense", sparse_vector_name: str = "sparse", url: str = "http://localhost:6333",):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.client = QdrantClient(
            url=url,
            prefer_grpc=True
        )

    def create_collection(self):
        if self.client.collection_exists(self.collection_name):
            logger.warning(f"Collection {self.collection_name} already exists, skipping creation.")
            return

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_vector_name: models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    self.sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams()
                    )
                },
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                )
            )
            logger.info(f"Collection {self.collection_name} created with {self.dense_vector_name} and {self.sparse_vector_name}")
        except Exception as e:
            raise CollectionCreationError("Error when creating collection") from e
        
    def upload(self, embed_docs: list[dict], batch_size: int = 32):

        if not embed_docs:
            raise ValueError("No documents found for upload")
        
        try:
            for batch in tqdm(batch_iterate(embed_docs, batch_size),
                              total=(len(embed_docs) + batch_size - 1) // batch_size,
                              desc="Uploading batches to Qdrant"):

                points = [models.PointStruct(  # Qdrant expects a list of points
                    id=get_uuid(),
                    vector={
                        self.dense_vector_name: doc["dense_vector"],
                        self.sparse_vector_name: doc["sparse_vector"]
                    },
                    payload=doc["payload"]
                ) for doc in batch
                ]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

        except Exception as e:
            raise UploadError(f"Failed to upload documents to Qdrant") from e
        
    def list_documents(self):
        """List all the documents in knowledge base"""

        seen_docs = set()
        file_names = []
        offset = None

        try:
            while True:
                results, offset = self.client.scroll(collection_name=self.collection_name,
                                                    limit=100,
                                                    with_payload=True,
                                                    with_vectors=False,
                                                    offset=offset)
                for point in results:
                    file_name = point.payload.get("file_name")
                    if file_name and file_name not in seen_docs:
                        seen_docs.add(file_name)
                        file_names.append(file_name)
                
                if offset is None:
                    break
            
            return file_names

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    def delete_document(self, file_name: str):
        """Delete all the points for a specific file"""
        
        try:
            self.client.delete(collection_name=self.collection_name,
                            points_selector=models.Filter(
                                must=[models.FieldCondition(
                                    key="file_name",
                                    match=models.MatchValue(value=file_name)
                                )]
                            )
                            )
            logger.info(f"All points deleted for {file_name}")
        except Exception as e:
            logger.error(f"Failed to delete points for {file_name}: {e}")
            raise
            
    def check_health(self):
        """Check for broken points without payload"""

        if not self.client.collection_exists(collection_name=self.collection_name):
            raise CollectionDoesntExist(f"No collection exists {self.collection_name}")

        broken_points = 0
        offset = None
        batch_num = 0
        total_points = self.client.count(collection_name=self.collection_name).count
        print(f"Total points: {total_points}")

        while True:
            points, offset = self.client.scroll(collection_name=self.collection_name,
                                                limit=100,
                                                offset=offset)
            if not points:
                break
            
            batch_num +=1
            for p in points:
                if not p.payload:
                    broken_points += 1 
            print(f"Batch {batch_num}: processed {batch_num * 100}, broken so far: {broken_points}")

            if offset is None:
                break

        print(f"Total broken points without payload: {broken_points}")



if __name__ == "__main__":
    qdrantvdb = QdrantVDB(collection_name="research-papers-arxiv")
    qdrantvdb.check_health()

        