from src.chunking import MarkdownChunking
from src.embedding import Embedding, SparseEmbedding
from src.vectordb import QdrantVDB
from pathlib import Path
from src.utils import get_uuid
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def ingest_markdown_to_qdrant(input_path: Path):
    """
    Ingest markdown files into Qdrant.
    
    Pipeline:
        1. Chunk markdown files using heading-based splitting
        2. Generate dense and sparse embeddings
        3. Upload chunks with vectors and payload to Qdrant

    """

    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_path}")
    
    md_files = list(input_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files available: {input_path}")
    print(f"Total markdown files: {len(md_files)}")

    try:
        dense_embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=2)
        sparse_embed = SparseEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        logger.info("Dense and Sparse embedding model loaded")

    except Exception as e:
        logger.error(f"failed to load embedding models: {e}")
        raise

    try:
        qdrantvdb = QdrantVDB(collection_name="research-papers-arxiv", vector_dim=768)
        qdrantvdb.create_collection()

    except Exception as e:
        logger.error(f"failed to initialize qdrant vector db collection: {e}")
        raise
    
    chunker = MarkdownChunking(embed_model=dense_embed.model, chunker="heading")

    failed_files = []
    for i, md_file in enumerate(md_files, start=1):
        print(f"{i}/{len(md_files)} - Processing file: {md_file}")

        try:
            markdown_text = md_file.read_text(encoding="utf-8")

            doc_uuid = get_uuid()
            chunks = chunker.create_chunks(markdown_text=markdown_text,
                                doc_id=doc_uuid,
                                file_name=md_file.name)
            logger.info(f"Num chunks: {len(chunks)}")
            
            texts = [chunk.page_content for chunk in chunks]

            dense_vectors = dense_embed.embed_batch(texts=texts)
            sparse_vectors = sparse_embed.embed_batch(texts=texts)
            
            embed_docs = []
            for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
                embed_docs.append({
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    "payload": {
                        **chunk.metadata,
                        "text": chunk.page_content
                    }
                })

            qdrantvdb.upload(embed_docs)
            logger.info(f"Uploaded {len(embed_docs)} chunks from {md_file.name}")

        except Exception as e:
            logger.error(f"Failed processing: {md_file.name}: {e}")
            failed_files.append(md_file.name)
            continue
    
    if failed_files:
        logger.info(f"""Total failed markdown files: {len(failed_files)},
                    files failed: {failed_files} """)

if __name__ == "__main__":
    input_path = Path("md-research-papers")
    ingest_markdown_to_qdrant(input_path)
