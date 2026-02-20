from chunking import MarkdownChunking
from embedding import Embedding
from langchain_core.documents import Document
from vectordb import QdrantVDB
from retriever import Retriever
from rag_engine import RAG
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from utils import get_uuid
import logging

logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# main.py
from dotenv import load_dotenv
load_dotenv()

"""
test_chunks = [
    Document(
        page_content="I'm Hammad and I have 02 cute cats",
        metadata={"doc_id": "9846935d-9025-4978-84f4-70dffa3b669a", "chunk_index": 0, "source": "test.pdf"}
    ),

    Document(
        page_content="# Introduction to Quantum Computing\n\nQuantum computing is an emerging technology that leverages the principles of quantum mechanics to perform computations beyond the capabilities of classical computers.",
        metadata={"doc_id": "9848335d-9025-4978-84f4-70dffa3b669a", "chunk_index": 2, "source": "test2.pdf"}
    )

]
# query = "Explain the bipartite matching loss used in DETR training."
"""


def ingest_markdown_to_qdrant(input_path: Path):

    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_path}")
    
    md_files = list(input_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files available: {input_path}")
    print(f"Total markdown files: {len(md_files)}")

    try:
        hf_embed = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"device": "cpu", "trust_remote_code": True},  
            encode_kwargs={"batch_size": 4},
            cache_folder="./hf_cache"
        )
        embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=2)

    except Exception as e:
        logger.error(f"failed to load embedding models: {e}")
        raise

    try:
        qdrantvdb = QdrantVDB(collection_name="research-papers", vector_dim=768)
        qdrantvdb.create_collection()

    except Exception as e:
        logger.error(f"failed to initialize qdrant vector db collection: {e}")
        raise
    
    chunker = MarkdownChunking(embeddings=hf_embed, chunker="heading")

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
            
            result = embed.batch_embedding(docs=chunks)

            qdrantvdb.upload(result)

        except Exception as e:
            logger.error(f"Failed processing: {md_file.name}")
            failed_files.append(md_file.name)
            continue
    
    if failed_files:
        logger.info(f"""Total failed markdown files: {len(failed_files)},
                    files failed: {failed_files} """)



input_path = Path("md-research-papers")
ingest_markdown_to_qdrant(input_path)



"""
md_path = Path("md-research-papers/End-to-End_Object_Detection_with_Transformers.md")
markdown_text = md_path.read_text(encoding="utf-8")

chunks = chunker.create_chunks(markdown_text=markdown_text,
                               doc_id=doc_uuid,
                               file_name=md_path.name)

embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=2)

result = embed.batch_embedding(docs=chunks)

#print("Num chunks:", len(chunks))
#print("First chunk:", chunks[0])

qdrantvdb = QdrantVDB(collection_name="research-papers-testing12", vector_dim=768)

ret = Retriever(vectordb=qdrantvdb, embed_text=embed.embed_text)
#qdrantvdb.create_collection()
#qdrantvdb.upload(result)

result, latency = ret.search(query=query, top_k=3)
print(f"Retrieved chunks are....\n {result} \n {latency} seconds")

#rag = RAG(retriever=ret)
#res = rag.generate_response(query=query)

#print(result)
# print(latency)

#print(res)
"""