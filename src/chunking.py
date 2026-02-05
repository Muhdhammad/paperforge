from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import uuid


class SemanticChunking:
    def __init__(
        self,
        embeddings,
        breakpoint_threshold: int = 85,
        min_chunk_size: int = 50,
    ):
        self.min_chunk_size = min_chunk_size
        self.chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold,
        )
    
    def get_uuid():
        return str(uuid.uuid4())

    def create_chunks(
        self,
        markdown_text: str,
        doc_id: str,
        file_name: str,
    ) -> list[Document]:

        raw_chunks = self.chunker.create_documents([markdown_text])
        chunks = self._merge_chunks(raw_chunks)

        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": doc_id,
                "file_name": file_name,
                "chunk_index": idx,
                "chunk_size": len(chunk.page_content),
                "total_chunks": len(chunks),
            })

        return chunks

    def _merge_chunks(self, chunks: list[Document]) -> list[Document]:
        
        if not chunks:
            return []

        merged_chunks = []
        buffer = ""

        for chunk in chunks:
            content = chunk.page_content.strip()

            if len(content) < self.min_chunk_size:
                buffer = f"{buffer} {content}".strip()
            else:
                if buffer:
                    content = f"{buffer} {content}".strip()
                    buffer = ""
                merged_chunks.append(Document(page_content=content))

        if buffer:
            if merged_chunks:
                merged_chunks[-1].page_content += f" {buffer}"
            else:
                merged_chunks.append(Document(page_content=buffer))

        return merged_chunks

