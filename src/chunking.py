from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkdownChunking:
    def __init__(
        self,
        embed_model,
        chunker: str = "semantic",
        breakpoint_threshold: int = 85,
        min_chunk_size: int = 50,
    ):
        self.chunker_type = chunker
        self.min_chunk_size = min_chunk_size
        if self.chunker_type == "semantic":
            self.chunker = SemanticChunker(
                embeddings=embed_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=breakpoint_threshold,
            )

        elif self.chunker_type == "heading":
            self.chunker = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ],
            )
    
        else:
            raise ValueError(f"Unknown chunker {self.chunker_type}, use 'semantic' or 'heading" )
        logger.info(f"Chunker initialized: {self.chunker_type}")

        
    def create_chunks(
        self,
        markdown_text: str,
        doc_id: str,
        file_name: str,
    ) -> list[Document]:
        
        if self.chunker_type == "semantic":
            raw_chunks = self.chunker.create_documents([markdown_text])
            chunks = self._merge_chunks(raw_chunks)
        else:
            chunks = self.chunker.split_text(markdown_text)

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


