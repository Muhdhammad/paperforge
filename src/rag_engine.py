from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from src.config import CONFIG
from typing import Optional
import os

class RAG:
    def __init__(self, retriever, llm_name: str = "gpt-4-turbo", provider: str = "OPENAI", top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k
        self.llm_name = llm_name
        self.provider = provider
        self.llm = self._setup_llm()
        self.prompt_template_str = self.prompt_template()

    def _setup_llm(self):
        
        if self.provider == "OPENAI":
            llm =  ChatOpenAI(
                model=self.llm_name,
                max_completion_tokens=1024,
                temperature=0.1
            )
            return llm
        
        if self.provider == "GROQ":
            llm = ChatGroq(
                model=self.llm_name or "llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1024,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            return llm

    def prompt_template(self):
        return """You are analyzing a research paper to answer specific questions.

    Paper Context:
    {context}

    Question: {query}

    Provide a clear, accurate answer following these rules:

    1. ANSWER CONTENT:
    - Use only the information from the context above
    - Be specific: cite numbers, percentages, method names, or results when relevant
    - Use technical language appropriately for a research audience
    - Do not rephrase unless necessary; prefer wording from the context.


    2. IF INFORMATION IS MISSING:
    - State clearly: "The provided context does not contain information about [topic]."
    - Do NOT speculate or use external knowledge

    3. FORMAT:
    - Keep answers concise (2-4 sentences for simple questions, more for complex ones)
    - Use bullet points for lists or multiple components

    Answer:"""


    def generate_context(self, query, top_k: Optional[int] = None):
        contexts = []

        results, latency = self.retriever.search(query, top_k=top_k or self.top_k)

        for point in results:
            contexts.append(
                f"[Source: {point.payload.get('file_name')} | Chunk: {point.payload.get('chunk_index')}]\n"
                f"[Context: {point.payload.get('text')}]"
            )

        formatted_contexts = "\n\n --- \n\n".join(contexts)
        return formatted_contexts, results
            
    def generate_response(self, query: str, top_k: Optional[int] = None):
        contexts, results = self.generate_context(query=query, top_k=top_k)
        prompt = self.prompt_template_str.format(context=contexts, query=query)
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": [
                {
                    "paper": point.payload.get('file_name'),
                    "chunk_index": point.payload.get('chunk_index')
                }
                for point in results
            ]
        }


