from langchain_openai import ChatOpenAI
from openai import OpenAI 
from datasets import Dataset
from pathlib import Path
import logging
import json

from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.embedding import Embedding, SparseEmbedding
from src.vectordb import QdrantVDB
from src.retriever import Retriever
from src.config import CONFIG

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

def initialize():
    try:
        dense_embed = Embedding(model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=2)
        sparse_embed = SparseEmbedding()
        qdrantvdb = QdrantVDB(collection_name="research-papers-arxiv", vector_dim=768)
        return Retriever(vectordb=qdrantvdb, dense_embed=dense_embed, sparse_embed=sparse_embed)
    except Exception as e:
        raise ValueError(f"Initialization failed: {e}")
    

def retrieve_context(retriever, query: str, top_k: int = 10):
    results, latency = retriever.search(
        query=query,
        top_k=top_k,
        use_hybrid_search=True,
        reranking=True
    )

    logger.info(f"Retrieved contexts in {latency:.2f}s")
    return [point.payload["text"] for point in results]

def build_dataset(retriever, content):

    data = []

    for paper in content:
        logger.info(f"Processing paper: {paper['paper']}")
        for qa_pair in paper["qa_pairs"]:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            contexts = retrieve_context(retriever=retriever, query=question, top_k=10)

            data.append({
                "question": question, 
                "ground_truth": ground_truth,
                "contexts": contexts
            })
    
    dataset = Dataset.from_list(data)
    return dataset

def run_ragas_eval(dataset: Dataset):
    try:
        llm = LangchainLLMWrapper(ChatOpenAI(
            model="llama-3.1-8b-instant",  # or "mixtral-8x7b-32768" "llama-3.1-8b-instant"
            base_url="https://api.groq.com/openai/v1",
            api_key=CONFIG.GROQ_API_KEY,
            temperature=0.0,
            max_completion_tokens=4096,
            timeout=180
        ))
        results = evaluate(
            dataset=dataset,
            metrics=[ContextPrecision(), ContextRecall()],
            llm=llm,
            run_config=RunConfig(max_workers=1, timeout=180)
        )
        return results

    except Exception as e:
        raise ValueError(f"Ragas Evaluation failed: {e}")

def main():

    qa_path = Path("test/test-json/qa_pairs.json")
    with qa_path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    ret = initialize()
    testset = build_dataset(retriever=ret, content=content)
    results = run_ragas_eval(dataset=testset)
    print(results)

    results_df = results.to_pandas()
    print(results_df[['user_input', 'context_precision', 'context_recall']])

    results_df.to_csv("retriever_eval_results", index=False)


if __name__ == "__main__":
    main()
