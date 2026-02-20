from openai import OpenAI
from pathlib import Path
import json
import logging
from src.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# client = OpenAI(api_key=CONFIG.OPENAI_API_KEY)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONFIG.GROQ_API_KEY,
)
                    
PROMPT = """
You are given a research paper in markdown format.

Generate exactly 4 questions and concise answers based on the paper.
One question of each type: factual, conceptual, results/limitations, multi_context.

Rules:
- Answers must be answerable ONLY from the paper content, no outside knowledge
- Answers should be 1-3 sentences max, concise
- Questions should require reading the paper to answer, not guessable
- multi_context question MUST require combining information from at least 2 different sections of the paper to answer (e.g. methodology + results, or problem statement + limitations)

Return ONLY valid JSON, no extra text:
{{
  "paper": "<filename>",
  "qa_pairs": [
    {{"type": "factual", "question": "...", "answer": "..."}},
    {{"type": "conceptual", "question": "...", "answer": "..."}},
    {{"type": "results/limitations", "question": "...", "answer": "..."}},
    {{"type": "multi_context", "question": "...", "answer": "..."}}
  ]
}}

Paper content:
{markdown_text}
"""

def generate_qa(md_file: Path, model: str = "llama-3.3-70b-versatile"):
    if not md_file.exists():
        raise ValueError(f"Markdown file not found {md_file}")

    markdown_text = md_file.read_text(encoding="utf-8")

    try:
        response = client.text_generation(
            model=model,
            messages=[
                {"role": "user", "content": PROMPT.format(markdown_text=markdown_text)}
            ],
            temperature=0.2,
        )

        raw = response.choices[0].message.content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)

    except Exception as e:
        logger.error(f"Error generating QA for {md_file.name}: {e}")
        raise

md_file = Path("test/test-papers/Attention_Is_All_You_Need.md")
generate_qa(md_file=md_file, model="llama-3.3-70b-versatile")

