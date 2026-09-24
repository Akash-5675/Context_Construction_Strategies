"""
Proposition chunker — Phase 3.4.

Decomposes each abstract into atomic facts (propositions) using the Groq API
(llama-3.1-8b-instant). Each proposition is a single, self-contained factual
claim extracted from the source text. This is the most granular strategy and
is designed to maximise retrieval precision for specific clinical questions.

Condition key: proposition

Reference: Chen et al. (2023), "Dense X Retrieval: What Retrieval Granularity
Should We Use?" — atomic proposition retrieval outperforms sentence and passage
retrieval on factoid QA benchmarks.

Note: This chunker makes one Groq API call per document. For 1000 abstracts
at ~0.5s per call, expect ~8 minutes total. Results are cached in the output
.pkl file so the API is only called once.
"""

from __future__ import annotations
import os
import time
from dotenv import load_dotenv

load_dotenv()

_client = None

SYSTEM_PROMPT = (
    "You are a biomedical text analyst. Extract all atomic propositions from "
    "the given abstract. Each proposition must:\n"
    "1. Be a single, self-contained factual statement.\n"
    "2. Include enough context to be understood in isolation (expand pronouns, "
    "include the subject explicitly).\n"
    "3. Be derived directly from the text — do not infer or add external knowledge.\n\n"
    "Return ONLY a numbered list, one proposition per line. No preamble."
)

MODEL = "llama-3.1-8b-instant"
RATE_LIMIT_SLEEP = 0.5


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def _decompose(abstract: str) -> list[str]:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": abstract},
        ],
        temperature=0,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()
    propositions = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading numbering like "1.", "1)", "- "
        for prefix in ["- ", "• "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            dot = line.find(".")
            paren = line.find(")")
            cut = -1
            if dot != -1 and dot < 4:
                cut = dot + 1
            elif paren != -1 and paren < 4:
                cut = paren + 1
            if cut != -1:
                line = line[cut:].strip()
        if len(line.split()) >= 4:
            propositions.append(line)
    return propositions


class PropositionChunker:
    condition_key = "proposition"

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        propositions = _decompose(text)
        time.sleep(RATE_LIMIT_SLEEP)
        chunks = []
        for idx, prop in enumerate(propositions):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}__{self.condition_key}_{idx}",
                    "doc_id": doc_id,
                    "text": prop,
                    "word_count": len(prop.split()),
                    "strategy": self.condition_key,
                    "chunk_index": idx,
                }
            )
        return chunks
