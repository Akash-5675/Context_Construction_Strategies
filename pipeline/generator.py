"""
Phase 5 — Answer generation.

Calls Cloudflare Workers AI (llama-3.1-8b-instruct) with temperature=0
for reproducibility. Free tier: 10,000 requests/day, no credit card needed.
"""

from __future__ import annotations
import os
import time
import urllib.request
import urllib.error
import json

from dotenv import load_dotenv

load_dotenv()

MODEL = "@cf/meta/llama-3.1-8b-instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 512
RATE_LIMIT_SLEEP = 1
RETRY_BACKOFF = [2, 5, 10, 20]

# Once the daily quota is gone every call fails, and grinding through the
# remaining rows at 5 retries each wastes hours. Abort instead.
MAX_CONSECUTIVE_FAILURES = 10
_consecutive_failures = 0

SYSTEM_PROMPT = (
    "You are a precise biomedical assistant. Answer the question using ONLY "
    "the provided context. If the context does not contain enough information "
    "to answer, say 'Insufficient context.' Do not use external knowledge."
)


def generate(query: str, chunks: list[dict]) -> dict:
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")

    if not api_token or not account_id:
        raise EnvironmentError("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must be set in .env")

    context = "\n\n".join(
        f"[Source {i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    context_word_count = len(context.split())
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    global _consecutive_failures

    answer = "GENERATION_FAILED"
    latency_ms = 0.0

    t0 = time.perf_counter()
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                result = json.loads(resp.read().decode("utf-8"))
                answer = result["result"]["response"].strip()
            break
        except urllib.error.HTTPError as e:
            # 429 = quota/rate limit, 5xx = transient server error; both worth retrying
            if e.code == 429 or e.code >= 500:
                if attempt < 4:
                    time.sleep(RETRY_BACKOFF[attempt])
            else:
                raise
        except Exception:
            if attempt < 4:
                time.sleep(RETRY_BACKOFF[attempt])
            else:
                raise

    if not answer or answer == "GENERATION_FAILED":
        _consecutive_failures += 1
        if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise SystemExit(
                f"\nAborting: {_consecutive_failures} consecutive generation failures.\n"
                f"The daily quota is almost certainly exhausted. All completed rows are "
                f"saved — rerun the same command after the quota resets (00:00 UTC / "
                f"05:30 IST) and it will resume from here."
            )
        raise RuntimeError("Generation failed after 5 attempts — row skipped, will retry next run")

    _consecutive_failures = 0

    time.sleep(RATE_LIMIT_SLEEP)

    return {
        "answer": answer,
        "context_word_count": context_word_count,
        "latency_ms": latency_ms,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
