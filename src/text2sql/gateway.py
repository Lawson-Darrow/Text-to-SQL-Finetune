"""Frontier SQL generation via LLMGateway (OpenAI-compatible) for the frontier baseline.

Same prompt (SYSTEM + schema+question) and SQL extraction as the local path, so the
frontier-vs-open comparison is apples-to-apples. Concurrent requests since the gateway
has no batch-generation API.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from openai import OpenAI

from text2sql.data import SYSTEM
from text2sql.infer import extract_sql


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("LLMGATEWAY_API_KEY")
    if not key:
        raise RuntimeError("LLMGATEWAY_API_KEY not set (see .env)")
    return OpenAI(api_key=key, base_url=os.environ.get("LLMGATEWAY_BASE_URL", "https://api.llmgateway.io/v1"))


def _one(model: str, prompt: str) -> str:
    try:
        r = _client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        return extract_sql(r.choices[0].message.content or "")
    except Exception:  # noqa: BLE001
        return "SELECT 1"


def generate_sql_gateway(model: str, prompts: list[str], workers: int = 8) -> list[str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: _one(model, p), prompts))
