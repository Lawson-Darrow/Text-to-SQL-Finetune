"""Local generation: prompt -> SQL string, for base or fine-tuned HF models.

Used to produce predictions for the evaluator. Greedy decoding by default for
reproducibility. SQL extraction strips fences / prose so the evaluator sees clean SQL.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.data import SYSTEM


def load_model(name: str, dtype=torch.bfloat16, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only generation needs left padding
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    model.eval()
    return model, tok


def extract_sql(text: str) -> str:
    """Pull a single clean SQL statement out of a model completion."""
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^\s*sql\s*:?\s*", "", t, flags=re.I)
    if ";" in t:
        t = t.split(";")[0]
    return " ".join(t.split())


@torch.no_grad()
def generate_sql(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int = 192,
    batch_size: int = 8,
) -> list[str]:
    preds: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
            model.device
        )
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        for j in range(len(batch)):
            completion = tok.decode(
                gen[j][enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            preds.append(extract_sql(completion))
    return preds
