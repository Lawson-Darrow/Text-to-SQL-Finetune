"""LoRA bf16 SFT for text-to-SQL (Milestone 5), via trl SFTTrainer + peft.

Trains on chat-formatted (schema+question -> SQL) pairs with assistant-only loss.
Runs locally on the RTX 4090 (Ada). Saves a LoRA adapter for eval/merge.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from text2sql.data import SYSTEM, Example, Schema, build_prompt, serialize_schema


def build_messages(examples: list[Example], schemas: dict[str, Schema], style: str) -> list[dict]:
    """Each row: a chat with the gold SQL as the assistant turn."""
    rows = []
    for ex in examples:
        prompt = build_prompt(ex.question, serialize_schema(schemas[ex.db_id], style))
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ex.query},
                ]
            }
        )
    return rows


def finetune(
    model_name: str,
    train_rows: list[dict],
    output_dir: str,
    epochs: float = 1.0,
    lr: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 8,
    lora_r: int = 16,
    max_length: int = 768,
) -> str:
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=False,  # 1.5B fits in 24GB; avoids peft+checkpointing grad gotcha
        logging_steps=25,
        save_strategy="no",
        max_length=max_length,
        assistant_only_loss=False,  # robust first run (Qwen template lacks generation markers)
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(train_rows),
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()

    adapter_dir = str(Path(output_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    return adapter_dir
