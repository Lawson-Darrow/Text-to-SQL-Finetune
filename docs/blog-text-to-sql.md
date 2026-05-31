---
title: "What actually makes a small model better at SQL? I measured three levers."
date: 2026-05-31
tags: [llm, fine-tuning, text-to-sql, lora, evaluation]
summary: I fine-tuned open Qwen2.5-Coder models for text-to-SQL and isolated three levers — model size, LoRA fine-tuning, and schema representation. Size dominates, a one-epoch LoRA buys half a "size step," and a 100-example eval slice nearly fooled me.
---

# What actually makes a small model better at SQL? I measured three levers.

Plenty of projects fine-tune an LLM for text-to-SQL and report a single accuracy number.
That number doesn't tell you *what made it better*. So I built this to isolate the levers
and measure each one — with the official Spider evaluator and real confidence intervals.

The setup: open **Qwen2.5-Coder** models (1.5B / 3B / 7B), Spider, LoRA fine-tuning on a
single RTX 4090, and execution accuracy (run the predicted query, check it returns the
right rows). Three questions: does size matter, does fine-tuning matter, does the way you
describe the schema matter?

## 1. Size dominates

Base execution accuracy climbs cleanly with size: **1.5B 0.56 → 3B 0.69 → 7B 0.79** on the
full Spider dev set, confidence intervals non-overlapping. No surprise, but it sets the
scale for everything else: the other levers are measured against this gap.

## 2. A one-epoch LoRA buys half a "size step"

Fine-tuning the 1.5B lifted it from **0.56 to 0.63** — a real, significant +7 points. The
fine-tuned 1.5B lands halfway to the *3B base* in the accuracy-vs-size curve. That's the
efficiency headline: a cheap LoRA on a tiny model claws back a meaningful fraction of what
you'd otherwise pay for by doubling parameters.

## 3. Schema representation is a smaller, sneakier lever

How you serialize the database schema into the prompt matters — and not how I'd have
guessed. Adding column **types** consistently *hurt* execution at both sizes. Adding
primary/foreign **keys** helped the model match gold SQL style. A bare `minimal` schema
already got the 7B to ~0.91. More context isn't automatically better.

## The part that almost fooled me

My first fine-tuning eval ran on 100 examples and showed execution accuracy basically
*flat* (even slightly down) — while exact-match jumped 17 points. The tidy story would
have been "fine-tuning changes style, not correctness." It would have been wrong. On the
full 1,034-example dev set with bootstrap confidence intervals, the fine-tuning gain was
clearly positive and significant. **The 100-example slice was just noise.**

That's the real lesson, and it's why every headline number here carries a CI. The point of
the project was never the fine-tune — it was the measurement discipline that keeps you from
shipping a confident wrong conclusion.

Code, full results, and method: [github.com/Lawson-Darrow/Text-to-SQL-Finetune](https://github.com/Lawson-Darrow/Text-to-SQL-Finetune).
