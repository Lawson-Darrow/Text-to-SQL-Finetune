# Security Policy

## Supported versions

This is research-grade software under active development. Only the latest `main`
is supported; there are no backported fixes.

## Reporting a vulnerability

Please report security issues privately via
[GitHub Security Advisories](https://github.com/Lawson-Darrow/Text-to-SQL-Finetune/security/advisories/new)
rather than a public issue. We will acknowledge and respond as soon as we can.

## Scope notes

The frontier baselines send your natural-language questions and database schemas
to whatever LLM provider you configure (traffic routes through LLMGateway, an
OpenAI-compatible endpoint). Do not run them on sensitive schemas with a provider
you do not trust. Fine-tuning and the open-model evaluation run locally and send
nothing externally. Provider keys live in `.env`, which is gitignored; never
commit real keys.

Generated SQL is executed against the Spider databases during evaluation. Point
the executor only at the throwaway evaluation databases, never at a production or
sensitive database.
