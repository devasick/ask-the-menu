# Ask the Menu

A multi-tenant RAG service that answers diners' questions about one restaurant's menu
("What's vegan and under €15?", "Enthält das Korma Nüsse?"), grounded only in that
restaurant's data. Built as a standalone service that [MenuNow](https://menunow.de) talks to over HTTP.

> **Status: MVP step 6 of 7 (evals).** Schema, ingestion, the provider-agnostic LLM client,
> the rules-first query parser (EN/DE), the SQL route and hybrid search (pgvector + full text,
> RRF), grounded generation behind `POST /chat`, per-request query logs with cost, `GET
> /admin/stats`, optional Langfuse tracing and the eval suite are in place. The widget follows. The full README (architecture, design decisions, eval results) lands in step 7.

## Quick start

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
make up      # app :8000, postgres+pgvector :5432, langfuse :3000
make seed    # ingests the two fictional demo restaurants from seed/ (set LLM_BACKEND=fake to run without API keys)
curl localhost:8000/health
make test    # spins up its own pgvector container, needs no API keys
make lint typecheck
```

## Data notes

- All seed data is fictional. `seed/curry_house.json` (40 items, 8 FAQs) is the eval restaurant;
  `seed/pizzeria_finta.json` (5 items) exists only for cross-tenant isolation tests.
- Allergens use the 14 EU allergen groups (Regulation 1169/2011, Annex II).
  `"allergens": null` means **unknown** and `[]` means **none of the 14**. The service never
  treats unknown as safe.

## Evaluation

`make eval` runs [100 EN/DE guest questions](evals/datasets/curry_house.jsonl) through the
real pipeline: filters, open questions, allergens, FAQs, off-topic, prompt injection and
another restaurant's dishes. It writes [evals/results/latest.md](evals/results/latest.md)
(per category, plus every case worth a look) and the table below. Allergen safety below
100% or any cross-tenant leakage fails the run. `make eval ARGS=--judge` adds an
LLM-as-judge faithfulness score from `menu-smart`.

<!-- eval:start -->
100 questions, 2026-11-25, backend: offline (`LLM_BACKEND=fake`: bag-of-words embeddings, stand-in model).

| Metric | Value | n | Gate |
|---|---|---|---|
| hit@5 | 80.0% | 50 |  |
| MRR | 0.770 | 50 |  |
| SQL filter exact match | 100.0% | 33 |  |
| Citation precision | n/a | 54 |  |
| Route accuracy | 100.0% | 100 |  |
| Refusal accuracy | n/a | 97 |  |
| Allergen safety | 100.0% | 24 | = 100% ✅ |
| Cross-tenant leakage | 0 | 100 | = 0 ✅ |
| Answered without generation | 33.0% | 100 |  |
| Latency p50 | 4 ms | 100 |  |
| Latency p95 | 6 ms | 100 |  |
| Avg cost per question | $0.000000 | 100 |  |
| Judge faithfulness | n/a | 0 |  |
<!-- eval:end -->

## Migrating to LiteLLM

All model access goes through one interface, `app/llm/base.py::LLMClient`
(`chat`, `embed`, `model_id`). The only implementation, `app/llm/providers.py`, is the only
module allowed to import the `anthropic` and `openai` SDKs (a test enforces this). App code
names aliases (`menu-fast`, `menu-smart`, `menu-backup`, `menu-embed`), never model IDs.

To move to LiteLLM, add an `LLMClient` that calls the LiteLLM SDK (or a LiteLLM proxy over its
OpenAI-compatible API) and return it from `app/llm/__init__.py::make_llm_client`. Map each
alias to a LiteLLM model name or proxy route, and let the gateway own retries, fallbacks and
per-restaurant budgets. Nothing else changes: the pipeline, eval runner and tests depend only on
the interface, and tests keep using `app/llm/fake.py`.

## License

MIT
