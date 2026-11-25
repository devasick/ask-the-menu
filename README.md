# Ask the Menu

A multi-tenant RAG service that answers diners' questions about one restaurant's menu
("What's vegan and under €15?", "Enthält das Korma Nüsse?"), grounded only in that
restaurant's data. Built as a standalone service that [MenuNow](https://menunow.de) talks to over HTTP.

> **Status: MVP step 1 of 7 (foundation).** Schema, migrations, seed data, health check and
> test harness are in place. Ingestion, retrieval, generation, analytics, evals and the widget
> follow. The full README (architecture, design decisions, eval results) lands in step 7.

## Quick start

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
make up      # app :8000, postgres+pgvector :5432, langfuse :3000
make seed    # loads the two fictional demo restaurants from seed/
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

## License

MIT
