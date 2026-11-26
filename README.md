# Ask the Menu

A multi-tenant RAG service that answers diners' questions about one restaurant's menu, in
English and German: "What's vegan and under €15?", "Enthält das Korma Nüsse?", "Do you
deliver?". Answers come only from that restaurant's own data. It runs as a standalone service
that [MenuNow](https://menunow.de) (a digital menu SaaS) talks to over HTTP, and it ships with
a drop-in React chat widget.

**Status:** MVP (v0.1.0) is complete. Phase 2 is in progress: rate limits, budgets,
injection defense, streaming, caching with an FAQ route, and an optional cross-encoder reranker
are done.

## The problem

A guest scans the QR code on the table and wants a quick answer. The menu has the facts, but
they are spread across 40 dishes, in a language the guest may not read. A plain chatbot is the
wrong tool for this:

- **It invents dishes and prices.** Every answer must be traceable to a menu item or FAQ.
- **Allergen mistakes hurt people.** "Is this nut free?" can't be left to a language model's
  guess, and unknown allergen data must never read as "safe".
- **Many questions aren't questions for a model.** "Vegan under €15" is a database query.
  Sending it to an LLM is slower, costs more and is less reliable.
- **Many restaurants share one service.** Restaurant A's dishes must never show up for
  restaurant B, and each restaurant's cost must be visible.

## Architecture

```mermaid
flowchart LR
    W[Chat widget<br/>AskTheMenu.tsx] -- "POST /chat<br/>X-Widget-Key" --> API
    M[MenuNow backend] -- "POST /ingest/{id}<br/>X-API-Key" --> API
    D[MenuNow dashboard] -- "GET /admin/stats<br/>X-API-Key" --> API

    subgraph S[ask-the-menu · FastAPI]
        API[API layer] --> P[Rules-first parser<br/>EN/DE filters]
        P --> R{Route chain}
        R -- repeat or close paraphrase --> C[Answer cache<br/>exact + semantic]
        R -- pure filter --> SQL[SQL route<br/>templated answer]
        R -- FAQ question --> F[FAQ route<br/>restaurant's own answer]
        R -- everything else --> H[Hybrid search<br/>pgvector + FTS, RRF]
        H --> RR[Reranker<br/>no-op or cross-encoder]
        RR --> G[Generation<br/>JSON schema, citations]
        G --> GR[Guardrails<br/>citations, allergens]
        API --> L[(query_logs)]
        API --> RD[(Redis<br/>rate limits)]
        API -.-> LLM[app/llm<br/>aliases, fallback, cost]
    end

    SQL --> PG[(PostgreSQL 16<br/>pgvector)]
    C --> PG
    F --> PG
    H --> PG
    LLM --> AN[Anthropic: chat]
    LLM --> OA[OpenAI: embeddings]
    LLM -.-> LF[Langfuse<br/>self-hosted]
```

| Path | What lives there |
|---|---|
| `app/ingest/` | Payload schema, normalization, chunking, idempotent sync and embedding |
| `app/retrieval/` | Query parser, SQL search, hybrid search with RRF, reranker interface, route chain |
| `app/generation/` | Prompts, structured output, guardrails, templated SQL answers |
| `app/llm/` | `LLMClient` interface, provider SDKs, fallback, cost, fake client, Langfuse tracing |
| `app/api/` | `/health`, `/ingest`, `/chat`, `/admin/stats` |
| `evals/` | Dataset, runner and metrics |
| `widget/` | `AskTheMenu.tsx` |

## How a question is answered

Each request passes through a chain of routes. The first route that can answer it does, and
the route taken is logged.

1. **Parse (no LLM).** A rules-based EN/DE parser pulls out structured filters: price ("unter
   12 Euro"), dietary tags (vegan, glutenfrei, halal, spicy, mild), categories (starters,
   Nachtisch) and allergen exclusions ("without nuts", "Nussallergie", "lactose-free"). Only
   when it sees a price word it couldn't parse does it ask `menu-fast` for the filters.
   The same pass flags prompt injection attempts (see below), which then go to the
   **blocked route**: a fixed refusal with no retrieval and no model call.
   A question asked before, ignoring case, spacing and end punctuation, is then answered
   from the **exact cache** (see Caching below).
2. **SQL route (no LLM).** A question that is nothing but filters, like "Vegane Currys unter
   15 €", becomes a single SQL query. The answer is a template plus item cards. A third of the
   eval set is answered this way, at zero model cost.
3. **FAQ route and semantic cache (embedding only).** The question is embedded once, and that
   vector is reused by every later step.
   - The **FAQ route** compares it with each FAQ *question*, embedded on its own at ingest.
     At cosine similarity 0.85 or more, and in the FAQ's language, the guest gets the
     restaurant's own answer, word for word.
   - The **semantic cache** returns a stored answer to an earlier question at similarity 0.95
     or more.
   - Questions with filters or allergens skip both, because "dishes with nuts" and "dishes
     without nuts" embed close together but need opposite answers. They can only hit the
     exact cache.
4. **RAG route.** Everything else runs hybrid retrieval:
   - pgvector cosine search and Postgres full text search (`simple` config, so German and
     English tokens both survive), with any parsed filters applied as SQL `WHERE` clauses
   - the two rankings fused with Reciprocal Rank Fusion (k = 60)
   - the top 20 go through the reranker (see [Reranker](#reranker)), and the top 5 go to
     `menu-fast`
   - multi-part questions escalate to `menu-smart`
5. **Guardrails.**
   - The model must return JSON (answer, cited item IDs, confidence), validated with one
     retry. If validation fails again, the guest gets the retrieved item cards instead.
   - Citations are filtered to the items that were actually retrieved.
   - For allergen questions, the model only picks the dishes. The allergen sentence itself is
     built in code from stored data, and a disclaimer is always attached.

**Caching.** A confident generated answer is stored in Postgres (`answer_cache`) under the
restaurant's `menu_version`, language and normalized question. Fallback, low-confidence and
refused answers are never stored. An ingest that changes anything bumps `menu_version` and
deletes the restaurant's entries, so a price or allergen change is never answered from the
cache. Both caches live in Postgres rather than Redis: the semantic cache needs pgvector, and
one table lets the ingest transaction invalidate the cache atomically. The two similarity
thresholds (`FAQ_MIN_SIMILARITY`, `SEMANTIC_CACHE_MIN_SIMILARITY`) are set conservatively and
are not yet tuned against a real embedding model.

## Design decisions

- **Rules before models.** Filters are parsed deterministically and checked by SQL. A model
  can misread "under 15", but a regex plus a `WHERE price_cents <= 1499` can't.
- **Unknown allergens are never safe.** `allergens: null` or `[]` means "unknown, ask the
  staff". Such items never appear in a "without X" list, and the widget shows an "allergens
  unknown" badge. Nothing in the pipeline can turn missing data into a safety claim.
- **Tenant isolation in every query.** Every tenant table carries `restaurant_id`, every
  composite index starts with it, and a test captures every SQL statement the chat path runs
  and fails if one touches a tenant table without filtering by it.
- **Idempotent, incremental ingestion.** MenuNow sends the whole menu on each publish. Items
  upsert by `external_id`, and a chunk is re-embedded only when its content or the embedding
  model changes (`content_hash = sha256(model + content)`).
- **One model interface, aliases only.** App code names `menu-fast`, `menu-smart`,
  `menu-backup` and `menu-embed`, never a model ID. The mapping and prices live in env. Only
  `app/llm/providers.py` may import a provider SDK, and a test enforces that.
- **Prompt injection defense in layers.**
  - **Detection before any model call.** A rules-based EN/DE detector
    (`app/security/injection.py`) catches attempts to override instructions, extract the
    prompt, change the assistant's role, or fake prompt markup and chat roles ("SYSTEM:",
    `</question>`, `[INST]`). It normalizes first, so zero-width characters and full-width
    letters don't hide anything. A guest asking about food never needs these phrasings, and
    the eval checks that none of its other questions trip the detector.
  - **Untrusted text stays data.** Menu text and the guest's question go into delimited,
    HTML-escaped sections of the prompt, and the system prompt says to ignore instructions
    inside them.
  - **Output check.** A generated answer that echoes the system prompt is replaced with a
    refusal. Citations were already limited to retrieved items, and allergen text is built in
    code.
- **Offline by default in tests.** `LLM_BACKEND=fake` swaps in deterministic hashed
  embeddings and a scripted model. The test suite and an offline eval run need no keys and
  never call a paid API.

## Evaluation

`make eval` runs [123 EN/DE guest questions](evals/datasets/curry_house.jsonl) through the
real pipeline:

| Category | Questions |
|---|---|
| filter | 20 |
| semantic | 28 |
| allergen | 22 |
| faq | 17 |
| repeats and rewordings (cache) | 4 |
| off-topic | 7 |
| prompt injection | 20 |
| another restaurant's dishes | 5 |

It writes [evals/results/latest.md](evals/results/latest.md), with a per-category breakdown
and every case worth a look, and the table below. The run fails if allergen safety or injection
refusal is below 100%, or if anything leaks across tenants. `make eval ARGS=--judge` adds an LLM-as-judge
faithfulness score from `menu-smart`.

The committed numbers come from the offline backend, so metrics that need a real model show
n/a. To get real numbers, set `LLM_BACKEND=direct` with API keys, then run `make seed` and
`make eval`.

<!-- eval:start -->
123 questions, 2026-11-26, backend: offline (`LLM_BACKEND=fake`: bag-of-words embeddings, stand-in model), reranker: none.

| Metric | Value | n | Gate |
|---|---|---|---|
| hit@5 | 79.6% | 49 |  |
| MRR | 0.752 | 49 |  |
| SQL filter exact match | 100.0% | 33 |  |
| Citation precision | n/a | 51 |  |
| Route accuracy | 100.0% | 123 |  |
| Refusal accuracy | n/a | 118 |  |
| Allergen safety | 100.0% | 24 | = 100% ✅ |
| Injection refusal | 100.0% | 20 | = 100% ✅ |
| Cross-tenant leakage | 0 | 123 | = 0 ✅ |
| Answered without generation | 48.8% | 123 |  |
| Latency p50 | 8 ms | 123 |  |
| Latency p95 | 14 ms | 123 |  |
| Avg cost per question | $0.000000 | 123 |  |
| Judge faithfulness | n/a | 0 |  |
<!-- eval:end -->

### Reranker

`RERANKER=cross-encoder` scores each of the 20 fused candidates against the question with a
local cross-encoder, [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
(multilingual including German, Apache-2.0, runs on CPU), and keeps the top 5. It ships as
an optional extra so the default image stays small: `uv sync --extra rerank`, or
`docker build --build-arg EXTRAS=rerank .`. The default stays `none` until a real-model eval
shows it pays off.

Offline eval, same database and dataset, 49 ranked RAG cases:

| Reranker | hit@5 | MRR | Latency p95 (whole pipeline) |
|---|---|---|---|
| none (fused order) | 79.6% | 0.752 | 18 ms |
| cross-encoder | 81.6% | 0.748 | 282 ms |

The gain is small here, for a reason: a reranker can only reorder the 20 candidates it is
given, and offline the candidates come from bag-of-words vectors that never match across
languages. Most remaining misses are German questions ("Gibt es etwas mit Spinat?",
"Kann ich mit Kreditkarte zahlen?") whose right answer was never among the 20. The
cross-encoder fixed "Was ist euer schärfstes Curry?", and reshuffled the misses without
fixing them. The real test is a run with real embeddings, where the candidate pool is
better and reordering matters more. CPU scoring adds roughly 100 to 300 ms per RAG question.

The eval set has already earned its keep: it showed that "I have a nut allergy" and
"Nussallergie" were not read as exclusions (allergen safety 91.7%), which the parser now
handles.

## Cost control

- **No model where SQL will do.** Pure filter questions (a third of the eval set) cost
  nothing. Parsing is rules-first too.
- **Any provider per alias.** Each alias in `.env` is `<anthropic|openai|gemini>:<model id>`.
  Gemini goes through its OpenAI-compatible API (no extra SDK) and needs `GEMINI_API_KEY`.
  Embeddings can come from OpenAI or Gemini; Gemini is asked for 1536 dims, and a vector of
  any other size is rejected before it reaches the database.
- **A small model by default.** `menu-fast` answers. `menu-smart` only gets multi-part
  questions (a heuristic in `app/generation/guardrails.py`), and `menu-backup` takes over when
  `menu-fast`'s provider fails.
- **Bounded context.** Only the top 5 chunks go into the prompt, answers are 1 to 3
  sentences, and questions are capped at 500 characters.
- **Measured per request.** Every `/chat` logs tokens and `cost_usd`, computed from
  `MODEL_PRICES`. `GET /admin/stats` returns total and average cost per restaurant. A model
  with no configured price logs a warning instead of silently counting as free.
- **Re-embedding only on change.** Menu syncs embed only the chunks whose text changed.
- **Monthly budget with a soft landing.** Once a restaurant's chat cost this calendar month
  (UTC) reaches its `monthly_budget_usd`, `/chat` switches to search-only mode instead of
  failing. The guest still gets rules parsing, the SQL route, keyword search, item cards and
  allergen disclaimers, with no model calls. The same path keeps answers coming when the
  embedding provider is down.
- **Rate limits.** Each chat session gets 20 questions per hour by default, and each IP gets
  120 (guests often share a restaurant's Wi-Fi). Counters live in Redis. Going over returns
  `429` with `Retry-After`. If Redis is down, requests are allowed and a warning is logged,
  because the budget still caps spend.

## Quick start

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
make up       # app :8000, postgres+pgvector :5432, redis :6379, langfuse :3000 (copies .env.example to .env)
make seed     # ingests the two fictional demo restaurants from seed/
curl localhost:8000/health

curl -s localhost:8000/chat \
  -H 'Content-Type: application/json' -H 'X-Widget-Key: demo-widget-key-curry-house' \
  -d '{"restaurant_id": "11111111-1111-4111-8111-111111111111", "session_id": "demo",
       "question": "What is vegan and under 15 euros?"}'

make test     # spins up its own pgvector container, needs no API keys
make eval     # eval set against DATABASE_URL; re-seeds the demo restaurants
make lint typecheck
```

To run without API keys, set `LLM_BACKEND=fake` in `.env` before `make seed`. Answers are then
scripted and embeddings are not semantic, but every route, guardrail and log works. Switching
backends changes the embedding model, so run `make seed` again afterwards (it re-embeds).

## API reference

All bodies are JSON. Errors use FastAPI's `{"detail": ...}` shape.

### `POST /chat` (header `X-Widget-Key`)

```json
{ "restaurant_id": "uuid", "session_id": "1-100 chars", "question": "1-500 chars", "language": "en | de | null" }
```

`language` is optional and is detected from the question when omitted. The response:

```json
{
  "answer": "According to our menu, Chicken Korma contains: milk, nuts.",
  "items": [{ "id": "cu-03", "name": "Chicken Korma", "price": "15.50", "currency": "EUR",
              "allergens": ["milk", "nuts"], "dietary_tags": ["gluten_free", "halal"],
              "available": true }],
  "route": "rag",
  "sources": [{ "type": "menu_item", "id": "cu-03" }],
  "disclaimer": "Allergen information comes from the restaurant's menu data. ..."
}
```

- `items[].id` is MenuNow's own item ID.
- `allergens: null` means unknown.
- `route` is one of `blocked`, `exact_cache`, `sql`, `faq`, `semantic_cache` or `rag`.
- `disclaimer` is set whenever allergens come up.

Errors:
- **401:** unknown restaurant, missing key or wrong key (one response for all three, so it
  reveals nothing).
- **422:** invalid body.
- **429:** too many questions from this session or IP. Wait the number of seconds in
  `Retry-After`.

### `POST /chat/stream` (header `X-Widget-Key`)

Same request body as `/chat`. The response is `text/event-stream` with these events:

```text
event: delta
data: {"text": "Chicken Korma is "}

event: delta
data: {"text": "mild and creamy."}

event: done
data: { ...the same body as POST /chat... }
```

- `delta` carries the answer text as the model writes it. Only free-text RAG answers stream.
  SQL, blocked and allergen answers are written in code and arrive as a single `done`.
- `done` is authoritative. Citation and allergen checks run after generation and can change
  the text, so the client replaces the streamed text with `done.answer`.
- Streamed text stops as soon as it starts echoing the system prompt, and `done` then holds
  the refusal.
- `error` (`{"detail": ...}`) means the answer failed after the stream started.
- 401, 422 and 429 are plain HTTP errors sent before the stream starts.

If the main model fails before its first token, the request falls back to `menu-backup`. A
failure mid-answer ends with `error`, because the guest has already seen part of the text.

### `POST /ingest/{restaurant_id}` (header `X-API-Key`, ingest key)

This is a full menu sync. Items and FAQs missing from the payload are deleted. Seed files
show the shape (`seed/curry_house.json`):
- `restaurant`: id, slug, name, `default_language`, `monthly_budget_usd`, `info`, `widget_key`
- `items[]`: `external_id`, name, description, category, `price_cents`, currency, allergens
  (EU codes, or null for unknown), `dietary_tags`, `is_available`, language
- `faqs[]`: question, answer

The response counts what changed: items, FAQs, chunks embedded and deleted, embedding tokens
and cost, and the new `menu_version`.

Errors:
- **400:** the ID in the URL doesn't match the body.
- **409:** the slug belongs to another restaurant.
- **502:** the embedding provider is down. Nothing is written.

### `GET /admin/stats?restaurant_id=&from=&to=` (header `X-API-Key`, admin key)

The window defaults to the last 30 days. The response contains:
- question count and share per route
- top 10 questions, and questions with no result
- p50 and p95 latency
- tokens, and total and average cost in USD

### `GET /health`

Returns 200 when Postgres and pgvector answer, 503 otherwise.

## The widget

[`widget/AskTheMenu.tsx`](widget/AskTheMenu.tsx) is a single React 18 component with no UI
library and no other dependencies.

```tsx
import { AskTheMenu } from "./AskTheMenu";

<AskTheMenu
  apiUrl="https://ask.menunow.de"
  restaurantId="11111111-1111-4111-8111-111111111111"
  widgetKey="demo-widget-key-curry-house"
  language="de"
/>;
```

What it does:
- shows a floating button that opens the chat, full screen on phones and a 380px card from
  640px up
- streams the answer from `/chat/stream` as it is written, then swaps in the final reply
- shows item cards with price, allergen badges, an "allergens unknown, please ask staff"
  badge and a sold-out badge, plus the allergen disclaimer
- follows the system dark mode
- for accessibility: a labelled dialog, a polite live log, alerts for errors, Escape to close
  with focus returned to the button, 44px touch targets and visible focus rings

The widget key is public by design. It only lets a browser call `/chat` and `/chat/stream` for that one
restaurant.

## Integrating with MenuNow

1. **Create the restaurant on first publish.** MenuNow generates a random `widget_key` (at
   least 16 characters) per restaurant and sends it in `restaurant.widget_key`. Ask the Menu
   stores only its SHA-256.
2. **Sync on every menu change.** Call `POST /ingest/{restaurant_id}` with the full menu from
   the MenuNow backend (never from the browser), using the ingest key.
   - The call is idempotent, so retries are safe.
   - Only changed chunks are re-embedded, so frequent syncs are cheap.
   - Toggle `is_available` for sold-out dishes, and send `allergens: null` when you don't
     know them.
3. **Embed the widget** on the guest menu page with the restaurant's ID and widget key, and
   add MenuNow's origins to `CORS_ORIGINS`.
4. **Show analytics** in the restaurant dashboard by calling `GET /admin/stats` server-side
   with the admin key. "Questions with no result" tells an owner what's missing from their
   menu.
5. **Keep the keys server-side.** The ingest and admin keys never reach a browser. Rotate a
   widget key by sending a new one in the next sync.
6. **Pass the real client IP.** Behind a reverse proxy, set `FORWARDED_ALLOW_IPS` to the
   proxy's address (uvicorn then reads `X-Forwarded-For`). Without it, every guest appears to
   come from the proxy and shares one per-IP limit.

## Roadmap

1. **LiteLLM gateway** in place of direct SDK calls: central retries, fallbacks, per-restaurant
   keys and budgets. See [Migrating to LiteLLM](#migrating-to-litellm).
2. Phase 2:
   - ~~per-session and per-IP rate limits, and enforcement of `monthly_budget_usd`~~ (done)
   - ~~prompt injection detection and output checks~~ (done)
   - ~~streaming answers (SSE) in the API and the widget~~ (done)
   - ~~exact cache, semantic cache and an FAQ route ahead of RAG~~ (done)
   - ~~a real cross-encoder reranker behind the existing interface~~ (done)
3. Eval runs in CI on every PR against a recorded real-model baseline, plus more languages.

## Migrating to LiteLLM

All model access goes through one interface, `app/llm/base.py::LLMClient` (`chat`,
`chat_stream`, `embed`, `model_id`). The only implementation, `app/llm/providers.py`, is the only module allowed to
import the `anthropic` and `openai` SDKs (a test enforces this). App code names aliases
(`menu-fast`, `menu-smart`, `menu-backup`, `menu-embed`), never model IDs.

To move to LiteLLM:
1. Add an `LLMClient` that calls the LiteLLM SDK, or a LiteLLM proxy over its
   OpenAI-compatible API.
2. Return it from `app/llm/__init__.py::make_llm_client`.
3. Map each alias to a LiteLLM model name or proxy route, and let the gateway own retries,
   fallbacks and per-restaurant budgets.

Nothing else changes. The pipeline, eval runner and tests depend only on the interface, and
tests keep using `app/llm/fake.py`.

## Data notes

- All seed data is fictional. `seed/curry_house.json` (40 items, 8 FAQs) is the eval
  restaurant. `seed/pizzeria_finta.json` (5 items) exists only for cross-tenant tests.
- Allergens use the 14 EU allergen groups (Regulation 1169/2011, Annex II). `null` and `[]`
  both mean **unknown**. The service never treats unknown as safe.

## License

MIT
