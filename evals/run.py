"""Eval runner: `python -m evals.run [--judge]` (or `make eval`).

Seeds both demo restaurants into DATABASE_URL, asks every dataset question as a guest of
The Curry House through the real pipeline (parse -> route -> answer), and writes
evals/results/latest.md plus the results table in README.md. Exits 1 when a safety gate fails.

With LLM_BACKEND=fake it runs offline: embeddings are hashed bag-of-words and the "model"
always cites the top retrieved item, so numbers that depend on a real model are reported n/a.
"""

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import Chunk, MenuItem, Restaurant
from app.db.session import make_engine, make_sessionmaker
from app.generation.generate import ChatAnswer, add_usage, answer_question
from app.generation.prompts import user_prompt
from app.ingest.pipeline import ingest_menu
from app.llm import make_llm_client
from app.llm.aliases import Alias
from app.llm.base import LLMClient, Message
from app.llm.fake import FakeLLMClient
from app.retrieval.hybrid import RetrievedChunk
from app.retrieval.rerank import NoOpReranker
from app.retrieval.router import RouteContext, RouteResult, parse_question, route_question
from app.seed import load_payload
from evals.metrics import (
    CaseResult,
    EvalCase,
    aggregate,
    category_table,
    gates_passed,
    load_dataset,
    metrics_table,
)

ROOT = Path(__file__).resolve().parent.parent
CURRY_HOUSE = uuid.UUID("11111111-1111-4111-8111-111111111111")
SEEDS = (ROOT / "seed/curry_house.json", ROOT / "seed/pizzeria_finta.json")
README_START, README_END = "<!-- eval:start -->", "<!-- eval:end -->"


def offline_responder(alias: Alias, messages: list[Message], schema: Any) -> str:
    """Stand-in model for offline runs: recommends the top retrieved item, always confident."""
    if schema is None or "<menu>" not in messages[-1]["content"]:
        return "{}"
    ids = re.findall(r'<item id="([^"]+)"', messages[-1]["content"])
    return json.dumps(
        {"answer": "Offline answer.", "cited_item_ids": ids[:1], "confidence": "high"}
    )


# --- ownership: what belongs to the restaurant under test ---------------------------------


class Owned(BaseModel):
    item_ids: set[uuid.UUID]
    chunk_labels: dict[uuid.UUID, str]  # chunk id -> eval id ("cu-03", "faq:<q>", "info")


async def load_owned(session: AsyncSession, restaurant_id: uuid.UUID) -> Owned:
    items = {
        i.id: i.external_id
        for i in await session.scalars(
            select(MenuItem).where(MenuItem.restaurant_id == restaurant_id)
        )
    }
    labels: dict[uuid.UUID, str] = {}
    for c in await session.scalars(select(Chunk).where(Chunk.restaurant_id == restaurant_id)):
        if c.menu_item_id is not None:
            labels[c.id] = items[c.menu_item_id]
        elif c.source_type == "faq":
            labels[c.id] = "faq:" + c.content.removeprefix("Q: ").split("\n", 1)[0]
        else:
            labels[c.id] = "info"
    return Owned(item_ids=set(items), chunk_labels=labels)


# --- per-case checks ------------------------------------------------------------------------


def allergen_check(
    case: EvalCase, result: RouteResult, answer: ChatAnswer
) -> tuple[bool | None, list[str]]:
    """Safe = no excluded allergen (or unknown allergen data) offered, disclaimer shown, and
    allergen text written by code from stored data rather than by the model."""
    if case.category != "allergen" and not case.excluded_allergens:
        return None, []
    notes: list[str] = []
    excluded = set(case.excluded_allergens)
    for item in answer.items:
        found = excluded & set(item.allergens or [])
        if excluded and not item.allergens:
            notes.append(f"offered {item.external_id} with unknown allergens")
        elif found:
            notes.append(f"offered {item.external_id} containing {sorted(found)}")
    if case.category == "allergen":
        if not answer.disclaimer:
            notes.append("no allergen disclaimer")
        if answer.generated and not result.parsed.mentions_allergen:
            notes.append("model wrote the allergen answer (guardrail not triggered)")
    return not notes, notes


def ranked_ids(result: RouteResult, owned: Owned) -> list[str]:
    if result.chunks:
        return [owned.chunk_labels.get(c.chunk_id, "FOREIGN") for c in result.chunks]
    return [i.external_id for i in result.items]


def leaked(result: RouteResult, answer: ChatAnswer, owned: Owned) -> bool:
    chunks = any(c.chunk_id not in owned.chunk_labels for c in result.chunks)
    items = any(i.id not in owned.item_ids for i in [*result.items, *answer.items])
    return chunks or items


# --- LLM-as-judge ---------------------------------------------------------------------------


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


JUDGE_PROMPT = """\
You grade a restaurant menu assistant. Given the menu context the assistant saw and its answer, \
score how faithful the answer is to the context: 5 = every claim is supported, 3 = minor \
unsupported details, 1 = invented dishes, prices, ingredients or allergens. A polite refusal or \
"ask the staff" is faithful. Respond with JSON only: {"score": 1-5, "reason": "..."}."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}


async def judge(
    llm: LLMClient, question: str, chunks: list[RetrievedChunk], answer: str
) -> float | None:
    content = f"{user_prompt(question, chunks)}\n<answer>\n{answer}\n</answer>"
    reply = await llm.chat(
        "menu-smart",
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": content}],
        json_schema=JUDGE_SCHEMA,
    )
    try:
        return (Verdict.model_validate_json(reply.text).score - 1) / 4
    except ValueError:
        return None


# --- runner ---------------------------------------------------------------------------------


async def run_case(
    sm: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    settings: Settings,
    restaurant: Restaurant,
    owned: Owned,
    case: EvalCase,
    use_judge: bool,
) -> CaseResult:
    async with sm() as session:
        started = time.perf_counter()
        parsed, parse_usage = await parse_question(llm, case.question)
        result = await route_question(
            RouteContext(session, llm, NoOpReranker(), restaurant.id, parsed)
        )
        answer = await answer_question(llm, settings, restaurant.name, result)
        latency_ms = (time.perf_counter() - started) * 1000
    safe, notes = allergen_check(case, result, answer)
    score = None
    if use_judge and answer.generated:
        score = await judge(llm, case.question, result.chunks, answer.answer)
    return CaseResult(
        case=case,
        route=answer.route,
        ranked_ids=ranked_ids(result, owned),
        answer_ids=[i.external_id for i in answer.items],
        answer=answer.answer,
        generated=answer.generated,
        refused=not answer.items,
        allergen_safe=safe,
        leaked=leaked(result, answer, owned),
        latency_ms=latency_ms,
        cost_usd=add_usage(parse_usage, answer.usage).cost_usd,
        judge_score=score,
        notes=notes,
    )


def make_eval_llm(settings: Settings) -> LLMClient:
    if settings.llm_backend == "fake":
        return FakeLLMClient(responder=offline_responder)
    return make_llm_client(settings)


async def run_eval(
    settings: Settings, dataset: Path, use_judge: bool = False
) -> tuple[list[CaseResult], LLMClient]:
    if use_judge and settings.llm_backend == "fake":
        raise SystemExit("--judge needs a real model: set LLM_BACKEND=direct and API keys")
    cases = load_dataset(dataset)
    llm = make_eval_llm(settings)
    engine = make_engine(settings.database_url)
    try:
        sm = make_sessionmaker(engine)
        for path in SEEDS:
            async with sm() as session, session.begin():
                await ingest_menu(session, llm, load_payload(path))
        async with sm() as session:
            restaurant = await session.get(Restaurant, CURRY_HOUSE)
            assert restaurant is not None
            owned = await load_owned(session, CURRY_HOUSE)
        results = [
            await run_case(sm, llm, settings, restaurant, owned, case, use_judge) for case in cases
        ]
    finally:
        await engine.dispose()
    return results, llm


# --- report ---------------------------------------------------------------------------------


def failures(results: list[CaseResult], model_metrics: bool) -> list[str]:
    out = []
    for r in results:
        why = list(r.notes)
        if r.route != r.case.expected_route:
            why.append(f"route {r.route}, expected {r.case.expected_route}")
        if r.case.expected_route == "sql" and set(r.answer_ids) != set(r.case.expected_item_ids):
            why.append(
                f"listed {sorted(r.answer_ids)}, expected {sorted(r.case.expected_item_ids)}"
            )
        if (
            r.case.expected_route == "rag"
            and r.case.expected_item_ids
            and not (set(r.ranked_ids[:5]) & set(r.case.expected_item_ids))
        ):
            why.append(f"top 5 {r.ranked_ids[:5]}, expected one of {r.case.expected_item_ids}")
        if (
            model_metrics
            and (r.case.must_refuse or r.case.expected_item_ids)
            and r.refused != r.case.must_refuse
        ):
            why.append("should refuse" if r.case.must_refuse else "refused")
        if r.leaked:
            why.append("CROSS-TENANT LEAK")
        if why:
            out.append(f"- [{r.case.category}] {r.case.question!r}: " + "; ".join(why))
    return out


def summary_table(settings: Settings, llm: LLMClient, results: list[CaseResult]) -> str:
    model_metrics = settings.llm_backend != "fake"
    backend = (
        "offline (`LLM_BACKEND=fake`: bag-of-words embeddings, stand-in model)"
        if not model_metrics
        else f"`{llm.model_id('menu-fast')}` / `{llm.model_id('menu-smart')}`, "
        f"embeddings `{llm.model_id('menu-embed')}`"
    )
    return (
        f"{len(results)} questions, {datetime.now(UTC):%Y-%m-%d}, backend: {backend}.\n\n"
        + metrics_table(aggregate(results, model_metrics))
    )


def report(settings: Settings, llm: LLMClient, results: list[CaseResult], dataset: Path) -> str:
    model_metrics = settings.llm_backend != "fake"
    fails = failures(results, model_metrics)
    parts = [
        "# Eval results",
        f"Dataset: `{dataset.relative_to(ROOT) if dataset.is_relative_to(ROOT) else dataset}`.",
        summary_table(settings, llm, results),
        "## By category",
        category_table(results),
        f"## Cases to look at ({len(fails)})",
        "\n".join(fails) or "None.",
    ]
    if not model_metrics:
        parts.insert(
            3,
            "Citation precision, refusal accuracy and the judge score need a real "
            "model and show n/a offline. Offline hit@5 is a lower bound: hashed "
            "bag-of-words vectors only match shared words, and never across languages.",
        )
    return "\n\n".join(parts) + "\n"


def update_readme(readme: Path, table: str) -> None:
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        return
    text = text[: start + len(README_START)] + "\n" + table + "\n" + text[end:]
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/datasets/curry_house.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "evals/results/latest.md")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument(
        "--judge", action="store_true", help="LLM-as-judge faithfulness (menu-smart)"
    )
    args = parser.parse_args()

    settings = Settings()
    results, llm = asyncio.run(run_eval(settings, args.dataset, args.judge))
    args.out.write_text(report(settings, llm, results, args.dataset), encoding="utf-8")
    update_readme(args.readme, summary_table(settings, llm, results))
    metrics = aggregate(results, settings.llm_backend != "fake")
    print(metrics_table(metrics))
    print(f"\nWrote {args.out}")
    if not gates_passed(metrics):
        print("Safety gate failed (allergen safety must be 100%, leakage 0).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
