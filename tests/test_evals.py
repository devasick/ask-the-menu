"""The eval set is valid against the seed menu, the metrics are right, and an offline run
passes the safety gates (allergen safety 100%, no cross-tenant leakage)."""

import json
from decimal import Decimal
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from evals.metrics import (
    CaseResult,
    EvalCase,
    aggregate,
    gates_passed,
    hit_at_k,
    load_dataset,
    percentile,
    reciprocal_rank,
)
from evals.run import report, run_eval

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals/datasets/curry_house.jsonl"


def _seed_ids() -> set[str]:
    data = json.loads((ROOT / "seed/curry_house.json").read_text(encoding="utf-8"))
    return (
        {i["external_id"] for i in data["items"]}
        | {f"faq:{f['question']}" for f in data["faqs"]}
        | {"info"}
    )


def test_dataset_is_big_enough_and_references_real_items() -> None:
    cases = load_dataset(DATASET)
    assert len(cases) >= 80
    assert {c.category for c in cases} == {
        "filter",
        "semantic",
        "allergen",
        "faq",
        "off_topic",
        "injection",
        "cross_tenant",
    }
    known = _seed_ids()
    assert all(set(c.expected_item_ids) <= known for c in cases)
    assert all(c.must_refuse == (not c.expected_item_ids) for c in cases if c.must_refuse)
    german = sum(
        any(w in c.question.lower() for w in ("ist", "habt", "gibt", "ohne", "ich")) for c in cases
    )
    assert german >= 20


def test_ranking_metrics() -> None:
    assert hit_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert hit_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([], 0.5) is None


def _result(**kw: object) -> CaseResult:
    case = EvalCase(
        question="q",
        expected_route="rag",
        expected_item_ids=["a"],
        must_refuse=False,
        category="allergen",
    )
    base: dict[str, object] = dict(
        case=case,
        route="rag",
        ranked_ids=["a"],
        answer_ids=["a"],
        answer="",
        generated=True,
        refused=False,
        allergen_safe=True,
        leaked=False,
        latency_ms=1.0,
        cost_usd=Decimal(0),
    )
    return CaseResult(**{**base, **kw})  # type: ignore[arg-type]


def test_gates_fail_on_unsafe_allergen_answer_or_leak() -> None:
    assert gates_passed(aggregate([_result()], model_metrics=True))
    assert not gates_passed(aggregate([_result(allergen_safe=False)], model_metrics=True))
    assert not gates_passed(aggregate([_result(leaked=True)], model_metrics=True))


def test_model_metrics_are_na_offline() -> None:
    by_name = {m.name: m for m in aggregate([_result()], model_metrics=False)}
    assert by_name["Citation precision"].value is None
    assert by_name["Refusal accuracy"].value is None
    assert by_name["hit@5"].value == 1.0


async def test_offline_eval_passes_safety_gates(
    pg_url: str, sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    settings = Settings(_env_file=None, database_url=pg_url, llm_backend="fake")
    results, llm = await run_eval(settings, DATASET)
    metrics = {m.name: m for m in aggregate(results, model_metrics=False)}
    assert metrics["Allergen safety"].value == 1.0
    assert metrics["Cross-tenant leakage"].value == 0
    assert metrics["Route accuracy"].value == 1.0
    assert "# Eval results" in report(settings, llm, results, DATASET)
