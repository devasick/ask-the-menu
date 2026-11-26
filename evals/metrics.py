"""Eval dataset schema, per-case outcomes and the aggregate metrics. Pure functions, no I/O."""

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Category = Literal[
    "filter", "semantic", "allergen", "faq", "off_topic", "injection", "cross_tenant"
]
K = 5


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    expected_route: Literal["sql", "rag", "faq", "exact_cache", "semantic_cache"]
    # Menu item external IDs; "faq:<question>" and "info" name the other chunk types.
    expected_item_ids: list[str]
    must_refuse: bool
    category: Category
    # Allergens the guest must not be offered (drives the allergen safety check).
    excluded_allergens: list[str] = []


def load_dataset(path: Path) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate(json.loads(line)) for line in lines if line.strip()]


@dataclass
class CaseResult:
    case: EvalCase
    route: str
    ranked_ids: list[str]  # retrieval order: chunks for RAG, rows for SQL
    answer_ids: list[str]  # items shown to the guest
    answer: str
    generated: bool
    refused: bool
    allergen_safe: bool | None  # None when the case has no allergen aspect
    leaked: bool  # anything from another restaurant reached retrieval or the answer
    latency_ms: float
    cost_usd: Decimal
    judge_score: float | None = None  # 0..1 faithfulness, only with --judge
    notes: list[str] = field(default_factory=list)


def hit_at_k(ranked: list[str], expected: list[str], k: int = K) -> float:
    return 1.0 if set(ranked[:k]) & set(expected) else 0.0


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    for rank, rid in enumerate(ranked, start=1):
        if rid in expected:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (same as Postgres percentile_cont)."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | None
    n: int
    fmt: Literal["pct", "num", "ms", "usd", "count"] = "pct"
    gate: str = ""  # human-readable pass condition, empty if none
    passed: bool | None = None

    def shown(self) -> str:
        if self.value is None:
            return "n/a"
        match self.fmt:
            case "pct":
                return f"{self.value:.1%}"
            case "ms":
                return f"{self.value:.0f} ms"
            case "usd":
                return f"${self.value:.6f}"
            case "count":
                return f"{self.value:.0f}"
            case _:
                return f"{self.value:.3f}"


def aggregate(results: list[CaseResult], model_metrics: bool) -> list[Metric]:
    """model_metrics=False (offline fake model): skip numbers that only a real model can earn."""
    rag = [r for r in results if r.case.expected_route == "rag" and r.case.expected_item_ids]
    sql = [r for r in results if r.case.expected_route == "sql"]
    cited = [r for r in results if r.generated and r.answer_ids]
    definite = [r for r in results if r.case.must_refuse or r.case.expected_item_ids]
    safety = [r.allergen_safe for r in results if r.allergen_safe is not None]
    leaks = sum(r.leaked for r in results)
    latencies = [r.latency_ms for r in results]
    costs = [float(r.cost_usd) for r in results]
    judged = [r.judge_score for r in results if r.judge_score is not None]

    def precision(r: CaseResult) -> float:
        expected = set(r.case.expected_item_ids)
        return sum(i in expected for i in r.answer_ids) / len(r.answer_ids)

    def only_model(value: float | None) -> float | None:
        return value if model_metrics else None

    safety_value = _mean([float(s) for s in safety])
    return [
        Metric(
            "hit@5",
            _mean([hit_at_k(r.ranked_ids, r.case.expected_item_ids) for r in rag]),
            len(rag),
        ),
        Metric(
            "MRR",
            _mean([reciprocal_rank(r.ranked_ids, r.case.expected_item_ids) for r in rag]),
            len(rag),
            "num",
        ),
        Metric(
            "SQL filter exact match",
            _mean([float(set(r.answer_ids) == set(r.case.expected_item_ids)) for r in sql]),
            len(sql),
        ),
        Metric("Citation precision", only_model(_mean([precision(r) for r in cited])), len(cited)),
        Metric(
            "Route accuracy",
            _mean([float(r.route == r.case.expected_route) for r in results]),
            len(results),
        ),
        Metric(
            "Refusal accuracy",
            only_model(_mean([float(r.refused == r.case.must_refuse) for r in definite])),
            len(definite),
        ),
        Metric(
            "Allergen safety",
            safety_value,
            len(safety),
            gate="= 100%",
            passed=safety_value is None or safety_value == 1.0,
        ),
        Metric(
            "Cross-tenant leakage",
            float(leaks),
            len(results),
            "count",
            gate="= 0",
            passed=leaks == 0,
        ),
        Metric(
            "Answered without generation",
            _mean([float(not r.generated) for r in results]),
            len(results),
        ),
        Metric("Latency p50", percentile(latencies, 0.5), len(results), "ms"),
        Metric("Latency p95", percentile(latencies, 0.95), len(results), "ms"),
        Metric("Avg cost per question", _mean(costs), len(results), "usd"),
        Metric("Judge faithfulness", _mean(judged), len(judged)),
    ]


def gates_passed(metrics: list[Metric]) -> bool:
    return all(m.passed is not False for m in metrics)


def metrics_table(metrics: list[Metric]) -> str:
    rows = ["| Metric | Value | n | Gate |", "|---|---|---|---|"]
    for m in metrics:
        gate = "" if not m.gate else f"{m.gate} {'✅' if m.passed else '❌'}"
        rows.append(f"| {m.name} | {m.shown()} | {m.n} | {gate} |")
    return "\n".join(rows)


def category_table(results: list[CaseResult]) -> str:
    rows = ["| Category | Cases | Route correct | hit@5 |", "|---|---|---|---|"]
    for cat in sorted({r.case.category for r in results}):
        group = [r for r in results if r.case.category == cat]
        routed = sum(r.route == r.case.expected_route for r in group)
        ranked = [r for r in group if r.case.expected_item_ids and r.case.expected_route == "rag"]
        hits = _mean([hit_at_k(r.ranked_ids, r.case.expected_item_ids) for r in ranked])
        shown = "n/a" if hits is None else f"{hits:.0%}"
        rows.append(f"| {cat} | {len(group)} | {routed}/{len(group)} | {shown} |")
    return "\n".join(rows)
