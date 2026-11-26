"""Rerankers: the cross-encoder reorders by pair score; config picks the implementation."""

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.retrieval.rerank import CrossEncoderReranker, NoOpReranker, make_reranker


@dataclass(frozen=True)
class Doc:
    content: str


def overlap(pairs: list[tuple[str, str]]) -> list[float]:
    """Stand-in model: counts shared words."""
    return [float(len(set(q.split()) & set(p.split()))) for q, p in pairs]


async def test_cross_encoder_reorders_by_score_and_keeps_top_k() -> None:
    docs = [Doc("naan bread"), Doc("lamb curry spicy"), Doc("spicy lamb vindaloo curry")]
    ranked = await CrossEncoderReranker(overlap).rerank("spicy lamb curry", docs, top_k=2)
    assert [d.content for d in ranked] == ["lamb curry spicy", "spicy lamb vindaloo curry"]


async def test_ties_keep_the_fused_order_and_small_inputs_skip_the_model() -> None:
    def never(pairs: list[tuple[str, str]]) -> list[float]:
        raise AssertionError("model called")

    docs = [Doc("a"), Doc("b"), Doc("c")]
    assert await CrossEncoderReranker(lambda p: [0.0] * len(p)).rerank("q", docs, 3) == docs
    assert await CrossEncoderReranker(never).rerank("q", docs[:1], 5) == docs[:1]


def test_config_picks_the_reranker() -> None:
    assert isinstance(make_reranker(Settings(_env_file=None)), NoOpReranker)


def test_cross_encoder_without_the_extra_fails_clearly() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="rerank extra"):
            make_reranker(Settings(_env_file=None, reranker="cross-encoder"))
    else:
        pytest.skip("rerank extra installed")
