from app.retrieval.hybrid import build_tsquery, rrf_merge


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = rrf_merge(["a", "b", "c"], ["c", "a", "d"], k=60)
    order = [key for key, _ in fused]
    assert order[0] == "a"  # ranked 1st and 2nd
    assert order.index("c") < order.index("b")  # 3rd + 1st beats 2nd alone
    assert set(order) == {"a", "b", "c", "d"}
    assert fused[0][1] == 1 / 61 + 1 / 62


def test_rrf_handles_empty_and_single_lists() -> None:
    assert rrf_merge() == []
    assert [k for k, _ in rrf_merge(["x", "y"], [])] == ["x", "y"]


def test_tsquery_is_sanitized_or_query() -> None:
    assert build_tsquery(["korma", "nuts"]) == "korma:* | nuts:*"
    assert build_tsquery(["a&b|c!", "x"]) is None  # single letters dropped
    assert build_tsquery(["chili's", "Lassi"]) == "chili:* | lassi:*"
    assert build_tsquery([]) is None
