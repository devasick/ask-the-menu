from decimal import Decimal

from app.db.models import EMBEDDING_DIM
from app.llm.base import EmbedResult, LLMClient, Usage

BATCH_SIZE = 64


async def embed_texts(llm: LLMClient, texts: list[str]) -> EmbedResult:
    """Embed with `menu-embed` in batches and check the vector size matches the schema."""
    vectors: list[list[float]] = []
    tin, cost, model = 0, Decimal(0), ""
    for start in range(0, len(texts), BATCH_SIZE):
        result = await llm.embed("menu-embed", texts[start : start + BATCH_SIZE])
        bad = {len(v) for v in result.vectors} - {EMBEDDING_DIM}
        if bad or len(result.vectors) != len(texts[start : start + BATCH_SIZE]):
            raise ValueError(
                f"embedding model {result.model} returned dims {sorted(bad)} "
                f"or a wrong count; the schema expects {EMBEDDING_DIM}"
            )
        vectors += result.vectors
        tin += result.usage.tokens_in
        cost += result.usage.cost_usd
        model = result.model
    return EmbedResult(vectors=vectors, model=model, usage=Usage(tin, 0, cost))
