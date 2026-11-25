"""Ingest seed menus through the normal pipeline: `python -m app.seed [seed/x.json ...]`."""

import asyncio
import json
import sys
from pathlib import Path

from app.config import Settings
from app.db.session import make_engine, make_sessionmaker
from app.ingest.pipeline import ingest_menu
from app.ingest.schemas import MenuPayload
from app.llm import make_llm_client


def load_payload(path: Path) -> MenuPayload:
    return MenuPayload.model_validate(json.loads(path.read_text(encoding="utf-8")))


async def seed(settings: Settings, paths: list[Path]) -> None:
    engine = make_engine(settings.database_url)
    llm = make_llm_client(settings)
    try:
        for path in paths:
            async with make_sessionmaker(engine)() as session, session.begin():
                result = await ingest_menu(session, llm, load_payload(path))
            print(
                f"{path.name}: v{result.menu_version}, {result.chunks_embedded} chunks embedded, "
                f"{result.items_changed} items changed, {result.items_deleted} deleted"
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("seed").glob("*.json"))
    asyncio.run(seed(Settings(), paths))
