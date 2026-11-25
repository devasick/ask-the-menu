"""Load seed menus into the database: `python -m app.seed seed/curry_house.json ...`."""

import asyncio
import json
import sys
from pathlib import Path

from app.config import Settings
from app.db.session import make_engine, make_sessionmaker
from app.ingest.pipeline import sync_menu
from app.ingest.schemas import MenuPayload


def load_payload(path: Path) -> MenuPayload:
    return MenuPayload.model_validate(json.loads(path.read_text(encoding="utf-8")))


async def seed(database_url: str, paths: list[Path]) -> None:
    engine = make_engine(database_url)
    try:
        async with make_sessionmaker(engine)() as session, session.begin():
            for path in paths:
                result = await sync_menu(session, load_payload(path))
                print(f"{path.name}: {result}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("seed").glob("*.json"))
    asyncio.run(seed(Settings().database_url, paths))
