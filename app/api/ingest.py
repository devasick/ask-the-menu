import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.ingest.pipeline import IngestResult, ingest_menu
from app.ingest.schemas import MenuPayload
from app.llm.base import LLMClient, LLMError
from app.security.auth import require_ingest_key

router = APIRouter(tags=["ingest"], dependencies=[Depends(require_ingest_key)])


def get_llm(request: Request) -> LLMClient:
    llm: LLMClient = request.app.state.llm
    return llm


@router.post("/ingest/{restaurant_id}")
async def ingest(
    restaurant_id: uuid.UUID,
    payload: MenuPayload,
    session: Annotated[AsyncSession, Depends(get_session)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> IngestResult:
    """Full-menu sync for one restaurant. Idempotent; re-embeds only changed chunks."""
    if payload.restaurant.id != restaurant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "restaurant.id does not match the URL")
    try:
        async with session.begin():
            return await ingest_menu(session, llm, payload)
    except IntegrityError as exc:
        # e.g. the slug already belongs to another restaurant
        raise HTTPException(status.HTTP_409_CONFLICT, "conflicts with existing data") from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "embedding provider unavailable") from exc
