"""Herd roll-call report API (spec 19)."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import Response

from src.spatial.rollcall_service import HerdRollCallSummary, get_rollcall_service

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


async def _cop_state_dict() -> dict:
    from src.web.app import build_cop_state

    state = await build_cop_state()
    return state.model_dump(mode="json")


@router.get("/herd-rollcall", response_model=HerdRollCallSummary)
async def get_herd_rollcall() -> HerdRollCallSummary:
    cop_state = await _cop_state_dict()
    return get_rollcall_service().generate_rollcall(cop_state)


@router.get("/herd-rollcall/csv")
async def get_herd_rollcall_csv() -> Response:
    cop_state = await _cop_state_dict()
    summary = get_rollcall_service().generate_rollcall(cop_state)
    csv_text = get_rollcall_service().format_csv(summary)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"herd_rollcall_{date_str}.csv"
    return Response(
        content=csv_text.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
