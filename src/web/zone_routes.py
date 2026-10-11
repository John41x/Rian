"""Spatial zones and domain profile API (spec 13)."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from sqlalchemy import text

from src.common.database import async_session_factory
from src.common.domains import DomainProfile, get_domain_profile
from src.spatial.zones import ZoneManager

router = APIRouter(tags=["zones"])

_zone_manager: ZoneManager | None = None


def get_zone_manager() -> ZoneManager:
    global _zone_manager
    if _zone_manager is None:
        _zone_manager = ZoneManager()
    return _zone_manager


def reset_zone_manager() -> None:
    """Reset cached manager (used in tests when OPERATION_PROFILE changes)."""
    global _zone_manager
    _zone_manager = None


async def _fetch_active_tracks() -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT
            track_code,
            object_class,
            current_status,
            ST_X(last_known_point) AS lon,
            ST_Y(last_known_point) AS lat
        FROM entity_tracks
        WHERE current_status = 'ACTIVE'
        """
    )
    try:
        async with async_session_factory() as session:
            result = await session.execute(sql)
            rows = result.mappings().all()
    except Exception:
        return []

    tracks: List[Dict[str, Any]] = []
    for row in rows:
        tracks.append(
            {
                "track_code": row["track_code"],
                "object_class": row["object_class"],
                "status": row["current_status"],
                "last_known_point": [float(row["lon"]), float(row["lat"])],
            }
        )
    return tracks


@router.get("/api/v1/zones")
async def list_zones_endpoint() -> Dict[str, Any]:
    manager = get_zone_manager()
    await manager.load_from_database()
    tracks = await _fetch_active_tracks()
    features = manager.update_zone_headcounts(tracks)
    return {"type": "FeatureCollection", "features": features}


@router.get("/api/v1/domain-profile", response_model=DomainProfile)
async def domain_profile_endpoint() -> DomainProfile:
    return get_domain_profile()
