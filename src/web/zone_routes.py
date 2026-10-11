"""Spatial zones and domain profile API (spec 13)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from src.common.database import async_session_factory
from src.common.domains import DomainProfile, get_domain_profile
from src.spatial.zones import (
    GeofenceBreachEvent,
    ZoneCreateRequest,
    ZoneManager,
    ZoneUpdateRequest,
)

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
            id,
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
                "track_id": str(row["id"]),
                "track_code": row["track_code"],
                "object_class": row["object_class"],
                "entity_class": row["object_class"],
                "status": row["current_status"],
                "last_known_point": [float(row["lon"]), float(row["lat"])],
            }
        )
    return tracks


async def _persist_zone_feature(feature: Dict[str, Any]) -> None:
    """Best-effort PostGIS insert/update for a drawn zone."""
    props = feature.get("properties") or {}
    geom_json = json.dumps(feature.get("geometry"))
    extra = {k: v for k, v in props.items() if k not in ("name", "zone_code", "zone_type")}
    try:
        async with async_session_factory() as session:
            await get_zone_manager().ensure_table(session)
            await session.execute(
                text(
                    """
                    INSERT INTO spatial_zones (name, zone_code, zone_type, boundary, properties)
                    VALUES (
                        :name,
                        :zone_code,
                        :zone_type,
                        ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326),
                        CAST(:props AS jsonb)
                    )
                    ON CONFLICT (zone_code) DO UPDATE SET
                        name = EXCLUDED.name,
                        zone_type = EXCLUDED.zone_type,
                        boundary = EXCLUDED.boundary,
                        properties = EXCLUDED.properties,
                        updated_at = NOW()
                    """
                ),
                {
                    "name": props.get("name"),
                    "zone_code": props.get("zone_code"),
                    "zone_type": props.get("zone_type", "pasture"),
                    "geom": geom_json,
                    "props": json.dumps(extra),
                },
            )
            await session.commit()
    except Exception:
        pass


@router.get("/api/v1/zones")
async def list_zones_endpoint() -> Dict[str, Any]:
    manager = get_zone_manager()
    await manager.load_from_database()
    tracks = await _fetch_active_tracks()
    features = manager.update_zone_headcounts(tracks)
    return {"type": "FeatureCollection", "features": features}


@router.get("/api/v1/zones/breaches", response_model=List[GeofenceBreachEvent])
async def list_geofence_breaches() -> List[GeofenceBreachEvent]:
    manager = get_zone_manager()
    await manager.load_from_database()
    tracks = await _fetch_active_tracks()
    return manager.detect_geofence_breaches(tracks)


@router.post("/api/v1/zones", status_code=status.HTTP_201_CREATED)
async def create_zone_endpoint(request: ZoneCreateRequest) -> Dict[str, Any]:
    manager = get_zone_manager()
    try:
        feature = manager.create_zone(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    await _persist_zone_feature(feature)
    from src.web.app import _broadcast_sse

    await _broadcast_sse({"type": "zone_created", "zone": feature})
    return feature


@router.put("/api/v1/zones/{zone_id}")
async def update_zone_endpoint(
    zone_id: str, update: ZoneUpdateRequest
) -> Dict[str, Any]:
    manager = get_zone_manager()
    updated = manager.update_zone(zone_id, update)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zone not found")
    await _persist_zone_feature(updated)
    from src.web.app import _broadcast_sse

    await _broadcast_sse({"type": "zone_updated", "zone": updated})
    return updated


@router.delete("/api/v1/zones/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_zone_endpoint(zone_id: str) -> None:
    manager = get_zone_manager()
    if not manager.delete_zone(zone_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zone not found")
    from src.web.app import _broadcast_sse

    await _broadcast_sse({"type": "zone_deleted", "zone_id": zone_id})


@router.get("/api/v1/domain-profile", response_model=DomainProfile)
async def domain_profile_endpoint() -> DomainProfile:
    return get_domain_profile()
