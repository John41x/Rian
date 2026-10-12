"""Camera fleet provisioning and management API (spec 11)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.database import async_session_factory
from src.common.schemas import BoundingBox, DetectionEventProcessed, DetectionItem, EntityClass

router = APIRouter(tags=["cameras"])


class CameraCreateRequest(BaseModel):
    node_name: str
    sender_email: str
    lon: float
    lat: float
    azimuth_deg: float
    fov_angle: float = 85.0
    detection_radius_m: float = 25.0
    battery_level: Optional[int] = 100


class CameraUpdateRequest(BaseModel):
    node_name: Optional[str] = None
    sender_email: Optional[str] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    azimuth_deg: Optional[float] = None
    fov_angle: Optional[float] = None
    is_active: Optional[bool] = None


class CameraDetailResponse(BaseModel):
    id: str
    node_name: str
    sender_email: str
    location: List[float]
    azimuth_deg: float
    fov_angle: float
    status: str
    battery_level: Optional[int]
    fov_wedge: Optional[Dict[str, Any]]


class CameraEventItem(BaseModel):
    id: str
    event_timestamp: datetime
    object_class: str
    confidence: float
    bounding_box: Optional[List[float]] = None
    raw_image_path: Optional[str] = None
    processed_latency_ms: Optional[int] = None
    snapshot_id: Optional[str] = None


def _fov_wedge_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    from src.web.app import _fov_wedge_geojson

    radius = float(row.get("detection_radius_m") or 25.0)
    return _fov_wedge_geojson(
        float(row["lon"]),
        float(row["lat"]),
        float(row["azimuth_deg"]),
        float(row["fov_angle"] or 85.0),
        radius,
    )


def _row_to_camera_response(row: Dict[str, Any]) -> CameraDetailResponse:
    active = bool(row.get("is_active", True))
    return CameraDetailResponse(
        id=str(row["id"]),
        node_name=row["node_name"],
        sender_email=row["sender_email"],
        location=[float(row["lon"]), float(row["lat"])],
        azimuth_deg=float(row["azimuth_deg"]),
        fov_angle=float(row["fov_angle"] or 85.0),
        status="active" if active else "offline",
        battery_level=row.get("battery_level"),
        fov_wedge=_fov_wedge_for_row(row),
    )


_SELECT_CAMERA_SQL = """
    SELECT
        id,
        node_name,
        sender_email,
        ST_X(location) AS lon,
        ST_Y(location) AS lat,
        azimuth_deg,
        fov_angle,
        detection_radius_m,
        is_active,
        battery_level
    FROM sensor_nodes
"""


async def _fetch_camera(session: AsyncSession, camera_id: UUID) -> Optional[Dict[str, Any]]:
    result = await session.execute(
        text(_SELECT_CAMERA_SQL + " WHERE id = :id"),
        {"id": str(camera_id)},
    )
    row = result.mappings().first()
    return dict(row) if row else None


def _bbox_from_row(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, dict):
        return [
            float(raw.get("ymin", 0)),
            float(raw.get("xmin", 0)),
            float(raw.get("ymax", 0)),
            float(raw.get("xmax", 0)),
        ]
    return None


def _event_row_to_item(row: Dict[str, Any]) -> CameraEventItem:
    from src.ingestion.vlm_inspector import get_vlm_inspector

    event_id = row["id"]
    event_uuid = event_id if isinstance(event_id, UUID) else UUID(str(event_id))
    snapshot_id: Optional[str] = None
    if get_vlm_inspector().get_snapshot(event_uuid) is not None:
        snapshot_id = str(event_uuid)

    return CameraEventItem(
        id=str(event_id),
        event_timestamp=row["event_timestamp"],
        object_class=str(row["object_class"]),
        confidence=float(row["confidence"]),
        bounding_box=_bbox_from_row(row.get("bounding_box")),
        raw_image_path=row.get("raw_image_path"),
        processed_latency_ms=row.get("processed_latency_ms"),
        snapshot_id=snapshot_id,
    )


@router.get("/api/v1/cameras/{camera_id}", response_model=CameraDetailResponse)
async def get_camera(camera_id: UUID) -> CameraDetailResponse:
    async with async_session_factory() as session:
        row = await _fetch_camera(session, camera_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return _row_to_camera_response(row)


@router.get("/api/v1/cameras/{camera_id}/events", response_model=List[CameraEventItem])
async def list_camera_events(
    camera_id: UUID,
    limit: int = Query(20, ge=1, le=100),
) -> List[CameraEventItem]:
    async with async_session_factory() as session:
        existing = await _fetch_camera(session, camera_id)
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

        result = await session.execute(
            text(
                """
                SELECT
                    id,
                    event_timestamp,
                    object_class,
                    confidence,
                    bounding_box,
                    raw_image_path,
                    processed_latency_ms
                FROM detection_events
                WHERE sensor_id = :camera_id
                ORDER BY event_timestamp DESC
                LIMIT :limit
                """
            ),
            {"camera_id": str(camera_id), "limit": limit},
        )
        rows = [dict(r) for r in result.mappings().all()]

    return [_event_row_to_item(row) for row in rows]


@router.get("/api/v1/cameras", response_model=List[CameraDetailResponse])
async def list_cameras() -> List[CameraDetailResponse]:
    async with async_session_factory() as session:
        result = await session.execute(
            text(_SELECT_CAMERA_SQL + " ORDER BY node_name")
        )
        rows = result.mappings().all()
    return [_row_to_camera_response(dict(r)) for r in rows]


@router.post(
    "/api/v1/cameras",
    response_model=CameraDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_camera(body: CameraCreateRequest) -> CameraDetailResponse:
    camera_id = uuid4()
    battery = body.battery_level if body.battery_level is not None else 100
    async with async_session_factory() as session:
        try:
            result = await session.execute(
                text(
                    """
                    INSERT INTO sensor_nodes (
                        id, node_name, sender_email, location,
                        azimuth_deg, fov_angle, detection_radius_m,
                        battery_level, is_active
                    )
                    VALUES (
                        :id, :node_name, :sender_email,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                        :azimuth_deg, :fov_angle, :detection_radius_m,
                        :battery_level, TRUE
                    )
                    RETURNING
                        id, node_name, sender_email,
                        ST_X(location) AS lon, ST_Y(location) AS lat,
                        azimuth_deg, fov_angle, detection_radius_m,
                        is_active, battery_level
                    """
                ),
                {
                    "id": str(camera_id),
                    "node_name": body.node_name,
                    "sender_email": body.sender_email,
                    "lon": body.lon,
                    "lat": body.lat,
                    "azimuth_deg": body.azimuth_deg,
                    "fov_angle": body.fov_angle,
                    "detection_radius_m": body.detection_radius_m,
                    "battery_level": battery,
                },
            )
            row = dict(result.mappings().one())
            await session.commit()
        except Exception as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create camera: {exc}",
            ) from exc

    response = _row_to_camera_response(row)
    from src.web.app import _broadcast_sse

    await _broadcast_sse({"type": "camera_provisioned", "camera": response.model_dump()})
    return response


@router.patch("/api/v1/cameras/{camera_id}", response_model=CameraDetailResponse)
async def update_camera(camera_id: UUID, body: CameraUpdateRequest) -> CameraDetailResponse:
    updates: Dict[str, Any] = {}
    set_parts: List[str] = []

    if body.node_name is not None:
        set_parts.append("node_name = :node_name")
        updates["node_name"] = body.node_name
    if body.sender_email is not None:
        set_parts.append("sender_email = :sender_email")
        updates["sender_email"] = body.sender_email
    if body.lon is not None and body.lat is not None:
        set_parts.append("location = ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)")
        updates["lon"] = body.lon
        updates["lat"] = body.lat
    elif body.lon is not None or body.lat is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both lon and lat are required to update position",
        )
    if body.azimuth_deg is not None:
        set_parts.append("azimuth_deg = :azimuth_deg")
        updates["azimuth_deg"] = body.azimuth_deg
    if body.fov_angle is not None:
        set_parts.append("fov_angle = :fov_angle")
        updates["fov_angle"] = body.fov_angle
    if body.is_active is not None:
        set_parts.append("is_active = :is_active")
        updates["is_active"] = body.is_active

    if not set_parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    set_parts.append("updated_at = NOW()")
    updates["id"] = str(camera_id)

    async with async_session_factory() as session:
        existing = await _fetch_camera(session, camera_id)
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

        await session.execute(
            text(
                f"""
                UPDATE sensor_nodes
                SET {", ".join(set_parts)}
                WHERE id = :id
                """
            ),
            updates,
        )
        row = await _fetch_camera(session, camera_id)
        await session.commit()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    response = _row_to_camera_response(row)
    from src.web.app import _broadcast_sse

    await _broadcast_sse({"type": "camera_updated", "camera": response.model_dump()})
    return response


@router.delete("/api/v1/cameras/{camera_id}", status_code=status.HTTP_200_OK)
async def decommission_camera(camera_id: UUID) -> Dict[str, str]:
    async with async_session_factory() as session:
        existing = await _fetch_camera(session, camera_id)
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

        await session.execute(
            text(
                """
                UPDATE sensor_nodes
                SET is_active = FALSE, updated_at = NOW()
                WHERE id = :id
                """
            ),
            {"id": str(camera_id)},
        )
        await session.commit()

    from src.web.app import _broadcast_sse

    await _broadcast_sse({"type": "camera_decommissioned", "camera_id": str(camera_id)})
    return {"status": "decommissioned", "camera_id": str(camera_id)}


@router.post("/api/v1/cameras/{camera_id}/test-probe")
async def camera_test_probe(camera_id: UUID) -> Dict[str, Any]:
    async with async_session_factory() as session:
        row = await _fetch_camera(session, camera_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    if not row.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera is decommissioned",
        )

    now = datetime.now(timezone.utc)
    event = DetectionEventProcessed(
        event_id=uuid4(),
        sensor_id=row["id"] if isinstance(row["id"], UUID) else UUID(str(row["id"])),
        node_name=row["node_name"],
        sensor_location=(float(row["lon"]), float(row["lat"])),
        sensor_azimuth_deg=float(row["azimuth_deg"]),
        event_timestamp=now,
        detections=[
            DetectionItem(
                object_class=EntityClass.PERSON,
                confidence=0.91,
                bounding_box=BoundingBox(ymin=0.15, xmin=0.2, ymax=0.55, xmax=0.45),
            )
        ],
        raw_image_path="test_probe_snapshot.jpg",
        processed_latency_ms=18,
    )

    from src.engine import get_engine
    from src.web.app import _broadcast_sse

    engine = get_engine()
    engine.cop_broadcast = _broadcast_sse
    updated_tracks = await engine.handle_detection_event(event)

    track_payload = [
        {
            "track_id": str(t.track_id),
            "track_code": t.track_code,
            "status": t.status.value,
        }
        for t in updated_tracks
    ]
    return {
        "status": "probe_complete",
        "camera_id": str(camera_id),
        "tracks": track_payload,
        "directive_status": "pending" if not track_payload else "evaluated",
    }
