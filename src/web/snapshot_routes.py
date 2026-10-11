"""VLM forensic snapshot gallery API (spec 16)."""

from __future__ import annotations

from typing import List
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

from src.ingestion.vlm_inspector import (
    BoundingBoxOverlay,
    SnapshotRecord,
    VlmInspectionReport,
    get_vlm_inspector,
)

router = APIRouter(tags=["snapshots"])


class GenerateTrackSnapshotRequest(BaseModel):
    entity_class: str = "cow"
    camera_name: str = "Simulation Camera"
    confidence: float = 0.88


@router.get("/api/v1/cameras/{camera_id}/snapshots", response_model=List[SnapshotRecord])
async def list_camera_snapshots(camera_id: UUID, limit: int = 10) -> List[SnapshotRecord]:
    inspector = get_vlm_inspector()
    return inspector.get_camera_snapshots(camera_id, limit=limit)


@router.get("/api/v1/snapshots/{snapshot_id}/image")
async def get_snapshot_image(snapshot_id: UUID) -> Response:
    inspector = get_vlm_inspector()
    if inspector.get_snapshot(snapshot_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found")
    data = inspector.get_snapshot_image(snapshot_id)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not available")
    return Response(content=data, media_type="image/jpeg")


@router.post(
    "/api/v1/snapshots/{snapshot_id}/vlm-analyze",
    response_model=VlmInspectionReport,
)
async def vlm_analyze_snapshot(snapshot_id: UUID) -> VlmInspectionReport:
    inspector = get_vlm_inspector()
    try:
        return await inspector.analyze_snapshot(snapshot_id)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        ) from None


@router.get("/api/v1/cop/entity/{track_id}/snapshot", response_model=SnapshotRecord)
async def get_entity_track_snapshot(track_id: UUID) -> SnapshotRecord:
    inspector = get_vlm_inspector()
    record = inspector.get_track_snapshot(track_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No snapshot for track",
        )
    return record


@router.post(
    "/api/v1/cop/entity/{track_id}/snapshot",
    response_model=SnapshotRecord,
    status_code=status.HTTP_201_CREATED,
)
async def generate_entity_track_snapshot(
    track_id: UUID,
    body: GenerateTrackSnapshotRequest | None = None,
) -> SnapshotRecord:
    """Create a synthetic test snapshot linked to an entity track (COP UI)."""
    req = body or GenerateTrackSnapshotRequest()
    inspector = get_vlm_inspector()
    sensor_id = uuid4()
    box = BoundingBoxOverlay(
        label=req.entity_class,
        confidence=req.confidence,
        ymin=0.25,
        xmin=0.3,
        ymax=0.75,
        xmax=0.65,
    )
    return inspector.record_snapshot(
        sensor_id=sensor_id,
        camera_name=req.camera_name,
        entity_class=req.entity_class,
        confidence=req.confidence,
        bounding_boxes=[box],
        track_id=track_id,
    )
