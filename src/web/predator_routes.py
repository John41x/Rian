"""Predator threat detection REST API (spec 17)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from geopy.distance import geodesic
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.common.database import async_session_factory
from src.common.schemas import EntityClass
from src.tracking.predator_service import (
    PredatorThreatEvent,
    PredatorProximitySummary,
    get_predator_service,
)

router = APIRouter(prefix="/api/v1/predators", tags=["predators"])

HERD_COW_04_LON = -119.8200
HERD_COW_04_LAT = 36.7540
SIM_COYOTE_SEPARATION_M = 350.0


class TriggerDeterrentBody(BaseModel):
    deterrent_type: str = Field(default="acoustic_siren")


async def _fetch_active_track_dicts() -> List[Dict[str, Any]]:
    from src.web.app import build_cop_state

    state = await build_cop_state()
    return list(state.tracks)


@router.get("/threats", response_model=PredatorProximitySummary)
async def list_predator_threats() -> PredatorProximitySummary:
    tracks = await _fetch_active_track_dicts()
    service = get_predator_service()
    return service.evaluate_threats(tracks)


@router.post("/{threat_id}/trigger-deterrent", response_model=PredatorThreatEvent)
async def trigger_predator_deterrent(
    threat_id: UUID,
    body: TriggerDeterrentBody | None = None,
) -> PredatorThreatEvent:
    deterrent_type = (body.deterrent_type if body else "acoustic_siren")
    service = get_predator_service()
    tracks = await _fetch_active_track_dicts()
    service.evaluate_threats(tracks)
    updated = service.trigger_deterrent(threat_id, deterrent_type=deterrent_type)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Threat not found",
        )
    from src.web.app import _broadcast_sse

    await _broadcast_sse(
        {
            "type": "predator_deterrent",
            "threat_id": str(threat_id),
            "deterrent_type": deterrent_type,
            "event": updated.model_dump(mode="json"),
        }
    )
    return updated


@router.post("/simulate-incursion", response_model=PredatorThreatEvent)
async def simulate_predator_incursion() -> PredatorThreatEvent:
    now = datetime.now(timezone.utc)
    coyote_lon = HERD_COW_04_LON
    coyote_lat = HERD_COW_04_LAT
    dest = geodesic(meters=SIM_COYOTE_SEPARATION_M).destination(
        (HERD_COW_04_LAT, HERD_COW_04_LON),
        bearing=90,
    )
    coyote_lon = float(dest.longitude)
    coyote_lat = float(dest.latitude)

    async with async_session_factory() as session:
        herd_row = await session.execute(
            text(
                """
                SELECT id FROM entity_tracks
                WHERE track_code = 'HERD-COW-04' AND current_status = 'ACTIVE'
                LIMIT 1
                """
            )
        )
        herd = herd_row.mappings().first()
        if herd is None:
            sensor = await session.execute(
                text("SELECT id FROM sensor_nodes WHERE node_name = 'cam-gate-north' LIMIT 1")
            )
            sensor_id = sensor.scalar()
            await session.execute(
                text(
                    """
                    INSERT INTO entity_tracks
                        (track_code, object_class, current_status, first_detected_at,
                         last_detected_at, last_sensor_id, last_known_point,
                         estimated_heading_deg, estimated_velocity_ms, confidence_score)
                    VALUES
                        ('HERD-COW-04', :cls, 'ACTIVE', :ts, :ts, :sensor_id,
                         ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                         180, 1.4, 0.88)
                    """
                ),
                {
                    "cls": EntityClass.COW.value,
                    "ts": now,
                    "sensor_id": sensor_id,
                    "lon": HERD_COW_04_LON,
                    "lat": HERD_COW_04_LAT,
                },
            )

        await session.execute(
            text("DELETE FROM entity_tracks WHERE track_code = 'TRK-COYOTE-01'")
        )
        sensor = await session.execute(
            text("SELECT id FROM sensor_nodes WHERE node_name = 'cam-gate-north' LIMIT 1")
        )
        sensor_id = sensor.scalar()
        await session.execute(
            text(
                """
                INSERT INTO entity_tracks
                    (track_code, object_class, current_status, first_detected_at,
                     last_detected_at, last_sensor_id, last_known_point,
                     estimated_heading_deg, estimated_velocity_ms, confidence_score)
                VALUES
                    ('TRK-COYOTE-01', :cls, 'ACTIVE', :ts, :ts, :sensor_id,
                     ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                     270, 2.2, 0.91)
                """
            ),
            {
                "cls": "coyote",
                "ts": now,
                "sensor_id": sensor_id,
                "lon": coyote_lon,
                "lat": coyote_lat,
            },
        )
        await session.commit()

    tracks = await _fetch_active_track_dicts()
    service = get_predator_service()
    summary = service.evaluate_threats(tracks)
    coyote_threats = [
        t for t in summary.threats if t.predator_track_code == "TRK-COYOTE-01"
    ]
    if not coyote_threats:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulated incursion did not produce an active threat",
        )
    from src.web.app import _broadcast_sse, build_cop_state

    state = await build_cop_state()
    await _broadcast_sse(
        {
            "type": "predator_incursion_simulated",
            "threat": coyote_threats[0].model_dump(mode="json"),
            "state": state.model_dump(mode="json"),
        }
    )
    return coyote_threats[0]
