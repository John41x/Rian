"""
Scenario 1: Unauthorized truck enters via North Gate and transits the main corridor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import psycopg

from src.common.config import settings
from src.common.schemas import EntityClass
from tools.kinematics import estimate_tactical_margin_seconds, road_travel_time_seconds
from tools.mock_camera import MockCameraEmitter
from tools.synthetic_gis import COMPACT_DEMO_TRAIL, SENSOR_NODES, local_m_to_lonlat

logger = logging.getLogger(__name__)

NORTH_GATE_XY = (5000.0, 10000.0)
SOUTH_GATE_XY = (5000.0, 0.0)
CORRIDOR_LENGTH_M = 10_000.0


def _north_camera_email() -> str:
    for cam in SENSOR_NODES:
        if cam.node_name == "cam-gate-north":
            return cam.sender_email
    raise RuntimeError("cam-gate-north not configured")


def run(emit_smtp: bool = True) -> dict:
    """
    Simulate truck detection at North Gate and validate intercept margin to South Gate.
    """
    emitter = MockCameraEmitter.from_settings()
    camera_email = _north_camera_email()

    if emit_smtp:
        try:
            emitter.emit(camera_email, EntityClass.TRUCK.value)
        except OSError as exc:
            logger.warning("SMTP emit skipped (no listener): %s", exc)

    truck_speed_kmh = 45.0
    target_eta_s = road_travel_time_seconds(CORRIDOR_LENGTH_M, truck_speed_kmh)
    margin_s = estimate_tactical_margin_seconds(
        target_distance_m=CORRIDOR_LENGTH_M,
        target_speed_kmh=truck_speed_kmh,
        interceptor_distance_m=CORRIDOR_LENGTH_M * 0.35,
        interceptor_speed_kmh=60.0,
    )

    if margin_s < 180.0:
        raise AssertionError(
            f"Expected tactical_margin_seconds >= 180, got {margin_s:.1f}"
        )

    north_lon, north_lat = local_m_to_lonlat(*NORTH_GATE_XY)
    # Compact COP map: place truck on visible pasture corridor (spec 14)
    demo_lon, demo_lat = COMPACT_DEMO_TRAIL[0]
    track_lon, track_lat = demo_lon - 0.004, demo_lat
    now = datetime.now(timezone.utc)

    with psycopg.connect(settings.sync_database_url) as conn:
        sensor = conn.execute(
            "SELECT id FROM sensor_nodes WHERE node_name = %s",
            ("cam-gate-north",),
        ).fetchone()
        south_cp = conn.execute(
            "SELECT id FROM strategic_chokepoints WHERE name = %s",
            ("South Gate",),
        ).fetchone()

        if sensor and south_cp:
            track_id = conn.execute(
                """
                INSERT INTO entity_tracks
                    (track_code, object_class, current_status, first_detected_at,
                     last_detected_at, last_sensor_id, last_known_point,
                     estimated_heading_deg, estimated_velocity_ms, confidence_score)
                VALUES (%s, %s, 'ACTIVE', %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s)
                RETURNING id
                """,
                (
                    "TRK-001",
                    EntityClass.TRUCK.value,
                    now,
                    now,
                    sensor[0],
                    track_lon,
                    track_lat,
                    180.0,
                    truck_speed_kmh / 3.6,
                    0.92,
                ),
            ).fetchone()[0]

            target_eta = now + timedelta(seconds=target_eta_s)
            interceptor_eta = now + timedelta(seconds=target_eta_s - margin_s)
            conn.execute(
                """
                INSERT INTO tactical_directives
                    (track_id, chokepoint_id, target_eta, interceptor_eta, tactical_margin_seconds)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (track_id, south_cp[0], target_eta, interceptor_eta, margin_s),
            )
            conn.commit()

            row = conn.execute(
                """
                SELECT tactical_margin_seconds FROM tactical_directives
                WHERE track_id = %s ORDER BY created_at DESC LIMIT 1
                """,
                (track_id,),
            ).fetchone()
            if row and row[0] < 180.0:
                raise AssertionError(f"DB tactical_margin_seconds {row[0]} < 180")

    return {
        "scenario": "vehicle_intrusion",
        "track_code": "TRK-001",
        "tactical_margin_seconds": margin_s,
        "target_corridor_eta_s": target_eta_s,
    }
