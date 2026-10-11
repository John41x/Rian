"""
Scenario 3: Disjoint simultaneous detections 4 km apart — kinematic gating rejects association.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import psycopg

from src.common.config import settings
from src.common.schemas import EntityClass
from tools.kinematics import kinematic_gate_allows
from tools.mock_camera import MockCameraEmitter
from tools.synthetic_gis import SENSOR_NODES

logger = logging.getLogger(__name__)

CAM_NORTH = "cam-gate-north"
CAM_CORRIDOR = "cam-corridor-4k"
GATE_DELTA_SECONDS = 30.0


def _email_for(node_name: str) -> str:
    for cam in SENSOR_NODES:
        if cam.node_name == node_name:
            return cam.sender_email
    raise KeyError(node_name)


def run(emit_smtp: bool = True) -> dict:
    emitter = MockCameraEmitter.from_settings()
    t0 = datetime.now(timezone.utc)

    if emit_smtp:
        try:
            emitter.emit(_email_for(CAM_NORTH), EntityClass.PERSON.value)
        except OSError as exc:
            logger.warning("SMTP emit skipped: %s", exc)

    with psycopg.connect(settings.sync_database_url) as conn:
        north = conn.execute(
            """
            SELECT id, ST_X(location::geometry), ST_Y(location::geometry)
            FROM sensor_nodes WHERE node_name = %s
            """,
            (CAM_NORTH,),
        ).fetchone()
        corridor = conn.execute(
            """
            SELECT id, ST_X(location::geometry), ST_Y(location::geometry)
            FROM sensor_nodes WHERE node_name = %s
            """,
            (CAM_CORRIDOR,),
        ).fetchone()

        if not north or not corridor:
            raise RuntimeError("Simulation DB not seeded; run init-db first")

        north_id, north_lon, north_lat = north
        corridor_id, corridor_lon, corridor_lat = corridor

        allows = kinematic_gate_allows(
            EntityClass.PERSON,
            north_lon,
            north_lat,
            corridor_lon,
            corridor_lat,
            GATE_DELTA_SECONDS,
        )
        if allows:
            raise AssertionError(
                "Kinematic gate should reject 4 km in 30 s for a person"
            )

        conn.execute(
            "DELETE FROM entity_tracks WHERE track_code IN ('TRK-GATE-1', 'TRK-GATE-2')"
        )

        conn.execute(
            """
            INSERT INTO entity_tracks
                (track_code, object_class, current_status, first_detected_at,
                 last_detected_at, last_sensor_id, last_known_point)
            VALUES (%s, %s, 'ACTIVE', %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
            """,
            (
                "TRK-GATE-1",
                EntityClass.PERSON.value,
                t0,
                t0,
                north_id,
                north_lon,
                north_lat,
            ),
        )

        t1 = t0 + timedelta(seconds=GATE_DELTA_SECONDS)
        if emit_smtp:
            try:
                emitter.emit(_email_for(CAM_CORRIDOR), EntityClass.PERSON.value)
            except OSError as exc:
                logger.warning("SMTP emit skipped: %s", exc)

        conn.execute(
            """
            INSERT INTO entity_tracks
                (track_code, object_class, current_status, first_detected_at,
                 last_detected_at, last_sensor_id, last_known_point)
            VALUES (%s, %s, 'ACTIVE', %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
            """,
            (
                "TRK-GATE-2",
                EntityClass.PERSON.value,
                t1,
                t1,
                corridor_id,
                corridor_lon,
                corridor_lat,
            ),
        )
        conn.commit()

        active_count = conn.execute(
            """
            SELECT COUNT(*) FROM entity_tracks
            WHERE current_status = 'ACTIVE'
              AND track_code IN ('TRK-GATE-1', 'TRK-GATE-2')
            """
        ).fetchone()[0]

    if active_count != 2:
        raise AssertionError(f"Expected 2 active tracks, found {active_count}")

    return {
        "scenario": "multi_target_gating",
        "kinematic_gate_rejected": True,
        "active_tracks": active_count,
        "delta_t_seconds": GATE_DELTA_SECONDS,
    }
