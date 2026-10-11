"""
Scenario: Stray cow drifts south across compact demo pastures (spec 14).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import psycopg

from src.common.config import settings
from src.common.schemas import EntityClass

logger = logging.getLogger(__name__)

# North Pasture center — drifts south through Creek Draw toward South Meadow
START_LON = -119.8200
START_LAT = 36.7540
HEADING_DEG = 180.0
SPEED_MS = 1.4


def run(emit_smtp: bool = True) -> dict:
    """Seed an active cow track on the compact estate for kinematic COP demo."""
    now = datetime.now(timezone.utc)

    with psycopg.connect(settings.sync_database_url) as conn:
        sensor = conn.execute(
            "SELECT id FROM sensor_nodes WHERE node_name = %s",
            ("cam-gate-north",),
        ).fetchone()
        sensor_id = sensor[0] if sensor else None

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
                "HERD-COW-04",
                EntityClass.COW.value,
                now,
                now,
                sensor_id,
                START_LON,
                START_LAT,
                HEADING_DEG,
                SPEED_MS,
                0.88,
            ),
        ).fetchone()[0]
        conn.commit()

    if emit_smtp:
        logger.info("Stray cow scenario track %s at (%.5f, %.5f)", track_id, START_LON, START_LAT)

    return {
        "scenario": "stray_cow",
        "track_id": str(track_id),
        "track_code": "HERD-COW-04",
        "start_lonlat": (START_LON, START_LAT),
        "heading_deg": HEADING_DEG,
        "speed_ms": SPEED_MS,
    }
