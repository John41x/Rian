"""Track state worker tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import psycopg

from src.common.config import settings
from src.common.schemas import EntityClass, TrackStatus
from src.tracking.state_worker import TrackStateWorker


@pytest.mark.asyncio
async def test_worker_marks_stale_track(clean_test_tracks, t0):
    track_id = uuid4()
    code = "TRK-TEST-STALE"
    stale_time = t0 - timedelta(minutes=45)
    with psycopg.connect(settings.sync_database_url) as conn:
        conn.execute(
            """
            INSERT INTO entity_tracks (
                id, track_code, object_class, current_status,
                first_detected_at, last_detected_at, last_known_point
            ) VALUES (
                %s, %s, 'person', 'ACTIVE', %s, %s,
                ST_SetSRID(ST_MakePoint(-119.82, 36.745), 4326)
            )
            """,
            (str(track_id), code, stale_time, stale_time),
        )
        conn.commit()

    worker = TrackStateWorker()
    now = t0
    states = await worker.tick(now=now)
    stale = [s for s in states if s.track_id == track_id]
    assert stale
    assert stale[0].status == TrackStatus.STALE

    with psycopg.connect(settings.sync_database_url) as conn:
        status = conn.execute(
            "SELECT current_status FROM entity_tracks WHERE id = %s",
            (str(track_id),),
        ).fetchone()[0]
    assert status == "STALE"


@pytest.mark.asyncio
async def test_worker_publishes_callback(clean_test_tracks, t0):
    track_id = uuid4()
    code = "TRK-TEST-CB"
    with psycopg.connect(settings.sync_database_url) as conn:
        conn.execute(
            """
            INSERT INTO entity_tracks (
                id, track_code, object_class, current_status,
                first_detected_at, last_detected_at, last_known_point
            ) VALUES (
                %s, %s, 'person', 'ACTIVE', %s, %s,
                ST_SetSRID(ST_MakePoint(-119.82, 36.745), 4326)
            )
            """,
            (str(track_id), code, t0, t0),
        )
        conn.commit()

    received = []

    async def on_update(state):
        received.append(state)

    worker = TrackStateWorker(on_track_update=on_update)
    await worker.tick(now=t0 + timedelta(seconds=30))
    assert any(s.track_id == track_id for s in received)
