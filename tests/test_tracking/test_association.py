"""HMM track association tests."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import numpy as np
import pytest
import psycopg

from src.common.config import settings
from src.common.schemas import EntityClass, TrackStatus
from src.tracking.association import (
    ActiveTrackRow,
    DetectionAssociationInput,
    TrackAssociator,
    compute_association_score,
    visual_association_probability,
)
from src.tracking.constants import TAU_ASSOCIATE
from src.tracking.isochrone import build_local_friction_window


def test_visual_probability_matrix():
    assert visual_association_probability(EntityClass.CAR, EntityClass.CAR) == 1.0
    assert visual_association_probability(EntityClass.CAR, EntityClass.TRUCK) == 0.15
    assert visual_association_probability(EntityClass.COW, EntityClass.CAR) == 0.0


def test_hmm_scores_plausible_corridor_detection():
    side = 51
    friction = build_local_friction_window(side, EntityClass.PERSON, slope_deg=0.0)
    center_lon, center_lat = -119.82, 36.745
    t0 = __import__("datetime").datetime(2026, 10, 10, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc)

    track = ActiveTrackRow(
        track_id=uuid4(),
        track_code="TRK-LOCAL",
        object_class=EntityClass.PERSON,
        last_detected_at=t0,
        last_lon=center_lon,
        last_lat=center_lat,
        estimated_heading_deg=90.0,
        estimated_velocity_ms=1.0,
        confidence_score=1.0,
    )
    m_per_deg_lon = 111_320.0 * __import__("math").cos(__import__("math").radians(center_lat))
    east_m = 400.0
    det_lon = center_lon + east_m / m_per_deg_lon
    det_lat = center_lat
    detection = DetectionAssociationInput(
        detection_lon=det_lon,
        detection_lat=det_lat,
        detection_time=t0 + timedelta(minutes=8),
        object_class=EntityClass.PERSON,
        confidence=0.9,
    )
    score, gated = compute_association_score(
        detection, track, friction, center_lon, center_lat
    )
    assert gated
    assert score >= TAU_ASSOCIATE


@pytest.mark.asyncio
async def test_associator_links_successive_detections(clean_test_tracks, t0, ranch_corridor_points):
    start, end = ranch_corridor_points
    side = 51
    friction = build_local_friction_window(side, EntityClass.PERSON, slope_deg=0.0)
    center = (
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
    )

    associator = TrackAssociator()
    first = DetectionAssociationInput(
        detection_lon=start[0],
        detection_lat=start[1],
        detection_time=t0,
        object_class=EntityClass.PERSON,
        confidence=0.95,
    )
    state1 = await associator.associate(
        first, friction_override=friction, friction_center=center
    )
    assert state1.track_code.startswith("TRK-")

    second = DetectionAssociationInput(
        detection_lon=end[0],
        detection_lat=end[1],
        detection_time=t0 + timedelta(minutes=12),
        object_class=EntityClass.PERSON,
        confidence=0.92,
    )
    state2 = await associator.associate(
        second, friction_override=friction, friction_center=center
    )
    assert state2.track_id == state1.track_id
    assert state2.status == TrackStatus.ACTIVE

    with psycopg.connect(settings.sync_database_url) as conn:
        count = conn.execute(
            """
            SELECT COUNT(*) FROM entity_tracks
            WHERE object_class = 'person' AND current_status = 'ACTIVE'
              AND id = %s
            """,
            (str(state1.track_id),),
        ).fetchone()[0]
    assert count == 1
