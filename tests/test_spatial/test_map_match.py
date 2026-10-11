"""Tests for HMM road map-matching and snapping."""

from __future__ import annotations

import math
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.spatial.map_match import CandidateEdge, RoadMapMatcher, _offline_candidate_edges


class TestProbabilityFunctions:
    def test_emission_at_zero_distance(self):
        sigma = 20.0
        expected = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
        assert RoadMapMatcher.calculate_emission_probability(0.0, sigma_z=sigma) == pytest.approx(
            expected
        )

    def test_emission_at_one_sigma(self):
        sigma = 20.0
        at_zero = RoadMapMatcher.calculate_emission_probability(0.0, sigma_z=sigma)
        at_sigma = RoadMapMatcher.calculate_emission_probability(sigma, sigma_z=sigma)
        assert at_sigma == pytest.approx(at_zero * math.exp(-0.5))

    def test_transition_matching_distances(self):
        beta = 30.0
        p = RoadMapMatcher.calculate_transition_probability(100.0, 100.0, beta=beta)
        assert p == pytest.approx(1.0 / beta)

    def test_transition_penalizes_discrepancy(self):
        beta = 30.0
        p_match = RoadMapMatcher.calculate_transition_probability(100.0, 100.0, beta=beta)
        p_gap = RoadMapMatcher.calculate_transition_probability(100.0, 50.0, beta=beta)
        assert p_gap == pytest.approx((1.0 / beta) * math.exp(-50.0 / beta))
        assert p_gap < p_match


class TestOfflineGeometry:
    def test_offline_candidates_near_road(self):
        edges = [
            {
                "id": 10,
                "source": 1,
                "target": 2,
                "lon_lat_path": [(-119.82, 36.74), (-119.819, 36.741)],
            }
        ]
        # Point slightly offset from the segment (~tens of meters)
        cands = _offline_candidate_edges(-119.8195, 36.7405, 150.0, edges)
        assert len(cands) == 1
        assert cands[0].edge_id == 10
        assert cands[0].distance_m < 100.0

    def test_offline_candidates_beyond_radius_excluded(self):
        edges = [
            {
                "id": 1,
                "source": 1,
                "target": 2,
                "lon_lat_path": [(-119.82, 36.74), (-119.819, 36.741)],
            }
        ]
        cands = _offline_candidate_edges(0.0, 0.0, 150.0, edges)
        assert cands == []


def _mock_session_factory(rows: List[Any] | None, exc: Exception | None = None):
    """Build async session factory returning predetermined SQL rows."""

    class _Ctx:
        def __init__(self, session: MagicMock) -> None:
            self._session = session

        async def __aenter__(self) -> MagicMock:
            return self._session

        async def __aexit__(self, *args: Any) -> None:
            return None

    session = MagicMock()
    if exc is not None:
        session.execute = AsyncMock(side_effect=exc)
    else:
        result = MagicMock()
        result.fetchall.return_value = rows or []
        session.execute = AsyncMock(return_value=result)

    factory = MagicMock(side_effect=lambda: _Ctx(session))
    return factory, session


@pytest.mark.asyncio
async def test_find_candidate_edges_uses_db_rows():
    row = MagicMock()
    row.edge_id = 42
    row.source = 5
    row.target = 6
    row.distance_m = 12.5
    row.snapped_lon = -119.819
    row.snapped_lat = 36.741

    factory, session = _mock_session_factory([row])
    matcher = RoadMapMatcher(session_factory=factory)

    cands = await matcher.find_candidate_edges(-119.8191, 36.7409, radius_m=150.0)
    assert len(cands) == 1
    assert cands[0] == CandidateEdge(
        edge_id=42,
        source=5,
        target=6,
        distance_m=12.5,
        snapped_point=(-119.819, 36.741),
    )
    session.execute.assert_awaited()


@pytest.mark.asyncio
async def test_find_candidate_edges_offline_on_db_failure():
    factory, _ = _mock_session_factory(None, exc=RuntimeError("no postgis"))
    matcher = RoadMapMatcher(session_factory=factory)

    cands = await matcher.find_candidate_edges(-119.8195, 36.7405, radius_m=150.0)
    assert len(cands) >= 1
    assert cands[0].distance_m < 150.0


@pytest.mark.asyncio
async def test_snap_point_near_road():
    factory, _ = _mock_session_factory(None, exc=RuntimeError("offline"))
    matcher = RoadMapMatcher(session_factory=factory)

    raw_lon, raw_lat = -119.8195, 36.7405
    slon, slat, edge_id = await matcher.snap_point(raw_lon, raw_lat, max_radius_m=150.0)
    assert edge_id is not None
    assert math.hypot(slon - raw_lon, slat - raw_lat) < 0.01


@pytest.mark.asyncio
async def test_snap_point_far_from_road_returns_raw():
    factory, _ = _mock_session_factory(None, exc=RuntimeError("offline"))
    matcher = RoadMapMatcher(session_factory=factory)

    lon, lat = 0.0, 0.0
    slon, slat, edge_id = await matcher.snap_point(lon, lat, max_radius_m=150.0)
    assert edge_id is None
    assert slon == lon
    assert slat == lat


@pytest.mark.asyncio
async def test_match_trajectory_snaps_sequence():
    factory, _ = _mock_session_factory(None, exc=RuntimeError("offline"))
    matcher = RoadMapMatcher(session_factory=factory)

    points = [
        (-119.82, 36.74),
        (-119.8192, 36.7408),
        (-119.818, 36.742),
    ]
    matched = await matcher.match_trajectory(points)
    assert len(matched) == len(points)
    for orig, snap in zip(points, matched):
        assert math.hypot(orig[0] - snap[0], orig[1] - snap[1]) < 0.05


@pytest.mark.asyncio
async def test_exports_from_spatial_package():
    from src.spatial import CandidateEdge as CE
    from src.spatial import RoadMapMatcher as RMM

    assert CE is CandidateEdge
    assert RMM is RoadMapMatcher
