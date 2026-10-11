"""Unit tests for tactical interception solver."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.schemas import EntityClass
from src.dispatch.interception import (
    InterceptionSolver,
    TACTICAL_MARGIN_SECONDS,
    _ChokepointEval,
    _TrackRow,
)
from src.spatial.routing import RouteResult


@pytest.fixture
def track_row() -> _TrackRow:
    return _TrackRow(
        track_id=uuid.uuid4(),
        track_code="TRK-TEST-SOLVER",
        object_class=EntityClass.PERSON,
        last_known_point=(-119.82, 36.74),
        estimated_heading_deg=180.0,
        estimated_velocity_ms=1.5,
        last_detected_at=datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def chokepoints():
    cp_near = uuid.uuid4()
    cp_far = uuid.uuid4()
    return [
        {
            "id": cp_near,
            "name": "Near Gate",
            "location": (-119.819, 36.741),
            "is_accessible_to_response": True,
        },
        {
            "id": cp_far,
            "name": "Far Gate",
            "location": (-119.810, 36.750),
            "is_accessible_to_response": True,
        },
    ]


class TestInterceptionSolverSelection:
    def test_select_best_actionable_prefers_earliest_target_arrival(self):
        solver = InterceptionSolver()
        cp_a = uuid.uuid4()
        cp_b = uuid.uuid4()
        evaluations = [
            _ChokepointEval(cp_a, "A", (-119.82, 36.74), 400.0, 100.0, 300.0, []),
            _ChokepointEval(cp_b, "B", (-119.81, 36.75), 250.0, 50.0, 200.0, []),
            _ChokepointEval(uuid.uuid4(), "C", (-119.80, 36.76), 600.0, 100.0, 100.0, []),
        ]
        best = solver.select_best_actionable(evaluations)
        assert best is not None
        assert best.name == "B"
        assert best.margin_s >= TACTICAL_MARGIN_SECONDS

    def test_select_best_actionable_returns_none_when_margin_insufficient(self):
        solver = InterceptionSolver()
        evaluations = [
            _ChokepointEval(uuid.uuid4(), "X", (-119.82, 36.74), 100.0, 50.0, 50.0, []),
        ]
        assert solver.select_best_actionable(evaluations) is None

    @pytest.mark.asyncio
    async def test_evaluate_and_select_with_margin_at_least_180s(
        self, track_row, chokepoints, monkeypatch
    ):
        solver = InterceptionSolver(session=MagicMock())

        async def fake_target_time(track, location):
            # Far gate takes longer for target; near gate still actionable
            if location[0] == chokepoints[0]["location"][0]:
                return 500.0
            return 900.0

        async def fake_route(start, end, travel_mode="vehicle", session=None, fallback_graph=None):
            return RouteResult(
                distance_m=1000.0,
                transit_time_s=120.0,
                coordinates=[start, end],
                used_fallback=True,
            )

        monkeypatch.setattr(solver, "target_travel_time_seconds", fake_target_time)
        monkeypatch.setattr(
            "src.dispatch.interception.find_shortest_route",
            fake_route,
        )

        evaluations = await solver.evaluate_chokepoints(
            track_row,
            interceptor_position=(-119.82, 36.74),
            chokepoints=chokepoints,
        )
        best = solver.select_best_actionable(evaluations)
        assert best is not None
        assert best.margin_s >= 180.0
        assert best.name == "Near Gate"

    @pytest.mark.asyncio
    async def test_solve_persists_directive(self, track_row, chokepoints, monkeypatch):
        session = AsyncMock()
        commit_called = False

        async def fake_commit():
            nonlocal commit_called
            commit_called = True

        session.commit = fake_commit
        session.execute = AsyncMock(
            return_value=MagicMock(
                first=MagicMock(return_value=(uuid.uuid4(),)),
                mappings=MagicMock(
                    return_value=MagicMock(
                        first=MagicMock(
                            return_value={
                                "id": track_row.track_id,
                                "track_code": track_row.track_code,
                                "object_class": track_row.object_class.value,
                                "lon": track_row.last_known_point[0],
                                "lat": track_row.last_known_point[1],
                                "estimated_heading_deg": track_row.estimated_heading_deg,
                                "estimated_velocity_ms": track_row.estimated_velocity_ms,
                                "last_detected_at": track_row.last_detected_at,
                            }
                        )
                    )
                ),
            )
        )

        solver = InterceptionSolver(session=session)

        target_cps = chokepoints

        async def fake_eval(track, interceptor_position, chokepoints_arg=None):
            return [
                _ChokepointEval(
                    target_cps[0]["id"],
                    target_cps[0]["name"],
                    target_cps[0]["location"],
                    500.0,
                    100.0,
                    400.0,
                    [(-119.82, 36.74), target_cps[0]["location"]],
                )
            ]

        monkeypatch.setattr(solver, "evaluate_chokepoints", fake_eval)

        directive = await solver.solve(
            track_row.track_id,
            interceptor_position=(-119.82, 36.74),
            reference_time=datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc),
        )
        assert directive is not None
        assert directive.safety_margin_seconds >= 180.0
        assert directive.chokepoint_name == "Near Gate"
        assert commit_called
