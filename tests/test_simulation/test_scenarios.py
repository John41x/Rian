"""Scenario runner smoke tests (DB-backed)."""

from __future__ import annotations

import pytest

from tools.synthetic_gis import seed_all


@pytest.fixture(scope="module")
def seeded_db():
    summary = seed_all()
    return summary


def test_vehicle_intrusion(seeded_db):
    from tools.scenarios.vehicle_intrusion import run

    result = run(emit_smtp=False)
    assert result["tactical_margin_seconds"] >= 180.0


def test_pedestrian_hike(seeded_db):
    from tools.scenarios.pedestrian_hike import run

    result = run(emit_smtp=False)
    assert result["anisotropy_error_pct"] <= 10.0


def test_multi_target_gating(seeded_db):
    from tools.scenarios.multi_target_gating import run

    result = run(emit_smtp=False)
    assert result["active_tracks"] == 2
    assert result["kinematic_gate_rejected"]
