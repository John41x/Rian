"""Spatiotemporal gating envelope tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.common.schemas import EntityClass
from src.tracking.gating import evaluate_spatiotemporal_gate


def test_gating_rejects_high_velocity_person(t0):
    """4 km in 30 s exceeds person v_max (spec scenario 3)."""
    north = (-119.8300, 36.7550)
    corridor = (-119.7900, 36.7550)
    result = evaluate_spatiotemporal_gate(
        north[0],
        north[1],
        t0,
        corridor[0],
        corridor[1],
        t0 + timedelta(seconds=30),
        EntityClass.PERSON,
    )
    assert not result.reachable
    assert result.implied_speed_ms > result.max_speed_ms
    assert result.distance_m > 3000


def test_gating_allows_slow_pedestrian_move(t0, ranch_corridor_points):
    start, end = ranch_corridor_points
    result = evaluate_spatiotemporal_gate(
        start[0],
        start[1],
        t0,
        end[0],
        end[1],
        t0 + timedelta(minutes=10),
        EntityClass.PERSON,
    )
    assert result.reachable
    assert result.implied_speed_ms < result.max_speed_ms


def test_gating_uses_detection_class_velocity(t0):
    """Vehicle class permits faster implied speed than person for same motion."""
    a = (-119.8200, 36.7450)
    b = (-119.8150, 36.7450)
    dt = t0 + timedelta(seconds=20)
    person = evaluate_spatiotemporal_gate(a[0], a[1], t0, b[0], b[1], dt, EntityClass.PERSON)
    car = evaluate_spatiotemporal_gate(a[0], a[1], t0, b[0], b[1], dt, EntityClass.CAR)
    assert person.distance_m == car.distance_m
    assert car.reachable or (car.max_speed_ms > person.max_speed_ms)
