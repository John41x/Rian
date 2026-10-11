"""Shared fixtures for dispatch module tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.common.schemas import EntityClass, TacticalDirective


@pytest.fixture
def reference_time() -> datetime:
    return datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_directive(reference_time: datetime) -> TacticalDirective:
    track_id = uuid.uuid4()
    cp_id = uuid.uuid4()
    target_eta = reference_time + timedelta(minutes=15)
    resp_eta = reference_time + timedelta(minutes=10)
    return TacticalDirective(
        directive_id=uuid.uuid4(),
        track_id=track_id,
        track_code="TRK-TEST-001",
        target_class=EntityClass.PERSON,
        target_chokepoint_id=cp_id,
        chokepoint_name="South Gate",
        chokepoint_location=(-119.818, 36.742),
        target_arrival_time=target_eta,
        interceptor_arrival_time=resp_eta,
        safety_margin_seconds=300.0,
        action_message="INTERCEPT AT South Gate",
    )
