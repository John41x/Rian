"""Predator threat detection service and API tests (spec 17)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.tracking.predator_service import (
    PredatorThreatLevel,
    PredatorThreatService,
    get_predator_service,
    reset_predator_service,
)
from src.web.app import app


@pytest.fixture(autouse=True)
def fresh_predator_service():
    reset_predator_service()
    yield
    reset_predator_service()


def _track(
    track_id: uuid.UUID,
    code: str,
    object_class: str,
    lon: float,
    lat: float,
) -> dict:
    return {
        "track_id": str(track_id),
        "track_code": code,
        "object_class": object_class,
        "status": "ACTIVE",
        "last_known_point": [lon, lat],
    }


def test_threat_level_critical_under_200m():
    service = PredatorThreatService()
    cow_id = uuid.uuid4()
    coyote_id = uuid.uuid4()
    # ~150 m east at demo latitude
    tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8183, 36.7540),
    ]
    summary = service.evaluate_threats(tracks)
    assert summary.active_threats_count == 1
    assert summary.threats[0].threat_level == PredatorThreatLevel.CRITICAL_INCURSION
    assert summary.threats[0].distance_meters < 200


def test_threat_level_warning_between_200_and_500m():
    service = PredatorThreatService()
    cow_id = uuid.uuid4()
    coyote_id = uuid.uuid4()
    tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8160, 36.7540),
    ]
    summary = service.evaluate_threats(tracks)
    threat = summary.threats[0]
    assert threat.threat_level == PredatorThreatLevel.HERD_THREAT_WARNING
    assert 200 <= threat.distance_meters < 500


def test_threat_level_advisory_between_500_and_1000m():
    service = PredatorThreatService()
    cow_id = uuid.uuid4()
    coyote_id = uuid.uuid4()
    tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8100, 36.7540),
    ]
    summary = service.evaluate_threats(tracks)
    threat = summary.threats[0]
    assert threat.threat_level == PredatorThreatLevel.PROXIMITY_ADVISORY
    assert 500 <= threat.distance_meters < 1000


def test_nominal_beyond_1000m_excluded_from_active_threats():
    service = PredatorThreatService()
    cow_id = uuid.uuid4()
    coyote_id = uuid.uuid4()
    tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8000, 36.7540),
    ]
    summary = service.evaluate_threats(tracks)
    assert summary.active_threats_count == 0
    assert summary.highest_threat_level == PredatorThreatLevel.NOMINAL
    assert summary.threats == []


def test_only_livestock_yields_zero_threats():
    service = PredatorThreatService()
    tracks = [_track(uuid.uuid4(), "HERD-COW-04", "cow", -119.82, 36.754)]
    summary = service.evaluate_threats(tracks)
    assert summary.active_threats_count == 0
    assert summary.threats == []


def test_only_predators_yields_zero_threats():
    service = PredatorThreatService()
    tracks = [_track(uuid.uuid4(), "TRK-COYOTE-01", "coyote", -119.82, 36.754)]
    summary = service.evaluate_threats(tracks)
    assert summary.active_threats_count == 0
    assert summary.threats == []


def test_trigger_deterrent_records_activation():
    service = PredatorThreatService()
    cow_id = uuid.uuid4()
    coyote_id = uuid.uuid4()
    tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8183, 36.7540),
    ]
    summary = service.evaluate_threats(tracks)
    threat_id = summary.threats[0].threat_id
    before = datetime.now(timezone.utc)
    updated = service.trigger_deterrent(threat_id, deterrent_type="acoustic_siren")
    assert updated is not None
    assert updated.deterrent_activated is True
    assert updated.deterrent_type == "acoustic_siren"
    assert updated.timestamp >= before
    refreshed = service.evaluate_threats(tracks)
    assert refreshed.threats[0].deterrent_activated is True


def test_get_predator_service_singleton():
    a = get_predator_service()
    b = get_predator_service()
    assert a is b


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_rest_threats_endpoint(client):
    coyote_id = uuid.uuid4()
    cow_id = uuid.uuid4()
    fake_tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8183, 36.7540),
    ]
    with patch(
        "src.web.predator_routes._fetch_active_track_dicts",
        new=AsyncMock(return_value=fake_tracks),
    ):
        resp = await client.get("/api/v1/predators/threats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_threats_count"] == 1
    assert body["threats"][0]["predator_track_code"] == "TRK-COYOTE-01"


@pytest.mark.asyncio
async def test_rest_trigger_deterrent_endpoint(client):
    service = get_predator_service()
    coyote_id = uuid.uuid4()
    cow_id = uuid.uuid4()
    fake_tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8183, 36.7540),
    ]
    summary = service.evaluate_threats(fake_tracks)
    threat_id = summary.threats[0].threat_id
    with patch(
        "src.web.predator_routes._fetch_active_track_dicts",
        new=AsyncMock(return_value=fake_tracks),
    ), patch("src.web.app._broadcast_sse", new=AsyncMock()):
        resp = await client.post(
            f"/api/v1/predators/{threat_id}/trigger-deterrent",
            json={"deterrent_type": "acoustic_siren"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["deterrent_activated"] is True
    assert data["deterrent_type"] == "acoustic_siren"


@pytest.mark.asyncio
async def test_rest_simulate_incursion_endpoint(client):
    coyote_id = uuid.uuid4()
    cow_id = uuid.uuid4()
    cop_tracks = [
        _track(cow_id, "HERD-COW-04", "cow", -119.8200, 36.7540),
        _track(coyote_id, "TRK-COYOTE-01", "coyote", -119.8168, 36.7540),
    ]

    from src.web.app import CopState

    fake_state = CopState(
        sensors=[],
        tracks=cop_tracks,
        reachability=[],
        chokepoints=[],
        directives=[],
        generated_at=datetime.now(timezone.utc),
    )

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    herd_result = MagicMock()
    herd_result.mappings.return_value.first.return_value = {"id": cow_id}
    sensor_result = MagicMock()
    sensor_result.scalar.return_value = uuid.uuid4()
    mock_session.execute.side_effect = [herd_result, MagicMock(), sensor_result, MagicMock()]

    with patch(
        "src.web.predator_routes.async_session_factory"
    ) as mock_factory, patch(
        "src.web.app.build_cop_state",
        new=AsyncMock(return_value=fake_state),
    ), patch(
        "src.web.app._broadcast_sse",
        new=AsyncMock(),
    ):
        mock_factory.return_value.__aenter__.return_value = mock_session
        resp = await client.post("/api/v1/predators/simulate-incursion")

    assert resp.status_code == 200
    body = resp.json()
    assert body["predator_track_code"] == "TRK-COYOTE-01"
    assert body["nearest_livestock_code"] == "HERD-COW-04"
    assert body["distance_meters"] < 500

