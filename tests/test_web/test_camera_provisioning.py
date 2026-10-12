"""API tests for camera fleet provisioning (spec 11)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.common.config import settings
from src.common.database import async_session_factory
from src.common.schemas import EntityClass, TrackState, TrackStatus
from src.web.app import _fov_wedge_geojson, create_app


def _db_available() -> bool:
    try:
        with psycopg.connect(settings.sync_database_url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def db_available():
    if not _db_available():
        pytest.skip("PostgreSQL not available")
    return True


@pytest.fixture
async def client(db_available):
    from src.common.auth import bootstrap_initial_admin, ensure_users_table
    from src.common.database import async_session_factory

    async with async_session_factory() as session:
        await ensure_users_table(session)
        await bootstrap_initial_admin(session)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def unique_camera_payload():
    suffix = uuid4().hex[:8]
    return {
        "node_name": f"cam-test-{suffix}",
        "sender_email": f"cam-test-{suffix}@ranch.domain",
        "lon": -119.825,
        "lat": 36.751,
        "azimuth_deg": 135.0,
        "fov_angle": 85.0,
        "detection_radius_m": 25.0,
    }


@pytest.mark.asyncio
async def test_list_cameras(client, unique_camera_payload):
    create_resp = await client.post("/api/v1/cameras", json=unique_camera_payload)
    assert create_resp.status_code == 201
    created_id = create_resp.json()["id"]

    list_resp = await client.get("/api/v1/cameras")
    assert list_resp.status_code == 200
    cameras = list_resp.json()
    assert isinstance(cameras, list)
    ids = {c["id"] for c in cameras}
    assert created_id in ids

    match = next(c for c in cameras if c["id"] == created_id)
    assert match["node_name"] == unique_camera_payload["node_name"]
    assert match["sender_email"] == unique_camera_payload["sender_email"]
    assert match["location"] == [unique_camera_payload["lon"], unique_camera_payload["lat"]]
    assert match["fov_wedge"]["type"] == "Polygon"


@pytest.mark.asyncio
async def test_create_camera_valid_fov_wedge(client, unique_camera_payload):
    with patch("src.web.app._broadcast_sse", new=AsyncMock()) as mock_sse:
        resp = await client.post("/api/v1/cameras", json=unique_camera_payload)

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "active"
    assert body["battery_level"] == 100

    expected_wedge = _fov_wedge_geojson(
        unique_camera_payload["lon"],
        unique_camera_payload["lat"],
        unique_camera_payload["azimuth_deg"],
        unique_camera_payload["fov_angle"],
        unique_camera_payload["detection_radius_m"],
    )
    assert body["fov_wedge"] == expected_wedge
    assert len(body["fov_wedge"]["coordinates"][0]) >= 4

    mock_sse.assert_awaited_once()
    sse_payload = mock_sse.await_args.args[0]
    assert sse_payload["type"] == "camera_provisioned"


@pytest.mark.asyncio
async def test_patch_camera_updates_azimuth_and_position(client, unique_camera_payload):
    create_resp = await client.post("/api/v1/cameras", json=unique_camera_payload)
    camera_id = create_resp.json()["id"]

    patch_body = {
        "lon": -119.83,
        "lat": 36.755,
        "azimuth_deg": 270.0,
    }
    with patch("src.web.app._broadcast_sse", new=AsyncMock()):
        patch_resp = await client.patch(f"/api/v1/cameras/{camera_id}", json=patch_body)

    assert patch_resp.status_code == 200
    updated = patch_resp.json()
    assert updated["location"] == [patch_body["lon"], patch_body["lat"]]
    assert updated["azimuth_deg"] == patch_body["azimuth_deg"]

    expected_wedge = _fov_wedge_geojson(
        patch_body["lon"],
        patch_body["lat"],
        patch_body["azimuth_deg"],
        unique_camera_payload["fov_angle"],
        unique_camera_payload["detection_radius_m"],
    )
    assert updated["fov_wedge"] == expected_wedge


@pytest.mark.asyncio
async def test_delete_camera_decommissions(client, unique_camera_payload):
    create_resp = await client.post("/api/v1/cameras", json=unique_camera_payload)
    camera_id = create_resp.json()["id"]

    with patch("src.web.app._broadcast_sse", new=AsyncMock()):
        del_resp = await client.delete(f"/api/v1/cameras/{camera_id}")

    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "decommissioned"

    list_resp = await client.get("/api/v1/cameras")
    cam = next(c for c in list_resp.json() if c["id"] == camera_id)
    assert cam["status"] == "offline"


@pytest.mark.asyncio
async def test_test_probe_triggers_detection_event(client, unique_camera_payload):
    create_resp = await client.post("/api/v1/cameras", json=unique_camera_payload)
    camera_id = create_resp.json()["id"]

    track_id = uuid4()
    mock_track = TrackState(
        track_id=track_id,
        track_code="TRK-PROBE-001",
        object_class=EntityClass.PERSON,
        status=TrackStatus.ACTIVE,
        first_detected_at=datetime.now(timezone.utc),
        last_detected_at=datetime.now(timezone.utc),
        last_known_point=(-119.825, 36.751),
    )

    mock_engine = AsyncMock()
    mock_engine.handle_detection_event = AsyncMock(return_value=[mock_track])

    with patch("src.engine.get_engine", return_value=mock_engine), patch(
        "src.web.app._broadcast_sse", new=AsyncMock()
    ):
        probe_resp = await client.post(f"/api/v1/cameras/{camera_id}/test-probe")

    assert probe_resp.status_code == 200
    data = probe_resp.json()
    assert data["status"] == "probe_complete"
    assert data["camera_id"] == camera_id
    assert data["tracks"][0]["track_code"] == "TRK-PROBE-001"
    assert data["directive_status"] == "evaluated"
    mock_engine.handle_detection_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_camera_by_id(client, unique_camera_payload):
    create_resp = await client.post("/api/v1/cameras", json=unique_camera_payload)
    camera_id = create_resp.json()["id"]

    get_resp = await client.get(f"/api/v1/cameras/{camera_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["id"] == camera_id
    assert body["node_name"] == unique_camera_payload["node_name"]
    assert body["sender_email"] == unique_camera_payload["sender_email"]

    missing = await client.get(f"/api/v1/cameras/{uuid4()}")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_list_camera_events(client, unique_camera_payload):
    create_resp = await client.post("/api/v1/cameras", json=unique_camera_payload)
    camera_id = create_resp.json()["id"]
    event_id = uuid4()
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO detection_events (
                    id, sensor_id, event_timestamp, object_class, confidence,
                    bounding_box, raw_image_path, processed_latency_ms
                )
                VALUES (
                    :id, :sensor_id, :event_timestamp, :object_class, :confidence,
                    CAST(:bounding_box AS jsonb), :raw_image_path, :processed_latency_ms
                )
                """
            ),
            {
                "id": str(event_id),
                "sensor_id": camera_id,
                "event_timestamp": now,
                "object_class": "person",
                "confidence": 0.88,
                "bounding_box": json.dumps([0.1, 0.2, 0.5, 0.6]),
                "raw_image_path": "events/test.jpg",
                "processed_latency_ms": 12,
            },
        )
        await session.commit()

    events_resp = await client.get(f"/api/v1/cameras/{camera_id}/events")
    assert events_resp.status_code == 200
    events = events_resp.json()
    assert isinstance(events, list)
    assert len(events) >= 1
    match = next(e for e in events if e["id"] == str(event_id))
    assert match["object_class"] == "person"
    assert match["confidence"] == 0.88
    assert match["processed_latency_ms"] == 12

    missing = await client.get(f"/api/v1/cameras/{uuid4()}/events")
    assert missing.status_code == 404
