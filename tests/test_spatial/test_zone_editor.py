"""Interactive pasture polygon editor and geofence breach tests (spec 18)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from src.common.domains import NORTH_PASTURE_TEST_LAT, NORTH_PASTURE_TEST_LON
from src.spatial.zones import (
    GeofenceBreachEvent,
    ZoneCreateRequest,
    ZoneManager,
    ZoneUpdateRequest,
)
from src.web.app import app
from src.web.zone_routes import reset_zone_manager

_TEST_POLYGON = [
    [-119.900, 36.800],
    [-119.898, 36.800],
    [-119.898, 36.802],
    [-119.900, 36.802],
]


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def manager():
    reset_zone_manager()
    return ZoneManager("ranch_livestock")


def test_create_zone_validates_polygon(manager: ZoneManager):
    req = ZoneCreateRequest(
        name="Alfalfa South 40",
        zone_code="PASTURE-ALFALFA-S",
        zone_type="pasture",
        color="#22c55e",
        coordinates=_TEST_POLYGON,
    )
    feature = manager.create_zone(req)
    assert feature["type"] == "Feature"
    assert feature["properties"]["name"] == "Alfalfa South 40"
    assert feature["properties"]["zone_code"] == "PASTURE-ALFALFA-S"
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]


def test_create_zone_rejects_invalid_polygon(manager: ZoneManager):
    with pytest.raises(ValueError):
        manager.create_zone(
            ZoneCreateRequest(
                name="Bad",
                zone_code="BAD-ZONE",
                coordinates=[[-119.9, 36.8], [-119.89, 36.8]],
            )
        )


def test_update_and_delete_zone(manager: ZoneManager):
    created = manager.create_zone(
        ZoneCreateRequest(
            name="Drawn Paddock",
            zone_code="PADDOCK-DRAWN",
            coordinates=_TEST_POLYGON,
        )
    )
    zone_id = created["properties"]["zone_id"]
    updated = manager.update_zone(
        zone_id,
        ZoneUpdateRequest(name="Renamed Paddock", color="#ff00ff"),
    )
    assert updated is not None
    assert updated["properties"]["name"] == "Renamed Paddock"
    assert updated["properties"]["color"] == "#ff00ff"
    assert manager.delete_zone(zone_id) is True
    assert manager.update_zone(zone_id, ZoneUpdateRequest(name="x")) is None


def test_detect_geofence_breaches_outside_pasture(manager: ZoneManager):
    track_id = UUID("11111111-1111-4111-8111-111111111111")
    tracks = [
        {
            "track_id": str(track_id),
            "track_code": "TRK-COW-STRAY",
            "object_class": "cow",
            "status": "ACTIVE",
            "last_known_point": [-119.95, 36.85],
            "assigned_zone_name": "North Pasture",
        }
    ]
    events = manager.detect_geofence_breaches(tracks)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, GeofenceBreachEvent)
    assert event.breach_type == "STRAY_OUT_OF_BOUNDS"
    assert event.track_code == "TRK-COW-STRAY"
    assert event.track_id == track_id


def test_detect_geofence_no_breach_inside_pasture(manager: ZoneManager):
    tracks = [
        {
            "track_code": "TRK-COW-OK",
            "object_class": "cow",
            "status": "ACTIVE",
            "last_known_point": [NORTH_PASTURE_TEST_LON, NORTH_PASTURE_TEST_LAT],
        }
    ]
    events = manager.detect_geofence_breaches(tracks)
    assert events == []


@pytest.mark.asyncio
async def test_api_create_zone(client):
    reset_zone_manager()
    body = {
        "name": "API Drawn Pasture",
        "zone_code": "API-DRAW-01",
        "zone_type": "pasture",
        "color": "#22c55e",
        "coordinates": _TEST_POLYGON,
    }
    with patch("src.web.app._broadcast_sse", new=AsyncMock()) as mock_sse:
        resp = await client.post("/api/v1/zones", json=body)
    assert resp.status_code == 201
    feature = resp.json()
    assert feature["properties"]["zone_code"] == "API-DRAW-01"
    mock_sse.assert_awaited()
    assert mock_sse.await_args.args[0]["type"] == "zone_created"


@pytest.mark.asyncio
async def test_api_update_and_delete_zone(client):
    reset_zone_manager()
    with patch("src.web.app._broadcast_sse", new=AsyncMock()):
        create = await client.post(
            "/api/v1/zones",
            json={
                "name": "Mutable Zone",
                "zone_code": "API-MUTABLE",
                "coordinates": _TEST_POLYGON,
            },
        )
    zone_id = create.json()["properties"]["zone_id"]
    with patch("src.web.app._broadcast_sse", new=AsyncMock()):
        put = await client.put(
            f"/api/v1/zones/{zone_id}",
            json={"name": "Updated Zone"},
        )
    assert put.status_code == 200
    assert put.json()["properties"]["name"] == "Updated Zone"
    with patch("src.web.app._broadcast_sse", new=AsyncMock()):
        delete = await client.delete(f"/api/v1/zones/{zone_id}")
    assert delete.status_code == 204


@pytest.mark.asyncio
async def test_api_geofence_breaches_endpoint(client):
    reset_zone_manager()
    manager = ZoneManager("ranch_livestock")
    breaches = manager.detect_geofence_breaches(
        [
            {
                "track_code": "TRK-COW-OUT",
                "object_class": "cow",
                "status": "ACTIVE",
                "last_known_point": [-119.95, 36.85],
            }
        ]
    )
    assert len(breaches) == 1
    with patch(
        "src.web.zone_routes._fetch_active_tracks",
        new=AsyncMock(
            return_value=[
                {
                    "track_code": "TRK-COW-OUT",
                    "object_class": "cow",
                    "status": "ACTIVE",
                    "last_known_point": [-119.95, 36.85],
                }
            ]
        ),
    ):
        resp = await client.get("/api/v1/zones/breaches")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["breach_type"] == "STRAY_OUT_OF_BOUNDS"
