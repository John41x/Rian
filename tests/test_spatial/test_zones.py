"""Domain profiles and spatial zone tests (spec 13)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.common.domains import (
    NORTH_PASTURE_TEST_LAT,
    NORTH_PASTURE_TEST_LON,
    get_domain_profile,
)
from src.spatial.zones import ZoneManager
from src.web.app import app
from src.web.zone_routes import reset_zone_manager


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def test_get_domain_profile_ranch_livestock():
    profile = get_domain_profile("ranch_livestock")
    assert profile.profile_type == "ranch_livestock"
    assert profile.app_title == "Rian Ranch Operations"
    assert profile.entity_label == "Livestock / Wildlife"
    assert profile.zone_label == "Pastures & Paddocks"
    assert len(profile.default_zones) == 4
    names = {z["name"] for z in profile.default_zones}
    assert "North Pasture" in names


def test_get_domain_profile_industrial_terminal():
    profile = get_domain_profile("industrial_terminal")
    assert profile.profile_type == "industrial_terminal"
    assert profile.app_title == "Rian Terminal Security"
    assert profile.entity_label == "Perimeter Intruder"
    assert profile.zone_label == "Storage Yards & Bays"
    assert profile.task_label == "Intercept Directives"


def test_zone_manager_point_in_north_pasture():
    manager = ZoneManager("ranch_livestock")
    zone = manager.find_zone_for_point(NORTH_PASTURE_TEST_LON, NORTH_PASTURE_TEST_LAT)
    assert zone is not None
    assert zone["properties"]["name"] == "North Pasture"
    assert zone["properties"]["zone_code"] == "PASTURE-NORTH"


def test_zone_manager_headcount_from_tracks():
    manager = ZoneManager("ranch_livestock")
    tracks = [
        {
            "status": "ACTIVE",
            "last_known_point": [NORTH_PASTURE_TEST_LON, NORTH_PASTURE_TEST_LAT],
        },
        {
            "status": "ACTIVE",
            "last_known_point": [NORTH_PASTURE_TEST_LON + 0.0001, NORTH_PASTURE_TEST_LAT],
        },
        {
            "status": "ACTIVE",
            "last_known_point": [-119.82, 36.736],
        },
    ]
    updated = manager.update_zone_headcounts(tracks)
    by_code = {z["properties"]["zone_code"]: z["properties"]["headcount"] for z in updated}
    assert by_code["PASTURE-NORTH"] == 2
    assert by_code["PASTURE-SOUTH"] == 1


@pytest.mark.asyncio
async def test_api_domain_profile(client):
    reset_zone_manager()
    resp = await client.get("/api/v1/domain-profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile_type"] == "ranch_livestock"
    assert body["app_title"] == "Rian Ranch Operations"
    assert "persona_system_prompt" in body


@pytest.mark.asyncio
async def test_api_zones_feature_collection(client):
    reset_zone_manager()
    resp = await client.get("/api/v1/zones")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    features = body["features"]
    assert len(features) >= 4
    north = next(
        f for f in features if f["properties"]["zone_code"] == "PASTURE-NORTH"
    )
    assert north["properties"]["name"] == "North Pasture"
    assert "headcount" in north["properties"]
