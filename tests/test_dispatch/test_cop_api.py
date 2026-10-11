"""FastAPI COP endpoint tests with httpx.AsyncClient."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.web.app import app, build_cop_state


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_cop_state_endpoint_returns_snapshot(client):
    fake_state = {
        "sensors": [{"id": "s1", "node_name": "cam-1"}],
        "tracks": [],
        "reachability": [],
        "chokepoints": [{"id": "c1", "name": "Gate"}],
        "directives": [],
        "interceptor": None,
        "generated_at": "2026-10-10T20:00:00+00:00",
    }
    with patch("src.web.app.build_cop_state", new=AsyncMock(return_value=type("S", (), fake_state)())):
        # build_cop_state returns CopState model in production; patch with dict-like object
        from src.web.app import CopState
        from datetime import datetime, timezone

        with patch(
            "src.web.app.build_cop_state",
            new=AsyncMock(
                return_value=CopState(
                    sensors=fake_state["sensors"],
                    tracks=fake_state["tracks"],
                    reachability=fake_state["reachability"],
                    chokepoints=fake_state["chokepoints"],
                    directives=fake_state["directives"],
                    generated_at=datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc),
                )
            ),
        ):
            resp = await client.get("/api/v1/cop/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "sensors" in body
    assert "chokepoints" in body
    assert body["chokepoints"][0]["name"] == "Gate"


@pytest.mark.asyncio
async def test_interceptor_telemetry_returns_routes(client):
    with patch(
        "src.web.app.SpatialService.get_chokepoints",
        new=AsyncMock(
            return_value=[
                {
                    "id": uuid.uuid4(),
                    "name": "South Gate",
                    "location": (-119.818, 36.742),
                    "is_accessible_to_response": True,
                }
            ]
        ),
    ), patch(
        "src.web.app.find_shortest_route",
        new=AsyncMock(
            return_value=type(
                "R",
                (),
                {
                    "transit_time_s": 30.0,
                    "distance_m": 500.0,
                    "coordinates": [(-119.82, 36.74), (-119.818, 36.742)],
                },
            )()
        ),
    ):
        resp = await client.post(
            "/api/v1/interceptor/telemetry",
            json={"lon": -119.82, "lat": 36.74, "unit_id": "unit-1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["interceptor"]["unit_id"] == "unit-1"
    assert len(data["routes"]) >= 1


@pytest.mark.asyncio
async def test_directive_status_update(client):
    directive_id = uuid.uuid4()
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(
        return_value=MagicMock(first=MagicMock(return_value=({"status": "ISSUED"},)))
    )
    mock_session.commit = AsyncMock()

    class _Ctx:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    with patch("src.web.app.async_session_factory", return_value=_Ctx()):
        resp = await client.post(
            f"/api/v1/directives/{directive_id}/status",
            json={"status": "ACKNOWLEDGED"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ACKNOWLEDGED"


@pytest.mark.asyncio
async def test_cop_dashboard_html(client):
    resp = await client.get("/cop")
    assert resp.status_code == 200
    assert "leaflet" in resp.text.lower()
    assert "Rian COP" in resp.text


@pytest.mark.asyncio
async def test_cop_stream_emits_initial_event(client):
    from datetime import datetime, timezone
    from src.web.app import CopState

    with patch(
        "src.web.app.build_cop_state",
        new=AsyncMock(
            return_value=CopState(
                sensors=[],
                tracks=[],
                reachability=[],
                chokepoints=[],
                directives=[],
                generated_at=datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc),
            )
        ),
    ):
        async with client.stream("GET", "/api/v1/cop/stream?max_events=1") as response:
            assert response.status_code == 200
            chunk = b""
            async for part in response.aiter_bytes():
                chunk += part
                if b"event: cop_state" in chunk:
                    break
            assert b"event: cop_state" in chunk


@pytest.mark.asyncio
async def test_build_cop_state_integration_empty_db():
    """Smoke test when database tables are empty or unavailable."""
    try:
        state = await build_cop_state()
    except Exception:
        pytest.skip("Database unavailable")
    assert state.generated_at is not None
