"""Roll-call service and report API tests (spec 19)."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.common.domains import NORTH_PASTURE_TEST_LAT, NORTH_PASTURE_TEST_LON
from src.ingestion.vlm_inspector import BoundingBoxOverlay, reset_vlm_inspector_for_tests
from src.spatial.rollcall_service import RollCallService, get_rollcall_service, reset_rollcall_service_for_tests
from src.spatial.zones import ZoneManager
from src.web.app import CopState, app


@pytest.fixture
def zone_manager() -> ZoneManager:
    return ZoneManager("ranch_livestock")


@pytest.fixture
def rollcall_service(zone_manager: ZoneManager) -> RollCallService:
    return reset_rollcall_service_for_tests(zone_manager=zone_manager)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _track_payload(
    *,
    hours_ago: float,
    lon: float = NORTH_PASTURE_TEST_LON,
    lat: float = NORTH_PASTURE_TEST_LAT,
    track_code: str = "COW-01",
    now: datetime | None = None,
) -> dict:
    reference = now or datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    last_seen = reference - timedelta(hours=hours_ago)
    track_id = uuid.uuid4()
    return {
        "track_id": str(track_id),
        "track_code": track_code,
        "object_class": "cow",
        "last_known_point": [lon, lat],
        "last_detected_at": last_seen.isoformat(),
        "last_camera_name": "North Gate Cam",
        "_uuid": track_id,
    }


def test_generate_rollcall_hours_zones_and_missing_status(rollcall_service: RollCallService):
    now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    active = _track_payload(hours_ago=2.0, track_code="ACTIVE-01", now=now)
    overdue = _track_payload(hours_ago=10.0, track_code="OVERDUE-01", now=now)
    missing = _track_payload(hours_ago=30.0, track_code="MISSING-01", now=now)
    cop_state = {"tracks": [active, overdue, missing]}

    summary = rollcall_service.generate_rollcall(cop_state, now=now)

    assert summary.total_tracked_count == 3
    assert summary.missing_count == 1
    assert summary.accounted_count == 2
    assert "North Pasture" in summary.zone_distribution

    by_code = {item.track_code: item for item in summary.items}
    assert by_code["ACTIVE-01"].health_status == "VERIFIED_ACTIVE"
    assert by_code["ACTIVE-01"].hours_since_last_seen == 2.0
    assert by_code["OVERDUE-01"].health_status == "OVERDUE_CHECK"
    assert by_code["MISSING-01"].health_status == "MISSING_CRITICAL"


def test_generate_rollcall_includes_vlm_summary_when_snapshot_exists(
    rollcall_service: RollCallService,
):
    inspector = reset_vlm_inspector_for_tests()
    now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    track = _track_payload(hours_ago=1.0, now=now)
    track_id = track["_uuid"]
    inspector.record_snapshot(
        sensor_id=uuid.uuid4(),
        camera_name="North Gate Cam",
        entity_class="cow",
        confidence=0.91,
        bounding_boxes=[
            BoundingBoxOverlay(
                label="cow",
                confidence=0.91,
                ymin=0.2,
                xmin=0.2,
                ymax=0.8,
                xmax=0.8,
            )
        ],
        track_id=track_id,
    )
    cop_state = {"tracks": [track]}
    summary = rollcall_service.generate_rollcall(cop_state, now=now)
    assert summary.items[0].vlm_summary is not None
    assert "cow" in summary.items[0].vlm_summary.lower()


def test_format_csv_headers_and_rows(rollcall_service: RollCallService):
    now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    cop_state = {"tracks": [_track_payload(hours_ago=3.5, now=now)]}
    summary = rollcall_service.generate_rollcall(cop_state, now=now)
    csv_text = rollcall_service.format_csv(summary)

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    assert rows[0] == [
        "Track Code",
        "Class",
        "Current Pasture/Zone",
        "Last Seen (UTC)",
        "Hours Ago",
        "Last Camera",
        "Health Status",
        "VLM Summary",
    ]
    assert rows[1][0] == "COW-01"
    assert rows[1][6] == "VERIFIED_ACTIVE"


@pytest.mark.asyncio
async def test_herd_rollcall_json_endpoint(client):
    track = _track_payload(hours_ago=5.0)
    fake = CopState(
        sensors=[],
        tracks=[{k: v for k, v in track.items() if k != "_uuid"}],
        reachability=[],
        chokepoints=[],
        directives=[],
        generated_at=datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc),
    )
    with patch("src.web.app.build_cop_state", new=AsyncMock(return_value=fake)):
        res = await client.get("/api/v1/reports/herd-rollcall")
    assert res.status_code == 200
    body = res.json()
    assert body["total_tracked_count"] == 1
    assert body["items"][0]["health_status"] == "OVERDUE_CHECK"


@pytest.mark.asyncio
async def test_herd_rollcall_csv_endpoint(client):
    track = _track_payload(hours_ago=1.0)
    fake = CopState(
        sensors=[],
        tracks=[{k: v for k, v in track.items() if k != "_uuid"}],
        reachability=[],
        chokepoints=[],
        directives=[],
        generated_at=datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc),
    )
    with patch("src.web.app.build_cop_state", new=AsyncMock(return_value=fake)):
        res = await client.get("/api/v1/reports/herd-rollcall/csv")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers.get("content-disposition", "")
    assert "herd_rollcall_" in res.headers.get("content-disposition", "")
    rows = list(csv.reader(io.StringIO(res.text)))
    assert rows[0][0] == "Track Code"
    assert rows[1][0] == "COW-01"
