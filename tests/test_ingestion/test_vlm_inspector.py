"""VLM forensic snapshot inspector (spec 16)."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.ingestion.vlm_inspector import (
    BoundingBoxOverlay,
    SnapshotRecord,
    VlmForensicInspector,
    VlmInspectionReport,
    get_vlm_inspector,
    reset_vlm_inspector_for_tests,
)
from src.web.app import app
from tools.mock_camera import generate_synthetic_jpeg


@pytest.fixture(autouse=True)
def isolated_inspector(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.ingestion.vlm_inspector.settings.STORAGE_ROOT",
        str(tmp_path),
    )
    reset_vlm_inspector_for_tests()
    yield
    reset_vlm_inspector_for_tests()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _sample_box() -> BoundingBoxOverlay:
    return BoundingBoxOverlay(
        label="cow",
        confidence=0.91,
        ymin=0.2,
        xmin=0.25,
        ymax=0.75,
        xmax=0.6,
        color="#22c55e",
    )


class TestRecordSnapshot:
    def test_record_snapshot_stores_metadata_and_boxes(self):
        inspector = VlmForensicInspector()
        sensor_id = uuid4()
        image = generate_synthetic_jpeg("cow")
        record = inspector.record_snapshot(
            sensor_id=sensor_id,
            camera_name="North Gate Cam",
            entity_class="cow",
            confidence=0.91,
            bounding_boxes=[_sample_box()],
            image_bytes=image,
        )
        assert isinstance(record, SnapshotRecord)
        assert record.sensor_id == sensor_id
        assert record.camera_name == "North Gate Cam"
        assert len(record.bounding_boxes) == 1
        assert record.bounding_boxes[0].label == "cow"
        assert record.image_url.endswith(f"/api/v1/snapshots/{record.snapshot_id}/image")

        fetched = inspector.get_snapshot(record.snapshot_id)
        assert fetched is not None
        assert fetched.snapshot_id == record.snapshot_id

        camera_list = inspector.get_camera_snapshots(sensor_id, limit=5)
        assert len(camera_list) == 1
        assert camera_list[0].snapshot_id == record.snapshot_id

    def test_get_snapshot_image_returns_valid_jpeg(self):
        inspector = VlmForensicInspector()
        image = generate_synthetic_jpeg("truck")
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Yard Cam",
            entity_class="truck",
            confidence=0.8,
            bounding_boxes=[_sample_box()],
            image_bytes=image,
        )
        data = inspector.get_snapshot_image(record.snapshot_id)
        assert data is not None
        assert data[:2] == b"\xff\xd8"
        img = Image.open(BytesIO(data))
        assert img.format == "JPEG"

    def test_get_snapshot_image_synthetic_fallback_when_bytes_missing(self):
        inspector = VlmForensicInspector()
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Fallback Cam",
            entity_class="person",
            confidence=0.7,
            bounding_boxes=[_sample_box()],
            image_bytes=generate_synthetic_jpeg("person"),
        )
        inspector._image_bytes.pop(record.snapshot_id, None)
        path = inspector._image_path(record.snapshot_id)
        if path.exists():
            path.unlink()
        data = inspector.get_snapshot_image(record.snapshot_id)
        assert data is not None
        assert data[:2] == b"\xff\xd8"


class TestAnalyzeSnapshot:
    @pytest.mark.asyncio
    async def test_local_reasoner_produces_report(self, monkeypatch):
        monkeypatch.setattr("src.ingestion.vlm_inspector.settings.DEEPSEEK_API_KEY", None)
        inspector = get_vlm_inspector()
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Pasture Cam",
            entity_class="cow",
            confidence=0.88,
            bounding_boxes=[_sample_box()],
        )
        report = await inspector.analyze_snapshot(record.snapshot_id)
        assert isinstance(report, VlmInspectionReport)
        assert report.snapshot_id == record.snapshot_id
        assert report.entity_class == "cow"
        assert report.visual_description
        assert report.environmental_context
        assert report.findings
        assert report.recommended_action
        assert report.provider == "Rian Local Forensic Analyzer"

        updated = inspector.get_snapshot(record.snapshot_id)
        assert updated is not None
        assert updated.vlm_analyzed is True

    @pytest.mark.asyncio
    async def test_deepseek_mock_produces_report(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.vlm_inspector.settings.DEEPSEEK_API_KEY",
            "test-key",
        )
        inspector = get_vlm_inspector()
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Gate Cam",
            entity_class="truck",
            confidence=0.92,
            bounding_boxes=[_sample_box()],
        )
        mock_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "snapshot_id": str(record.snapshot_id),
                                "entity_class": "truck",
                                "visual_description": "White pickup with ladder rack.",
                                "environmental_context": "Gravel access road, overcast.",
                                "risk_assessment": "MODERATE // UNAUTHORIZED VEHICLE",
                                "findings": ["Open tailgate", "No plate visible"],
                                "recommended_action": "Dispatch guard to verify credentials.",
                                "provider": "DeepSeek-VLM",
                            }
                        )
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = AsyncMock()
            mock_resp.raise_for_status = lambda: None
            mock_resp.json = lambda: mock_payload
            mock_post.return_value = mock_resp
            report = await inspector.analyze_snapshot(record.snapshot_id)

        assert report.provider == "DeepSeek-VLM"
        assert "pickup" in report.visual_description.lower()
        assert len(report.findings) == 2


class TestSnapshotRestEndpoints:
    @pytest.mark.asyncio
    async def test_camera_snapshots_list(self, client):
        inspector = get_vlm_inspector()
        camera_id = uuid4()
        record = inspector.record_snapshot(
            sensor_id=camera_id,
            camera_name="List Cam",
            entity_class="cow",
            confidence=0.85,
            bounding_boxes=[_sample_box()],
        )
        resp = await client.get(f"/api/v1/cameras/{camera_id}/snapshots")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) >= 1
        assert body[0]["snapshot_id"] == str(record.snapshot_id)

    @pytest.mark.asyncio
    async def test_snapshot_image_endpoint(self, client):
        inspector = get_vlm_inspector()
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Image Cam",
            entity_class="dog",
            confidence=0.77,
            bounding_boxes=[_sample_box()],
        )
        resp = await client.get(f"/api/v1/snapshots/{record.snapshot_id}/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/jpeg")
        assert resp.content[:2] == b"\xff\xd8"

    @pytest.mark.asyncio
    async def test_vlm_analyze_endpoint(self, client):
        inspector = get_vlm_inspector()
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Analyze Cam",
            entity_class="person",
            confidence=0.95,
            bounding_boxes=[_sample_box()],
        )
        resp = await client.post(f"/api/v1/snapshots/{record.snapshot_id}/vlm-analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot_id"] == str(record.snapshot_id)
        assert data["provider"]

    @pytest.mark.asyncio
    async def test_entity_track_snapshot_endpoint(self, client):
        track_id = uuid4()
        inspector = get_vlm_inspector()
        record = inspector.record_snapshot(
            sensor_id=uuid4(),
            camera_name="Track Cam",
            entity_class="cow",
            confidence=0.9,
            bounding_boxes=[_sample_box()],
            track_id=track_id,
        )
        resp = await client.get(f"/api/v1/cop/entity/{track_id}/snapshot")
        assert resp.status_code == 200
        assert resp.json()["snapshot_id"] == str(record.snapshot_id)

    @pytest.mark.asyncio
    async def test_entity_track_snapshot_missing_returns_404(self, client):
        missing = UUID("00000000-0000-0000-0000-000000000099")
        resp = await client.get(f"/api/v1/cop/entity/{missing}/snapshot")
        assert resp.status_code == 404
