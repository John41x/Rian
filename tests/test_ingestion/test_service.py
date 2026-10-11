"""End-to-end ingestion service dispatch tests (DB mocked)."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.common.schemas import DetectionEventInbound, DetectionEventProcessed, EntityClass
from src.ingestion.inference import InferenceConfig, YOLOInferenceProcessor
from src.ingestion.repository import SensorRecord
from src.ingestion.service import IngestionService, save_snapshot_image


@pytest.mark.asyncio
async def test_end_to_end_event_dispatch(tmp_path, synthetic_jpeg_bytes, monkeypatch):
    sensor_id = uuid4()
    event_id = uuid4()
    received_at = datetime(2026, 10, 10, 21, 30, 0, tzinfo=timezone.utc)

    async def fake_resolve(session, sender_email):
        assert sender_email == "camera@reolink.domain"
        return SensorRecord(
            id=sensor_id,
            node_name="Gate North",
            location=(-119.82, 36.74),
            azimuth_deg=45.0,
        )

    async def fake_insert(session, **kwargs):
        assert kwargs["sensor_id"] == sensor_id
        assert len(kwargs["detections"]) == 1
        assert kwargs["detections"][0].object_class == EntityClass.PERSON
        return [event_id]

    monkeypatch.setattr(
        "src.ingestion.service.resolve_sensor_by_email", fake_resolve
    )
    monkeypatch.setattr(
        "src.ingestion.service.insert_detection_events", fake_insert
    )

    class _SessionCtx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return False

    def session_factory():
        return _SessionCtx()

    dispatched: list[DetectionEventProcessed] = []

    async def on_processed(event: DetectionEventProcessed):
        dispatched.append(event)

    service = IngestionService(
        session_factory=session_factory,
        inference=YOLOInferenceProcessor(InferenceConfig(force_mock=True)),
        listeners=[on_processed],
    )

    inbound = DetectionEventInbound(
        sensor_sender_email="camera@reolink.domain",
        received_at=received_at,
        image_bytes=synthetic_jpeg_bytes,
    )

    result = await service.process_inbound(inbound)

    assert result is not None
    assert result.event_id == event_id
    assert result.sensor_id == sensor_id
    assert result.node_name == "Gate North"
    assert len(dispatched) == 1
    assert dispatched[0].event_id == event_id
    assert dispatched[0].detections[0].object_class == EntityClass.PERSON


def test_save_snapshot_image_path_layout(tmp_path, synthetic_jpeg_bytes):
    sensor_id = uuid4()
    received_at = datetime(2026, 10, 10, 15, 4, 5, tzinfo=timezone.utc)
    path = save_snapshot_image(
        synthetic_jpeg_bytes,
        sensor_id=sensor_id,
        received_at=received_at,
        storage_root=str(tmp_path),
    )
    assert "2026/10/10" in path.replace("\\", "/")
    assert str(sensor_id) in path
    assert path.endswith(".jpg")
