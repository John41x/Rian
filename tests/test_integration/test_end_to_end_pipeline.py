"""
End-to-end integration test suite for the Rian Autonomous Spatial State Engine.

Validates the full operational pipeline:
  Inbound Telemetry -> Ingestion & CV -> Kinematic Association ->
  Isochrone Wavefront Propagation -> Chokepoint Interception ->
  Twilio Alert Dispatch -> COP Broadcast.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from src.common.schemas import (
    BoundingBox,
    DetectionEventInbound,
    DetectionEventProcessed,
    DetectionItem,
    EntityClass,
    TacticalDirective,
    TrackState,
    TrackStatus,
)
from src.dispatch.alerter import AlertResult, TwilioAlerter
from src.engine import RianEngine
from src.ingestion.inference import InferenceConfig, YOLOInferenceProcessor
from src.web.app import app


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    import io
    from PIL import Image

    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_engine_lifecycle():
    engine = RianEngine(
        enable_smtp=False,
        enable_worker=False,
    )
    assert not engine.is_running
    await engine.start()
    assert engine.is_running
    await engine.stop()
    assert not engine.is_running


@pytest.mark.asyncio
async def test_engine_handles_detection_event_end_to_end():
    broadcast_events = []

    async def fake_broadcast(payload):
        broadcast_events.append(payload)

    engine = RianEngine(
        enable_smtp=False,
        enable_worker=False,
        interceptor_position_provider=lambda: (-119.82, 36.74),
        cop_broadcast=fake_broadcast,
    )

    track_id = uuid4()
    mock_track = TrackState(
        track_id=track_id,
        track_code="TRK-E2E-001",
        object_class=EntityClass.TRUCK,
        status=TrackStatus.ACTIVE,
        first_detected_at=datetime.now(timezone.utc),
        last_detected_at=datetime.now(timezone.utc),
        last_known_point=(-119.819, 36.741),
        estimated_velocity_ms=12.0,
        reachability_polygon={"type": "Polygon", "coordinates": []},
    )

    mock_directive = TacticalDirective(
        directive_id=uuid4(),
        track_id=track_id,
        track_code="TRK-E2E-001",
        target_class=EntityClass.TRUCK,
        target_chokepoint_id=uuid4(),
        chokepoint_name="North Gate",
        chokepoint_location=(-119.818, 36.742),
        target_arrival_time=datetime.now(timezone.utc),
        interceptor_arrival_time=datetime.now(timezone.utc),
        safety_margin_seconds=240.0,
        action_message="INTERCEPT AT North Gate — margin 4.0 min",
    )

    engine.associator.associate = AsyncMock(return_value=mock_track)
    engine.solver.solve = AsyncMock(return_value=mock_directive)
    engine.alerter.send_alert = AsyncMock(
        return_value=AlertResult(
            dispatched=True,
            channel="sms",
            message="[RIAN TACTICAL ALERT] Intercept North Gate",
            mock=True,
        )
    )

    event = DetectionEventProcessed(
        event_id=uuid4(),
        sensor_id=uuid4(),
        node_name="cam-gate-north",
        sensor_location=(-119.819, 36.741),
        sensor_azimuth_deg=180.0,
        event_timestamp=datetime.now(timezone.utc),
        detections=[
            DetectionItem(
                object_class=EntityClass.TRUCK,
                confidence=0.88,
                bounding_box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
            )
        ],
        raw_image_path="snapshots/test.jpg",
        processed_latency_ms=45,
    )

    results = await engine.handle_detection_event(event)
    assert len(results) == 1
    assert results[0].track_code == "TRK-E2E-001"

    engine.associator.associate.assert_awaited_once()
    engine.solver.solve.assert_awaited_once()
    engine.alerter.send_alert.assert_awaited_once_with(mock_directive)
    assert len(broadcast_events) >= 1
    assert broadcast_events[0]["track_code"] == "TRK-E2E-001"


@pytest.mark.asyncio
async def test_engine_inbound_ingestion_queue_processing(sample_jpeg_bytes):
    engine = RianEngine(
        enable_smtp=False,
        enable_worker=False,
        inference=YOLOInferenceProcessor(InferenceConfig(force_mock=True)),
    )

    mock_track = TrackState(
        track_id=uuid4(),
        track_code="TRK-TEST",
        object_class=EntityClass.PERSON,
        status=TrackStatus.ACTIVE,
        first_detected_at=datetime.now(timezone.utc),
        last_detected_at=datetime.now(timezone.utc),
        last_known_point=(-119.82, 36.74),
        estimated_velocity_ms=1.5,
        reachability_polygon={"type": "Polygon", "coordinates": []},
    )
    engine.associator.associate = AsyncMock(return_value=mock_track)
    engine.solver.solve = AsyncMock(return_value=None)

    inbound = DetectionEventInbound(
        sensor_sender_email="cam-gate-north@ranch.domain",
        received_at=datetime.now(timezone.utc),
        image_bytes=sample_jpeg_bytes,
        image_filename="snapshot_01.jpg",
    )

    processed = await engine.ingestion.process_inbound(inbound)
    assert processed is not None
    assert processed.node_name == "cam-gate-north"
    assert len(processed.detections) >= 1
    engine.associator.associate.assert_awaited_once()


@pytest.mark.asyncio
async def test_webhook_endpoint_enqueues_telemetry(sample_jpeg_bytes):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {
            "attachment": ("intruder.jpg", sample_jpeg_bytes, "image/jpeg"),
        }
        data = {
            "from": "cam-gate-north@ranch.domain",
        }
        resp = await client.post(
            "/api/v1/telemetry/smtp-webhook",
            data=data,
            files=files,
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "accepted"}
