"""Ingestion worker: queue consumer, inference, storage, and event dispatch."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Union
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.common.config import settings
from src.common.schemas import DetectionEventInbound, DetectionEventProcessed
from src.ingestion.inference import YOLOInferenceProcessor
from src.ingestion.queue import get_ingestion_queue
from src.ingestion.repository import insert_detection_events, resolve_sensor_by_email

logger = logging.getLogger(__name__)

EventListener = Callable[
    [DetectionEventProcessed], Union[None, Awaitable[None]]
]


class IngestionService:
    """Pulls inbound snapshots from the queue and runs the CV pipeline."""

    def __init__(
        self,
        queue: Optional[asyncio.Queue] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        inference: Optional[YOLOInferenceProcessor] = None,
        listeners: Optional[List[EventListener]] = None,
        record_empty_triggers: bool = False,
    ) -> None:
        self.queue = queue or get_ingestion_queue()
        self.session_factory = session_factory
        self.inference = inference or YOLOInferenceProcessor()
        self.listeners: List[EventListener] = list(listeners or [])
        self.record_empty_triggers = record_empty_triggers
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def add_listener(self, listener: EventListener) -> None:
        self.listeners.append(listener)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="ingestion-worker")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def process_inbound(self, inbound: DetectionEventInbound) -> Optional[DetectionEventProcessed]:
        """Process a single inbound snapshot (used by worker and tests)."""
        started = datetime.now(timezone.utc)
        detections = await self.inference.predict_async(inbound.image_bytes)
        if not detections:
            if self.record_empty_triggers:
                logger.info(
                    "No detections above threshold for %s",
                    inbound.sensor_sender_email,
                )
            else:
                logger.debug(
                    "Discarding empty trigger from %s", inbound.sensor_sender_email
                )
            return None

        if self.session_factory is None:
            raise RuntimeError("session_factory is required for persistence")

        async with self.session_factory() as session:
            sensor = await resolve_sensor_by_email(
                session, inbound.sensor_sender_email
            )
            if sensor is None:
                return None

            image_path = save_snapshot_image(
                inbound.image_bytes,
                sensor_id=sensor.id,
                received_at=inbound.received_at,
            )

            latency_ms = int(
                (datetime.now(timezone.utc) - started).total_seconds() * 1000
            )
            event_ids = await insert_detection_events(
                session,
                sensor_id=sensor.id,
                event_timestamp=inbound.received_at,
                detections=detections,
                raw_image_path=image_path,
                processed_latency_ms=latency_ms,
            )

        processed = DetectionEventProcessed(
            event_id=event_ids[0],
            sensor_id=sensor.id,
            node_name=sensor.node_name,
            sensor_location=sensor.location,
            sensor_azimuth_deg=sensor.azimuth_deg,
            event_timestamp=inbound.received_at,
            detections=detections,
            raw_image_path=image_path,
            processed_latency_ms=latency_ms,
        )
        await self._dispatch(processed)
        return processed

    async def _run_loop(self) -> None:
        while self._running:
            inbound: DetectionEventInbound = await self.queue.get()
            try:
                await self.process_inbound(inbound)
            except Exception:
                logger.exception("Failed to process inbound ingestion item")
            finally:
                self.queue.task_done()

    async def _dispatch(self, event: DetectionEventProcessed) -> None:
        for listener in self.listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Detection event listener failed")


def save_snapshot_image(
    image_bytes: bytes,
    *,
    sensor_id,
    received_at: datetime,
    storage_root: Optional[str] = None,
) -> str:
    root = Path(storage_root or settings.STORAGE_ROOT)
    ts = received_at.astimezone(timezone.utc)
    dest_dir = root / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sensor_id}_{ts:%Y%m%dT%H%M%S%fZ}_{uuid4()}.jpg"
    path = dest_dir / filename
    path.write_bytes(image_bytes)
    return os.fspath(path)
