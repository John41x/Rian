"""Ingestion queue backpressure behavior."""

import asyncio
from datetime import datetime, timezone

import pytest

from src.common.schemas import DetectionEventInbound
from src.ingestion.queue import enqueue_inbound


@pytest.mark.asyncio
async def test_enqueue_drops_oldest_when_full():
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    ts = datetime.now(timezone.utc)

    async def make_item(n: int) -> DetectionEventInbound:
        return DetectionEventInbound(
            sensor_sender_email=f"cam{n}@test",
            received_at=ts,
            image_bytes=f"{n}".encode(),
        )

    await enqueue_inbound(queue, await make_item(1))
    await enqueue_inbound(queue, await make_item(2))
    await enqueue_inbound(queue, await make_item(3))

    first = queue.get_nowait()
    second = queue.get_nowait()
    assert first.image_bytes == b"2"
    assert second.image_bytes == b"3"
