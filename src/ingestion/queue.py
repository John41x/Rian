"""Shared asyncio ingestion queue with backpressure handling."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.common.schemas import DetectionEventInbound

logger = logging.getLogger(__name__)

INGESTION_QUEUE_MAXSIZE = 500

_ingestion_queue: Optional[asyncio.Queue] = None


def get_ingestion_queue() -> asyncio.Queue:
    global _ingestion_queue
    if _ingestion_queue is None:
        _ingestion_queue = asyncio.Queue(maxsize=INGESTION_QUEUE_MAXSIZE)
    return _ingestion_queue


def set_ingestion_queue(queue: asyncio.Queue) -> None:
    global _ingestion_queue
    _ingestion_queue = queue


async def enqueue_inbound(
    queue: asyncio.Queue, inbound: DetectionEventInbound
) -> None:
    """Enqueue inbound telemetry; when full, drop oldest frames (prefer newer)."""
    while queue.full():
        try:
            queue.get_nowait()
            logger.warning(
                "Ingestion queue at capacity (%s); dropped oldest frame",
                INGESTION_QUEUE_MAXSIZE,
            )
        except asyncio.QueueEmpty:
            break
    await queue.put(inbound)
