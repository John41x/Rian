"""SMTP handler ACK and queueing behavior."""

import asyncio
import pytest
from aiosmtpd.smtp import Envelope

from src.common.schemas import DetectionEventInbound
from src.ingestion.smtp_server import CameraSMTPHandler, SMTP_SUCCESS
from tests.test_ingestion.conftest import build_multipart_email


class _FakeSession:
    pass


class _FakeServer:
    pass


@pytest.mark.asyncio
async def test_smtp_handler_returns_250_and_enqueues_jpeg(synthetic_jpeg_bytes):
    queue: asyncio.Queue = asyncio.Queue()
    handler = CameraSMTPHandler(queue=queue)
    sender = "camera@reolink.domain"
    raw = build_multipart_email(sender, synthetic_jpeg_bytes)
    envelope = Envelope()
    envelope.mail_from = sender
    envelope.content = raw

    response = await handler.handle_DATA(_FakeServer(), _FakeSession(), envelope)

    assert response == SMTP_SUCCESS
    assert queue.qsize() == 1
    inbound: DetectionEventInbound = queue.get_nowait()
    assert inbound.sensor_sender_email == sender
    assert inbound.image_bytes == synthetic_jpeg_bytes


@pytest.mark.asyncio
async def test_smtp_handler_acks_without_image(synthetic_jpeg_bytes):
    queue: asyncio.Queue = asyncio.Queue()
    handler = CameraSMTPHandler(queue=queue)
    envelope = Envelope()
    envelope.mail_from = "camera@reolink.domain"
    envelope.content = b"Subject: test\r\n\r\nNo attachment\r\n"

    response = await handler.handle_DATA(_FakeServer(), _FakeSession(), envelope)

    assert response == SMTP_SUCCESS
    assert queue.empty()
