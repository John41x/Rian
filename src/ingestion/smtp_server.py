"""Headless aiosmtpd daemon for Reolink-style outbound SMTP snapshot delivery."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from aiosmtpd.controller import Controller
from aiosmtpd.smtp import AuthResult, Envelope, Session, SMTP

from src.common.config import settings
from src.common.schemas import DetectionEventInbound
from src.ingestion.mime_utils import extract_jpeg_attachments, parse_email_message
from src.ingestion.queue import enqueue_inbound, get_ingestion_queue

logger = logging.getLogger(__name__)

SMTP_SUCCESS = "250 Message accepted for delivery"


def camera_authenticator(server: SMTP, session: Session, envelope: Envelope, mechanism: str, auth_data: Any) -> AuthResult:
    """
    Accepts SMTP AUTH credentials from edge cameras (Reolink Go Plus, Vosker, Cuddeback).
    Standalone cellular cameras require a non-empty password in their firmware setup.
    """
    return AuthResult(success=True)


class CameraSMTPHandler:
    """Non-blocking SMTP handler: ACK immediately, enqueue JPEG payloads."""

    def __init__(self, queue: Optional[asyncio.Queue] = None) -> None:
        self._queue = queue

    @property
    def queue(self) -> asyncio.Queue:
        if self._queue is not None:
            return self._queue
        return get_ingestion_queue()

    async def handle_DATA(
        self, server: SMTP, session: Session, envelope: Envelope
    ) -> str:
        sender = envelope.mail_from or ""
        try:
            message = parse_email_message(envelope.content)
            attachments = extract_jpeg_attachments(message)
            if not attachments:
                logger.warning(
                    "SMTP message from %s contained no JPEG attachments; discarding",
                    sender,
                )
                return SMTP_SUCCESS

            received_at = datetime.now(timezone.utc)
            for image_bytes, filename in attachments:
                inbound = DetectionEventInbound(
                    sensor_sender_email=sender,
                    received_at=received_at,
                    image_bytes=image_bytes,
                    image_filename=filename,
                )
                await enqueue_inbound(self.queue, inbound)
        except Exception:
            logger.exception(
                "Corrupted or truncated MIME from %s; discarding", sender
            )
        return SMTP_SUCCESS


class SMTPIngestionController:
    """Wraps aiosmtpd Controller for lifecycle management."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        handler: Optional[CameraSMTPHandler] = None,
        queue: Optional[asyncio.Queue] = None,
    ) -> None:
        self.host = host or settings.SMTP_HOST
        self.port = port or settings.SMTP_PORT
        self.handler = handler or CameraSMTPHandler(queue=queue)
        self._controller: Optional[Controller] = None

    def start(self) -> None:
        if self._controller is not None:
            return
        self._controller = Controller(
            self.handler,
            hostname=self.host,
            port=self.port,
            authenticator=camera_authenticator,
            auth_require_tls=False,
        )
        self._controller.start()
        logger.info("SMTP ingestion daemon listening on %s:%s (AUTH enabled)", self.host, self.port)

    def stop(self) -> None:
        if self._controller is None:
            return
        self._controller.stop()
        self._controller = None
        logger.info("SMTP ingestion daemon stopped")
