"""HTTP webhook fallback for managed inbound email parsers (SendGrid / AWS SES)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from src.common.schemas import DetectionEventInbound
from src.ingestion.mime_utils import extract_jpeg_attachments
from src.ingestion.queue import enqueue_inbound, get_ingestion_queue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telemetry"])


@router.post("/api/v1/telemetry/smtp-webhook")
async def smtp_webhook(
    request: Request,
    from_email: Optional[str] = Form(None, alias="from"),
    email_raw: Optional[str] = Form(None, alias="email"),
    attachment: Optional[UploadFile] = File(None),
) -> Dict[str, str]:
    """
    Accept SendGrid/AWS SES style inbound parse payloads and enqueue JPEG snapshots.
    """
    queue = get_ingestion_queue()
    received_at = datetime.now(timezone.utc)
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        await _enqueue_from_json(queue, body, received_at)
        return {"status": "accepted"}

    sender = from_email or ""
    if email_raw:
        sender = sender or _parse_sender_from_raw_email(email_raw)
        attachments = extract_jpeg_attachments(
            BytesParser(policy=policy.default).parsestr(email_raw)
        )
        for image_bytes, filename in attachments:
            inbound = DetectionEventInbound(
                sensor_sender_email=sender,
                received_at=received_at,
                image_bytes=image_bytes,
                image_filename=filename,
            )
            await enqueue_inbound(queue, inbound)
        return {"status": "accepted"}

    if attachment is not None:
        data = await attachment.read()
        if data:
            inbound = DetectionEventInbound(
                sensor_sender_email=sender or "unknown@webhook",
                received_at=received_at,
                image_bytes=data,
                image_filename=attachment.filename or "snapshot.jpg",
            )
            await enqueue_inbound(queue, inbound)

    return {"status": "accepted"}


async def _enqueue_from_json(
    queue, body: Dict[str, Any], received_at: datetime
) -> None:
    sender = str(body.get("from") or body.get("sender") or "")
    raw_email = body.get("email")
    if isinstance(raw_email, str) and raw_email.strip():
        sender = sender or _parse_sender_from_raw_email(raw_email)
        attachments = extract_jpeg_attachments(
            BytesParser(policy=policy.default).parsestr(raw_email)
        )
        for image_bytes, filename in attachments:
            inbound = DetectionEventInbound(
                sensor_sender_email=sender,
                received_at=received_at,
                image_bytes=image_bytes,
                image_filename=filename,
            )
            await enqueue_inbound(queue, inbound)
        return

    attachments = body.get("attachments") or []
    if isinstance(attachments, list):
        for item in attachments:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("data")
            if isinstance(content, str):
                import base64

                image_bytes = base64.b64decode(content)
            elif isinstance(content, (bytes, bytearray)):
                image_bytes = bytes(content)
            else:
                continue
            inbound = DetectionEventInbound(
                sensor_sender_email=sender or "unknown@webhook",
                received_at=received_at,
                image_bytes=image_bytes,
                image_filename=item.get("filename") or "snapshot.jpg",
            )
            await enqueue_inbound(queue, inbound)


def _parse_sender_from_raw_email(raw: str) -> str:
    from email.utils import parseaddr

    message = BytesParser(policy=policy.default).parsestr(raw)
    return parseaddr(message.get("From", ""))[1] or ""
