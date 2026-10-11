"""MIME email parsing utilities for camera snapshot attachments."""

from __future__ import annotations

import logging
from email import policy
from email.message import Message
from email.parser import BytesParser
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

JPEG_CONTENT_TYPES = frozenset({"image/jpeg", "image/jpg", "image/pjpeg"})


def parse_email_message(raw: bytes) -> Message:
    return BytesParser(policy=policy.default).parsebytes(raw)


def extract_jpeg_attachments(message: Message) -> List[Tuple[bytes, Optional[str]]]:
    """
    Walk a MIME message and return (jpeg_bytes, filename) for each JPEG part.
    Accepts image/jpeg content types or attachment disposition with .jpg/.jpeg names.
    """
    found: List[Tuple[bytes, Optional[str]]] = []

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            payload = _jpeg_payload_from_part(part)
            if payload is not None:
                found.append(payload)
    else:
        payload = _jpeg_payload_from_part(message)
        if payload is not None:
            found.append(payload)

    return found


def _jpeg_payload_from_part(part: Message) -> Optional[Tuple[bytes, Optional[str]]]:
    content_type = (part.get_content_type() or "").lower()
    filename = part.get_filename()
    is_jpeg_type = content_type in JPEG_CONTENT_TYPES or "jpeg" in content_type
    is_jpeg_name = bool(
        filename and filename.lower().endswith((".jpg", ".jpeg"))
    )

    try:
        raw = part.get_payload(decode=True)
    except Exception:
        logger.warning("Failed to decode MIME part payload", exc_info=True)
        return None

    if not raw:
        return None

    has_jpeg_magic = _looks_like_jpeg(raw)
    if not is_jpeg_type and not is_jpeg_name and not has_jpeg_magic:
        return None
    if not has_jpeg_magic and not is_jpeg_type:
        return None
    if not has_jpeg_magic and is_jpeg_type:
        logger.warning("MIME part claimed JPEG but magic bytes missing")
        return None

    name = filename or "snapshot.jpg"
    return raw, name


def _looks_like_jpeg(data: bytes) -> bool:
    return len(data) >= 3 and data[0:2] == b"\xff\xd8"
