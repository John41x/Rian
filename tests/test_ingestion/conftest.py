"""Shared fixtures for ingestion unit tests."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest
from PIL import Image


@pytest.fixture
def synthetic_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (64, 48), color=(120, 80, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_received_at() -> datetime:
    return datetime(2026, 10, 10, 21, 30, 0, tzinfo=timezone.utc)


def build_multipart_email(
    sender: str,
    jpeg_bytes: bytes,
    filename: str = "snapshot.jpg",
) -> bytes:
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = "ingest@rian.local"
    msg["Subject"] = "Reolink Motion Alert"
    msg.attach(MIMEText("Motion detected", "plain"))
    attachment = MIMEApplication(jpeg_bytes, _subtype="jpeg")
    attachment.add_header(
        "Content-Disposition", "attachment", filename=filename
    )
    msg.attach(attachment)
    return msg.as_bytes()
