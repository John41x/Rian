"""
Mock Reolink Go Plus SMTP snapshot emitter for simulation and CI.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import smtplib
from PIL import Image, ImageDraw

from src.common.config import settings

DEFAULT_SUBJECT = "Motion detection alert from Reolink Go Plus"
DEFAULT_RECIPIENT = "ingest@rian.internal"
SAMPLE_IMAGE_DIR = Path("tests/data")


def _ensure_sample_dir() -> Path:
    SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return SAMPLE_IMAGE_DIR


def generate_synthetic_jpeg(
    entity_class: str = "person",
    label: str | None = None,
    width: int = 640,
    height: int = 480,
) -> bytes:
    """Create a minimal JPEG with an on-image annotation for test ingestion."""
    img = Image.new("RGB", (width, height), color=(40, 44, 52))
    draw = ImageDraw.Draw(img)
    text = label or f"SIM:{entity_class}"
    draw.rectangle([40, 40, width - 40, height - 40], outline=(220, 180, 60), width=3)
    draw.text((60, 60), text, fill=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def write_sample_image(
    entity_class: str,
    path: Path | None = None,
) -> Path:
    """Persist a sample JPEG under tests/data for fixture paths."""
    out = path or _ensure_sample_dir() / f"sample_{entity_class}.jpg"
    out.write_bytes(generate_synthetic_jpeg(entity_class))
    return out


def build_reolink_mime_message(
    camera_email: str,
    image_bytes: bytes,
    *,
    subject: str = DEFAULT_SUBJECT,
    body_text: str | None = None,
    filename: str = "snapshot.jpg",
    recipient: str = DEFAULT_RECIPIENT,
) -> EmailMessage:
    """Craft multipart MIME matching Reolink cellular camera emails."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = camera_email
    msg["To"] = recipient
    gate = camera_email.split("@")[0].replace("cam-", "").replace("-", " ").title()
    msg.set_content(body_text or f"A motion detection event has occurred at {gate}.")
    msg.add_attachment(
        image_bytes,
        maintype="image",
        subtype="jpeg",
        filename=filename,
    )
    return msg


def send_mock_camera_detection(
    smtp_host: str = "127.0.0.1",
    smtp_port: int = 2525,
    camera_email: str = "cam-gate-north@ranch.domain",
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    entity_class: str = "person",
    subject: str = DEFAULT_SUBJECT,
    recipient: str = DEFAULT_RECIPIENT,
) -> EmailMessage:
    """Send an authentic Reolink-style snapshot email via SMTP."""
    if image_bytes is None:
        if image_path:
            image_bytes = Path(image_path).read_bytes()
        else:
            image_bytes = generate_synthetic_jpeg(entity_class)

    msg = build_reolink_mime_message(
        camera_email,
        image_bytes,
        subject=subject,
        recipient=recipient,
    )
    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as client:
        client.send_message(msg)
    return msg


@dataclass
class MockCameraEmitter:
    """Configurable mock camera client for scenario runners."""

    smtp_host: str = "127.0.0.1"
    smtp_port: int = 2525
    recipient: str = DEFAULT_RECIPIENT

    def __post_init__(self) -> None:
        if self.smtp_host == "0.0.0.0":
            self.smtp_host = "127.0.0.1"

    @classmethod
    def from_settings(cls) -> "MockCameraEmitter":
        host = settings.SMTP_HOST
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return cls(smtp_host=host, smtp_port=settings.SMTP_PORT)

    def emit(
        self,
        camera_email: str,
        entity_class: str,
        *,
        image_path: Optional[str] = None,
        subject: str = DEFAULT_SUBJECT,
    ) -> EmailMessage:
        return send_mock_camera_detection(
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            camera_email=camera_email,
            image_path=image_path,
            entity_class=entity_class,
            subject=subject,
            recipient=self.recipient,
        )
