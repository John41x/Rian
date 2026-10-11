"""Unit tests for mock Reolink SMTP camera emitter."""

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from unittest.mock import MagicMock, patch

from tools.mock_camera import (
    MockCameraEmitter,
    build_reolink_mime_message,
    generate_synthetic_jpeg,
    send_mock_camera_detection,
)


def test_generate_synthetic_jpeg_is_valid():
    data = generate_synthetic_jpeg("truck")
    assert data[:2] == b"\xff\xd8"


def test_mime_structure():
    msg = build_reolink_mime_message(
        "cam-gate-north@ranch.domain",
        generate_synthetic_jpeg("person"),
    )
    assert msg["Subject"] == "Motion detection alert from Reolink Go Plus"
    assert msg["From"] == "cam-gate-north@ranch.domain"
    assert msg["To"] == "ingest@rian.internal"

    plain_parts = [
        p for p in msg.walk() if p.get_content_type() == "text/plain"
    ]
    assert plain_parts
    assert "motion detection" in plain_parts[0].get_content().lower()

    image_parts = [
        p for p in msg.walk() if p.get_content_maintype() == "image"
    ]
    assert image_parts
    assert image_parts[0].get_content_subtype() == "jpeg"
    assert image_parts[0]["Content-Disposition"].startswith('attachment; filename="snapshot.jpg"')


def test_smtp_transmission_local():
    captured: list = []

    class _FakeSMTP:
        def __init__(self, host, port, timeout=10):
            captured.append((host, port))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send_message(self, msg):
            captured.append(msg)

    with patch("tools.mock_camera.smtplib.SMTP", _FakeSMTP):
        send_mock_camera_detection(
            smtp_host="127.0.0.1",
            smtp_port=2525,
            camera_email="cam-gate-north@ranch.domain",
            entity_class="truck",
        )

    assert captured[0] == ("127.0.0.1", 2525)
    msg = captured[1]
    raw = msg.as_bytes()
    parsed = BytesParser(policy=policy.default).parsebytes(raw)
    assert parsed["From"] == "cam-gate-north@ranch.domain"
    attachments = [
        p for p in parsed.walk() if p.get_content_maintype() == "image"
    ]
    assert attachments


def test_mock_camera_emitter_class():
    mock_client = MagicMock()

    with patch("tools.mock_camera.smtplib.SMTP") as smtp_cls:
        smtp_cls.return_value.__enter__.return_value = mock_client
        emitter = MockCameraEmitter(smtp_host="127.0.0.1", smtp_port=2525)
        emitter.emit("cam-gate-east@ranch.domain", "person")
        mock_client.send_message.assert_called_once()
