"""MIME parsing tests for camera snapshot emails."""

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.ingestion.mime_utils import extract_jpeg_attachments, parse_email_message
from tests.test_ingestion.conftest import build_multipart_email


def test_extract_jpeg_from_multipart_reolink_style(synthetic_jpeg_bytes):
    raw = build_multipart_email(
        "camera-gate-north@reolink.domain", synthetic_jpeg_bytes
    )
    message = parse_email_message(raw)
    attachments = extract_jpeg_attachments(message)

    assert len(attachments) == 1
    data, name = attachments[0]
    assert data == synthetic_jpeg_bytes
    assert name == "snapshot.jpg"


def test_discards_non_jpeg_parts(synthetic_jpeg_bytes):
    msg = MIMEMultipart()
    msg.attach(MIMEText("no image here", "plain"))
    png = MIMEApplication(b"\x89PNG\r\n", _subtype="png")
    png.add_header("Content-Disposition", "attachment", filename="x.png")
    msg.attach(png)

    assert extract_jpeg_attachments(msg) == []

    msg.attach(
        MIMEApplication(synthetic_jpeg_bytes, _subtype="jpeg")
    )
    assert len(extract_jpeg_attachments(msg)) == 1
