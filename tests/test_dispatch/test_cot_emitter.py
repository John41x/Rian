"""Unit tests for CoT XML builder and broadcaster (spec 06)."""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timedelta, timezone

import pytest

from src.common.schemas import EntityClass, TacticalDirective, TrackState, TrackStatus
from src.dispatch.cot_emitter import CotBroadcaster, CotMessageBuilder


@pytest.fixture
def reference_time() -> datetime:
    return datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_track(reference_time: datetime) -> TrackState:
    return TrackState(
        track_id=uuid.uuid4(),
        track_code="TRK-COT-001",
        object_class=EntityClass.CAR,
        status=TrackStatus.ACTIVE,
        first_detected_at=reference_time,
        last_detected_at=reference_time,
        last_known_point=(-119.819, 36.741),
        estimated_heading_deg=127.5,
        estimated_velocity_ms=8.25,
    )


def _parse_event(xml: str) -> ElementTree.Element:
    return ElementTree.fromstring(xml)


def test_build_track_event_xml_structure(sample_track: TrackState, reference_time: datetime):
    builder = CotMessageBuilder()
    xml = builder.build_track_event(sample_track, stale_seconds=120, now=reference_time)
    root = _parse_event(xml)

    assert root.tag == "event"
    assert root.get("version") == "2.0"
    assert root.get("uid") == "TRK-COT-001"
    assert root.get("type") == "a-u-G-E-V"
    assert root.get("how") == "m-g"
    assert root.get("time") == "2026-10-10T20:00:00.000000Z"
    assert root.get("start") == root.get("time")
    assert root.get("stale") == "2026-10-10T20:02:00.000000Z"

    point = root.find("point")
    assert point is not None
    assert point.get("lat") == "36.741"
    assert point.get("lon") == "-119.819"
    assert point.get("hae") == "0.0"
    assert point.get("ce") == "15.0"
    assert point.get("le") == "5.0"


def test_build_track_event_speed_and_course(sample_track: TrackState, reference_time: datetime):
    builder = CotMessageBuilder()
    xml = builder.build_track_event(sample_track, now=reference_time)
    root = _parse_event(xml)
    detail = root.find("detail")
    assert detail is not None
    track_el = detail.find("track")
    assert track_el is not None
    assert track_el.get("speed") == "8.25"
    assert track_el.get("course") == "127.5"
    contact = detail.find("contact")
    assert contact is not None
    assert contact.get("callsign") == "TRK-COT-001"


def test_entity_class_cot_types(reference_time: datetime):
    builder = CotMessageBuilder()
    base = {
        "track_id": uuid.uuid4(),
        "track_code": "T",
        "status": TrackStatus.ACTIVE,
        "first_detected_at": reference_time,
        "last_detected_at": reference_time,
        "last_known_point": (0.0, 0.0),
    }
    cases = [
        (EntityClass.PERSON, "a-u-G-U-C"),
        (EntityClass.TRUCK, "a-u-G-E-V"),
        (EntityClass.COW, "a-n-G"),
    ]
    for entity_class, expected_type in cases:
        track = TrackState(object_class=entity_class, **base)
        root = _parse_event(builder.build_track_event(track, now=reference_time))
        assert root.get("type") == expected_type


def test_stale_timestamp_window(sample_track: TrackState, reference_time: datetime):
    builder = CotMessageBuilder()
    xml = builder.build_track_event(sample_track, stale_seconds=90, now=reference_time)
    root = _parse_event(xml)
    stale = datetime.strptime(root.get("stale"), "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    start = datetime.strptime(root.get("start"), "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    assert (stale - start) == timedelta(seconds=90)


def test_build_directive_event(sample_directive: TacticalDirective, reference_time: datetime):
    builder = CotMessageBuilder()
    xml = builder.build_directive_event(
        sample_directive, stale_seconds=300, now=reference_time
    )
    root = _parse_event(xml)

    assert root.get("version") == "2.0"
    assert root.get("uid") == str(sample_directive.directive_id)
    assert root.get("type") == "b-m-p-s-p-i"
    assert root.get("stale") == "2026-10-10T20:05:00.000000Z"

    point = root.find("point")
    assert point is not None
    assert point.get("lat") == "36.742"
    assert point.get("lon") == "-119.818"

    detail = root.find("detail")
    assert detail is not None
    assert detail.find("contact").get("callsign") == "TRK-TEST-001"
    remarks = detail.find("remarks")
    assert remarks is not None
    assert remarks.text == "INTERCEPT AT South Gate"


@pytest.mark.asyncio
async def test_broadcaster_mock_mode(sample_track: TrackState, sample_directive: TacticalDirective):
    broadcaster = CotBroadcaster(mock_mode=True)
    await broadcaster.send_track(sample_track)
    await broadcaster.send_directive(sample_directive)

    assert len(broadcaster.mock_sent) == 2
    track_root = _parse_event(broadcaster.mock_sent[0])
    assert track_root.get("uid") == sample_track.track_code
    directive_root = _parse_event(broadcaster.mock_sent[1])
    assert directive_root.get("type") == "b-m-p-s-p-i"
    assert "INTERCEPT AT South Gate" in broadcaster.mock_sent[1]
