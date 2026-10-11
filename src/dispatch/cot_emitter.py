"""
Cursor-on-Target (CoT) XML builder and UDP broadcaster for ATAK / TAK clients (spec 06).
"""

from __future__ import annotations

import asyncio
import logging
import os
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.common.schemas import EntityClass, TacticalDirective, TrackState

logger = logging.getLogger(__name__)

_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_DEFAULT_HOW = "m-g"
_POINT_HAE = "0.0"
_POINT_CE = "15.0"
_POINT_LE = "5.0"

_DIRECTIVE_COT_TYPE = "b-m-p-s-p-i"

_ENTITY_COT_TYPE: dict[EntityClass, str] = {
    EntityClass.CAR: "a-u-G-E-V",
    EntityClass.TRUCK: "a-u-G-E-V",
    EntityClass.PERSON: "a-u-G-U-C",
    EntityClass.COW: "a-n-G",
    EntityClass.HORSE: "a-n-G",
    EntityClass.DOG: "a-n-G",
}


def _utc_now(now: Optional[datetime] = None) -> datetime:
    if now is not None:
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _format_cot_time(dt: datetime) -> str:
    utc = _utc_now(dt)
    return utc.strftime(_DATETIME_FORMAT)


def _cot_type_for_entity(entity_class: EntityClass) -> str:
    try:
        return _ENTITY_COT_TYPE[entity_class]
    except KeyError:
        return "a-u-G"


class CotMessageBuilder:
    """Builds CoT v2.0 XML events from Rian track and directive contracts."""

    def build_track_event(
        self,
        track: TrackState,
        stale_seconds: int = 120,
        now: Optional[datetime] = None,
    ) -> str:
        reference = _utc_now(now)
        stale = reference + timedelta(seconds=stale_seconds)
        lon, lat = track.last_known_point
        speed = track.estimated_velocity_ms if track.estimated_velocity_ms is not None else 0.0
        course = (
            track.estimated_heading_deg if track.estimated_heading_deg is not None else 0.0
        )

        event = ElementTree.Element("event")
        event.set("version", "2.0")
        event.set("uid", track.track_code)
        event.set("type", _cot_type_for_entity(track.object_class))
        event.set("how", _DEFAULT_HOW)
        time_str = _format_cot_time(reference)
        event.set("time", time_str)
        event.set("start", time_str)
        event.set("stale", _format_cot_time(stale))

        point = ElementTree.SubElement(event, "point")
        point.set("lat", str(lat))
        point.set("lon", str(lon))
        point.set("hae", _POINT_HAE)
        point.set("ce", _POINT_CE)
        point.set("le", _POINT_LE)

        detail = ElementTree.SubElement(event, "detail")
        contact = ElementTree.SubElement(detail, "contact")
        contact.set("callsign", track.track_code)
        track_el = ElementTree.SubElement(detail, "track")
        track_el.set("speed", str(speed))
        track_el.set("course", str(course))
        remarks = ElementTree.SubElement(detail, "remarks")
        remarks.text = f"Rian track {track.track_code} ({track.object_class.value})"

        return ElementTree.tostring(event, encoding="unicode")

    def build_directive_event(
        self,
        directive: TacticalDirective,
        stale_seconds: int = 300,
        now: Optional[datetime] = None,
    ) -> str:
        reference = _utc_now(now)
        stale = reference + timedelta(seconds=stale_seconds)
        lon, lat = directive.chokepoint_location

        event = ElementTree.Element("event")
        event.set("version", "2.0")
        event.set("uid", str(directive.directive_id))
        event.set("type", _DIRECTIVE_COT_TYPE)
        event.set("how", _DEFAULT_HOW)
        time_str = _format_cot_time(reference)
        event.set("time", time_str)
        event.set("start", time_str)
        event.set("stale", _format_cot_time(stale))

        point = ElementTree.SubElement(event, "point")
        point.set("lat", str(lat))
        point.set("lon", str(lon))
        point.set("hae", _POINT_HAE)
        point.set("ce", _POINT_CE)
        point.set("le", _POINT_LE)

        detail = ElementTree.SubElement(event, "detail")
        contact = ElementTree.SubElement(detail, "contact")
        contact.set("callsign", directive.track_code)
        track_el = ElementTree.SubElement(detail, "track")
        track_el.set("speed", "0.0")
        track_el.set("course", "0.0")
        remarks = ElementTree.SubElement(detail, "remarks")
        remarks.text = directive.action_message

        return ElementTree.tostring(event, encoding="unicode")


class CotBroadcaster:
    """Sends CoT XML over UDP; mock mode records payloads for tests."""

    def __init__(
        self,
        broadcast_ip: Optional[str] = None,
        port: Optional[int] = None,
        mock_mode: bool = False,
        builder: Optional[CotMessageBuilder] = None,
    ) -> None:
        self.broadcast_ip = broadcast_ip or os.environ.get("COT_BROADCAST_IP", "239.2.3.1")
        self.port = port or int(os.environ.get("COT_PORT", "6969"))
        self.mock_mode = mock_mode
        self.builder = builder or CotMessageBuilder()
        self._mock_sent: List[str] = []
        self._transport: Optional[asyncio.DatagramTransport] = None

    @property
    def mock_sent(self) -> List[str]:
        return list(self._mock_sent)

    async def send_track(self, track: TrackState) -> None:
        xml = self.builder.build_track_event(track)
        await self._send(xml)

    async def send_directive(self, directive: TacticalDirective) -> None:
        xml = self.builder.build_directive_event(directive)
        await self._send(xml)

    async def _send(self, payload: str) -> None:
        if self.mock_mode:
            self._mock_sent.append(payload)
            logger.debug("CoT mock send (%d bytes)", len(payload))
            return

        data = payload.encode("utf-8")
        loop = asyncio.get_running_loop()
        if self._transport is None:
            self._transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=(self.broadcast_ip, self.port),
            )
        self._transport.sendto(data)

    async def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
