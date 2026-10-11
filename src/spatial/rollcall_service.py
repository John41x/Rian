"""
Automated daily herd inventory roll-call and CSV export (spec 19).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel

from src.common.config import settings
from src.common.domains import get_domain_profile
from src.ingestion.vlm_inspector import get_vlm_inspector
from src.spatial.zones import ZoneManager


class RollCallItem(BaseModel):
    track_id: UUID
    track_code: str
    entity_class: str
    current_zone: str
    last_known_point: List[float]
    last_seen_timestamp: datetime
    last_camera_name: str
    hours_since_last_seen: float
    health_status: str
    vlm_summary: Optional[str] = None


class HerdRollCallSummary(BaseModel):
    generated_at_utc: datetime
    profile_type: str
    total_tracked_count: int
    accounted_count: int
    missing_count: int
    zone_distribution: Dict[str, int]
    items: List[RollCallItem]


def _parse_timestamp(value: Any, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value).strip()
    if not text:
        return fallback
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _health_status_for_hours(hours: float) -> str:
    if hours < 4.0:
        return "VERIFIED_ACTIVE"
    if hours <= 24.0:
        return "OVERDUE_CHECK"
    return "MISSING_CRITICAL"


def _vlm_summary_for_track(track_id: UUID) -> Optional[str]:
    inspector = get_vlm_inspector()
    record = inspector.get_track_snapshot(track_id)
    if record is None:
        return None
    report = inspector._local_forensic_report(record)
    parts = [report.visual_description.strip()]
    if report.risk_assessment:
        parts.append(report.risk_assessment.strip())
    return " // ".join(p for p in parts if p)


class RollCallService:
    """Builds pasture census roll-call summaries from COP track state."""

    def __init__(self, zone_manager: Optional[ZoneManager] = None) -> None:
        self._zone_manager = zone_manager

    def _manager(self) -> ZoneManager:
        if self._zone_manager is not None:
            return self._zone_manager
        return ZoneManager(settings.OPERATION_PROFILE)

    def generate_rollcall(
        self,
        cop_state: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> HerdRollCallSummary:
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)

        profile = get_domain_profile(settings.OPERATION_PROFILE)
        zone_manager = self._manager()
        items: List[RollCallItem] = []
        zone_distribution: Dict[str, int] = {}

        for track in cop_state.get("tracks") or []:
            track_id_raw = track.get("track_id") or track.get("id")
            if not track_id_raw:
                continue
            track_id = UUID(str(track_id_raw))
            track_code = str(track.get("track_code") or track_id)
            entity_class = str(
                track.get("entity_class")
                or track.get("object_class")
                or "unknown"
            )

            point = track.get("last_known_point")
            if point and len(point) >= 2:
                lon, lat = float(point[0]), float(point[1])
            else:
                lon_val = track.get("lon")
                lat_val = track.get("lat")
                if lon_val is None or lat_val is None:
                    lon, lat = 0.0, 0.0
                else:
                    lon, lat = float(lon_val), float(lat_val)

            zone_feature = zone_manager.find_zone_for_point(lon, lat)
            if zone_feature:
                zone_name = str(
                    zone_feature.get("properties", {}).get("name")
                    or zone_feature.get("name")
                    or "Unknown Zone"
                )
            else:
                zone_name = "Unassigned"

            last_seen = _parse_timestamp(track.get("last_detected_at"), reference)
            hours = max(
                0.0,
                (reference - last_seen).total_seconds() / 3600.0,
            )
            status = _health_status_for_hours(hours)
            camera = str(track.get("last_camera_name") or track.get("camera_name") or "")

            item = RollCallItem(
                track_id=track_id,
                track_code=track_code,
                entity_class=entity_class,
                current_zone=zone_name,
                last_known_point=[lon, lat],
                last_seen_timestamp=last_seen,
                last_camera_name=camera,
                hours_since_last_seen=round(hours, 2),
                health_status=status,
                vlm_summary=_vlm_summary_for_track(track_id),
            )
            items.append(item)
            zone_distribution[zone_name] = zone_distribution.get(zone_name, 0) + 1

        missing_count = sum(1 for i in items if i.health_status == "MISSING_CRITICAL")
        accounted_count = len(items) - missing_count

        return HerdRollCallSummary(
            generated_at_utc=reference,
            profile_type=profile.profile_type,
            total_tracked_count=len(items),
            accounted_count=accounted_count,
            missing_count=missing_count,
            zone_distribution=zone_distribution,
            items=items,
        )

    def format_csv(self, summary: HerdRollCallSummary) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            [
                "Track Code",
                "Class",
                "Current Pasture/Zone",
                "Last Seen (UTC)",
                "Hours Ago",
                "Last Camera",
                "Health Status",
                "VLM Summary",
            ]
        )
        for item in summary.items:
            last_seen = item.last_seen_timestamp
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            last_seen_str = last_seen.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            writer.writerow(
                [
                    item.track_code,
                    item.entity_class,
                    item.current_zone,
                    last_seen_str,
                    f"{item.hours_since_last_seen:.2f}",
                    item.last_camera_name,
                    item.health_status,
                    item.vlm_summary or "",
                ]
            )
        return buffer.getvalue()


_rollcall_singleton: Optional[RollCallService] = None


def get_rollcall_service() -> RollCallService:
    global _rollcall_singleton
    if _rollcall_singleton is None:
        _rollcall_singleton = RollCallService()
    return _rollcall_singleton


def reset_rollcall_service_for_tests(zone_manager: Optional[ZoneManager] = None) -> RollCallService:
    global _rollcall_singleton
    _rollcall_singleton = RollCallService(zone_manager=zone_manager)
    return _rollcall_singleton
