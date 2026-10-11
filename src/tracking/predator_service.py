"""Predator threat proximity engine and deterrent state (spec 17)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from geopy.distance import geodesic
from pydantic import BaseModel

_THREAT_RANK = {
    "NOMINAL": 0,
    "PROXIMITY_ADVISORY": 1,
    "HERD_THREAT_WARNING": 2,
    "CRITICAL_INCURSION": 3,
}


class PredatorThreatLevel(str, Enum):
    NOMINAL = "NOMINAL"
    PROXIMITY_ADVISORY = "PROXIMITY_ADVISORY"
    HERD_THREAT_WARNING = "HERD_THREAT_WARNING"
    CRITICAL_INCURSION = "CRITICAL_INCURSION"


class PredatorThreatEvent(BaseModel):
    threat_id: UUID
    predator_track_id: UUID
    predator_track_code: str
    predator_class: str
    nearest_livestock_id: UUID
    nearest_livestock_code: str
    distance_meters: float
    threat_level: PredatorThreatLevel
    headline: str
    deterrent_activated: bool = False
    deterrent_type: Optional[str] = None
    timestamp: datetime


class PredatorProximitySummary(BaseModel):
    active_threats_count: int
    highest_threat_level: PredatorThreatLevel
    threats: List[PredatorThreatEvent]
    nursery_zone_breaches: int = 0


class PredatorThreatService:
    PREDATOR_CLASSES = {"dog", "coyote", "wolf", "bear"}
    LIVESTOCK_CLASSES = {"cow", "horse", "sheep", "calf"}

    def __init__(self) -> None:
        self._threat_by_id: Dict[UUID, PredatorThreatEvent] = {}
        self._deterrent_by_predator: Dict[UUID, PredatorThreatEvent] = {}

    @staticmethod
    def _track_id(track: Dict[str, Any]) -> Optional[UUID]:
        raw = track.get("track_id") or track.get("id")
        if raw is None:
            return None
        return UUID(str(raw))

    @staticmethod
    def _track_point(track: Dict[str, Any]) -> Optional[tuple[float, float]]:
        point = track.get("last_known_point")
        if not point or len(point) < 2:
            return None
        return float(point[0]), float(point[1])

    @staticmethod
    def _is_active(track: Dict[str, Any]) -> bool:
        status = track.get("status") or track.get("current_status")
        if status is None:
            return True
        return str(status).upper() == "ACTIVE"

    @classmethod
    def _classify_threat_level(cls, distance_m: float) -> PredatorThreatLevel:
        if distance_m < 200:
            return PredatorThreatLevel.CRITICAL_INCURSION
        if distance_m < 500:
            return PredatorThreatLevel.HERD_THREAT_WARNING
        if distance_m < 1000:
            return PredatorThreatLevel.PROXIMITY_ADVISORY
        return PredatorThreatLevel.NOMINAL

    @classmethod
    def distance_between_tracks(
        cls,
        predator: Dict[str, Any],
        livestock: Dict[str, Any],
    ) -> Optional[float]:
        p_pt = cls._track_point(predator)
        l_pt = cls._track_point(livestock)
        if p_pt is None or l_pt is None:
            return None
        return float(geodesic((p_pt[1], p_pt[0]), (l_pt[1], l_pt[0])).meters)

    def evaluate_threats(self, tracks: List[Dict[str, Any]]) -> PredatorProximitySummary:
        now = datetime.now(timezone.utc)
        predators: List[Dict[str, Any]] = []
        livestock: List[Dict[str, Any]] = []

        for track in tracks:
            if not self._is_active(track):
                continue
            obj = str(track.get("object_class", "")).lower()
            if obj in self.PREDATOR_CLASSES:
                predators.append(track)
            elif obj in self.LIVESTOCK_CLASSES:
                livestock.append(track)

        events: List[PredatorThreatEvent] = []
        for predator in predators:
            if not livestock:
                break
            predator_id = self._track_id(predator)
            if predator_id is None:
                continue
            predator_code = str(predator.get("track_code") or "PREDATOR")
            predator_class = str(predator.get("object_class", "coyote")).lower()

            nearest: Optional[Dict[str, Any]] = None
            nearest_dist: Optional[float] = None
            for herd in livestock:
                dist = self.distance_between_tracks(predator, herd)
                if dist is None:
                    continue
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest = herd

            if nearest is None or nearest_dist is None:
                continue

            livestock_id = self._track_id(nearest)
            if livestock_id is None:
                continue
            livestock_code = str(nearest.get("track_code") or "HERD")

            level = self._classify_threat_level(nearest_dist)
            headline = (
                f"{predator_code} ({predator_class}) closing on "
                f"{livestock_code} ({nearest_dist:.0f}m separation)"
            )

            prior = self._deterrent_by_predator.get(predator_id)
            threat_id = prior.threat_id if prior else uuid4()
            deterrent_activated = prior.deterrent_activated if prior else False
            deterrent_type = prior.deterrent_type if prior else None
            timestamp = prior.timestamp if prior and prior.deterrent_activated else now

            event = PredatorThreatEvent(
                threat_id=threat_id,
                predator_track_id=predator_id,
                predator_track_code=predator_code,
                predator_class=predator_class,
                nearest_livestock_id=livestock_id,
                nearest_livestock_code=livestock_code,
                distance_meters=nearest_dist,
                threat_level=level,
                headline=headline,
                deterrent_activated=deterrent_activated,
                deterrent_type=deterrent_type,
                timestamp=timestamp,
            )
            if level != PredatorThreatLevel.NOMINAL:
                events.append(event)
            self._threat_by_id[event.threat_id] = event

        active = [e for e in events if e.threat_level != PredatorThreatLevel.NOMINAL]
        if active:
            highest = max(
                active,
                key=lambda e: _THREAT_RANK.get(e.threat_level.value, 0),
            ).threat_level
        else:
            highest = PredatorThreatLevel.NOMINAL

        return PredatorProximitySummary(
            active_threats_count=len(active),
            highest_threat_level=highest,
            threats=active,
            nursery_zone_breaches=0,
        )

    def trigger_deterrent(
        self,
        threat_id: UUID,
        deterrent_type: str = "acoustic_siren",
    ) -> Optional[PredatorThreatEvent]:
        event = self._threat_by_id.get(threat_id)
        if event is None:
            return None
        updated = event.model_copy(
            update={
                "deterrent_activated": True,
                "deterrent_type": deterrent_type,
                "timestamp": datetime.now(timezone.utc),
            }
        )
        self._threat_by_id[threat_id] = updated
        self._deterrent_by_predator[updated.predator_track_id] = updated
        return updated

    def reset(self) -> None:
        self._threat_by_id.clear()
        self._deterrent_by_predator.clear()


_predator_service: Optional[PredatorThreatService] = None


def get_predator_service() -> PredatorThreatService:
    global _predator_service
    if _predator_service is None:
        _predator_service = PredatorThreatService()
    return _predator_service


def reset_predator_service() -> None:
    """Clear singleton state (tests)."""
    global _predator_service
    if _predator_service is not None:
        _predator_service.reset()
    _predator_service = None
