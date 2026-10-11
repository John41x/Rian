"""Kinematic spatiotemporal gating envelope (spec §3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from geopy.distance import geodesic

from src.common.schemas import CLASS_MAX_VELOCITY, EntityClass


@dataclass(frozen=True)
class GatingResult:
    reachable: bool
    distance_m: float
    implied_speed_ms: float
    max_speed_ms: float
    delta_t_seconds: float


def evaluate_spatiotemporal_gate(
    track_lon: float,
    track_lat: float,
    track_time: datetime,
    detection_lon: float,
    detection_lat: float,
    detection_time: datetime,
    entity_class: EntityClass,
) -> GatingResult:
    """
    Evaluate d_jk <= v_max(c_k) * delta_t for association candidacy.

    ``entity_class`` is the incoming detection class c_k (spec §3.1).
    """
    delta_t = (detection_time - track_time).total_seconds()
    if delta_t <= 0:
        return GatingResult(
            reachable=False,
            distance_m=0.0,
            implied_speed_ms=float("inf"),
            max_speed_ms=CLASS_MAX_VELOCITY[entity_class],
            delta_t_seconds=delta_t,
        )

    distance_m = geodesic((track_lat, track_lon), (detection_lat, detection_lon)).meters
    max_speed_ms = CLASS_MAX_VELOCITY[entity_class]
    implied_speed_ms = distance_m / delta_t
    reachable = distance_m <= max_speed_ms * delta_t

    return GatingResult(
        reachable=reachable,
        distance_m=distance_m,
        implied_speed_ms=implied_speed_ms,
        max_speed_ms=max_speed_ms,
        delta_t_seconds=delta_t,
    )
