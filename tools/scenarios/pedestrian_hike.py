"""
Scenario 2: Pedestrian crosses east fence and climbs a 25° hillside (Tobler decay).
"""

from __future__ import annotations

import logging
from tools.kinematics import tobler_speed_kmh
from tools.mock_camera import MockCameraEmitter
from tools.synthetic_gis import SENSOR_NODES, local_m_to_lonlat

logger = logging.getLogger(__name__)

EAST_FENCE_XY = (10000.0, 5500.0)
HILL_DISTANCE_M = 1500.0
TARGET_SLOPE_DEG = 25.0


def _east_camera_email() -> str:
    for cam in SENSOR_NODES:
        if cam.node_name == "cam-gate-east":
            return cam.sender_email
    raise RuntimeError("cam-gate-east not configured")


def _travel_time_seconds(distance_m: float, slope_deg: float) -> float:
    speed_ms = max(tobler_speed_kmh(slope_deg) / 3.6, 0.1)
    return distance_m / speed_ms


def run(emit_smtp: bool = True) -> dict:
    emitter = MockCameraEmitter.from_settings()
    if emit_smtp:
        try:
            emitter.emit(_east_camera_email(), "person")
        except OSError as exc:
            logger.warning("SMTP emit skipped: %s", exc)

    hill_time = _travel_time_seconds(HILL_DISTANCE_M, TARGET_SLOPE_DEG)
    flat_time = _travel_time_seconds(HILL_DISTANCE_M, 0.0)

    measured_ratio = hill_time / flat_time
    expected_ratio = tobler_speed_kmh(0.0) / tobler_speed_kmh(TARGET_SLOPE_DEG)

    error_pct = abs(measured_ratio - expected_ratio) / expected_ratio * 100.0
    if error_pct > 10.0:
        raise AssertionError(
            f"Isochrone anisotropy error {error_pct:.1f}% > 10% "
            f"(measured_ratio={measured_ratio:.3f}, expected={expected_ratio:.3f})"
        )

    hill_xy = (EAST_FENCE_XY[0] - HILL_DISTANCE_M, EAST_FENCE_XY[1])
    lon, lat = local_m_to_lonlat(*EAST_FENCE_XY)
    return {
        "scenario": "pedestrian_hike",
        "track_code": "TRK-002",
        "east_fence_lonlat": (lon, lat),
        "hill_endpoint_xy": hill_xy,
        "hill_slope_deg": TARGET_SLOPE_DEG,
        "hill_travel_time_s": hill_time,
        "anisotropy_error_pct": error_pct,
    }
