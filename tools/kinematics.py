"""Shared kinematic helpers for simulation scenarios and validation."""

from __future__ import annotations

import math
from typing import Tuple

from geopy.distance import geodesic

from src.common.schemas import CLASS_MAX_VELOCITY, EntityClass


def elevation_m(x_m: float, y_m: float) -> float:
    """Synthetic DEM height in meters at local estate coordinates (0–10 km)."""
    return (
        400.0
        + 150.0 * math.sin(0.001 * x_m) * math.cos(0.001 * y_m)
        + 80.0
        * math.exp(-((x_m - 5000.0) ** 2 + (y_m - 5000.0) ** 2) / (2.0 * 1200.0 ** 2))
    )


def slope_degrees_grid(elevation: "object") -> "object":
    """Slope angle in degrees from a 2D elevation array (10 m cell size)."""
    import numpy as np

    dz_dy, dz_dx = np.gradient(elevation, 10.0, 10.0)
    return np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))


def tobler_speed_kmh(slope_deg: float) -> float:
    """Tobler hiking speed (km/h) for a given slope angle in degrees."""
    theta_rad = math.radians(slope_deg)
    return 6.0 * math.exp(-3.5 * abs(math.tan(theta_rad) + 0.05))


def kinematic_gate_allows(
    class_a: EntityClass,
    lon_a: float,
    lat_a: float,
    lon_b: float,
    lat_b: float,
    delta_t_seconds: float,
) -> bool:
    """Return True if travel from A to B within delta_t is physically plausible."""
    if delta_t_seconds <= 0:
        return False
    distance_m = geodesic((lat_a, lon_a), (lat_b, lon_b)).meters
    v_max = CLASS_MAX_VELOCITY[class_a]
    return distance_m <= v_max * delta_t_seconds


def road_travel_time_seconds(
    distance_m: float, max_speed_kmh: float = 40.0
) -> float:
    """Traversal time along a gravel ranch road."""
    speed_ms = max_speed_kmh / 3.6
    return distance_m / speed_ms


def estimate_tactical_margin_seconds(
    target_distance_m: float,
    target_speed_kmh: float,
    interceptor_distance_m: float,
    interceptor_speed_kmh: float = 60.0,
) -> float:
    """Positive margin when the interceptor reaches the chokepoint before the target."""
    target_eta = road_travel_time_seconds(target_distance_m, target_speed_kmh)
    interceptor_eta = road_travel_time_seconds(
        interceptor_distance_m, interceptor_speed_kmh
    )
    return target_eta - interceptor_eta
