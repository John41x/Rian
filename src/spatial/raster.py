"""
Terrain friction and cost-surface generation per specs/02-spatial-database-rasters.md.

Tobler hiking speed: v = 6 * exp(-3.5 * |tan(theta) + 0.05|) km/h
Impedance: seconds per meter = 3.6 / v_kmh
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from src.common.schemas import CLASS_MAX_SLOPE, EntityClass

INF = float("inf")

# Base friction multipliers C_land (spec §3.3)
LAND_COVER_FRICTION: Dict[str, float] = {
    "road_graded": 1.0,
    "two_track": 1.8,
    "pasture": 2.5,
    "scrub": 4.5,
    "timber_rock": 9.0,
    "barrier": INF,
}

# Aliases for network surface_type and NLCD-style tags
_LAND_COVER_ALIASES: Dict[str, str] = {
    "paved": "road_graded",
    "gravel": "road_graded",
    "graded": "road_graded",
    "wash": "two_track",
    "barrier": "barrier",
}

_VEHICLE_PASSABLE_LAND_COVER = frozenset({"road_graded", "two_track"})


def land_cover_multiplier(tag: str) -> float:
    """Return C_land for a surface classification tag."""
    normalized = _LAND_COVER_ALIASES.get(tag, tag)
    return LAND_COVER_FRICTION.get(normalized, LAND_COVER_FRICTION["pasture"])


def tobler_speed_kmh(slope_deg: Union[float, np.ndarray]) -> np.ndarray:
    """
    Tobler's Hiking Function (km/h) for foot travel on slope angle theta (degrees).

    v = 6 * exp(-3.5 * |tan(theta) + 0.05|)
    """
    slope = np.asarray(slope_deg, dtype=np.float64)
    tan_theta = np.tan(np.deg2rad(slope))
    return 6.0 * np.exp(-3.5 * np.abs(tan_theta + 0.05))


def tobler_seconds_per_meter(slope_deg: Union[float, np.ndarray]) -> np.ndarray:
    """Convert Tobler speed (km/h) to traversal impedance in seconds per meter."""
    v_kmh = tobler_speed_kmh(slope_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        impedance = np.where(v_kmh > 0, 3.6 / v_kmh, INF)
    return impedance


def vehicle_slope_threshold_deg(
    travel_mode: str = "vehicle",
    entity_class: Optional[EntityClass] = None,
) -> float:
    """Slope cutoff for wheeled vehicles: truck 15°, car/vehicle default 18°."""
    if entity_class is not None:
        return CLASS_MAX_SLOPE[entity_class]
    mode = travel_mode.lower()
    if mode == "truck":
        return CLASS_MAX_SLOPE[EntityClass.TRUCK]
    if mode in ("car", "vehicle"):
        return CLASS_MAX_SLOPE[EntityClass.CAR]
    return CLASS_MAX_SLOPE[EntityClass.CAR]


def vehicle_slope_impedance_mask(
    slope_deg: Union[float, np.ndarray],
    travel_mode: str = "vehicle",
    entity_class: Optional[EntityClass] = None,
) -> np.ndarray:
    """
    True where slope exceeds vehicle passability threshold (infinite impedance).
    """
    slope = np.asarray(slope_deg, dtype=np.float64)
    threshold = vehicle_slope_threshold_deg(travel_mode, entity_class)
    return np.abs(slope) > threshold


def _land_cover_array(tags: Union[str, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(tags, np.ndarray):
        if tags.shape != shape:
            raise ValueError("land_cover tags array must match slope matrix shape")
        flat = tags.ravel()
        multipliers = np.array([land_cover_multiplier(str(t)) for t in flat], dtype=np.float64)
        return multipliers.reshape(shape)
    multiplier = land_cover_multiplier(tags)
    return np.full(shape, multiplier, dtype=np.float64)


def compute_friction_surface(
    elevation: np.ndarray,
    slope_deg: np.ndarray,
    land_cover: Union[str, np.ndarray] = "pasture",
    travel_mode: str = "foot",
    entity_class: Optional[EntityClass] = None,
) -> np.ndarray:
    """
    Build a 2D friction surface (seconds per meter) from elevation, slope, and land cover.

    ``elevation`` is accepted for API symmetry with future DEM-derived features; slope
    drives Tobler and vehicle cutoff logic today.
    """
    if elevation.shape != slope_deg.shape:
        raise ValueError("elevation and slope matrices must have the same shape")

    shape = slope_deg.shape
    c_land = _land_cover_array(land_cover, shape)
    barrier = c_land >= INF

    mode = travel_mode.lower()
    is_vehicle = mode in ("vehicle", "car", "truck") or entity_class in (
        EntityClass.CAR,
        EntityClass.TRUCK,
    )

    if is_vehicle:
        friction = np.zeros(shape, dtype=np.float64)
        # Off-road / non-vehicle surfaces: infinite impedance (spec acceptance criteria)
        if isinstance(land_cover, np.ndarray):
            passable = np.array(
                [
                    _LAND_COVER_ALIASES.get(str(t), str(t)) in _VEHICLE_PASSABLE_LAND_COVER
                    for t in land_cover.ravel()
                ],
                dtype=bool,
            ).reshape(shape)
        else:
            tag = _LAND_COVER_ALIASES.get(land_cover, land_cover)
            passable = np.full(shape, tag in _VEHICLE_PASSABLE_LAND_COVER)

        # Baseline road travel ~40 km/h unless overridden by land cover tag speeds in network layer
        base_speed_kmh = 40.0 if land_cover_multiplier(str(land_cover) if not isinstance(land_cover, np.ndarray) else "road_graded") <= 1.0 else 25.0
        base_impedance = 3.6 / base_speed_kmh
        friction[:] = base_impedance * c_land
        friction = np.where(passable, friction, INF)
        steep = vehicle_slope_impedance_mask(slope_deg, travel_mode, entity_class)
        friction = np.where(steep, INF, friction)
    else:
        friction = tobler_seconds_per_meter(slope_deg) * c_land

    friction = np.where(barrier, INF, friction)
    return friction


def slope_from_elevation_grid(
    elevation: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """Finite-difference slope in degrees from a DEM window (for local sampling)."""
    dz_dy, dz_dx = np.gradient(elevation, cell_size_m, cell_size_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    return np.rad2deg(slope_rad)
