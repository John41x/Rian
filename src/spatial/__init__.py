"""Spatial database, raster friction surfaces, and pgRouting services."""

from src.spatial.raster import (
    LAND_COVER_FRICTION,
    compute_friction_surface,
    land_cover_multiplier,
    tobler_seconds_per_meter,
    tobler_speed_kmh,
    vehicle_slope_impedance_mask,
)
from src.spatial.map_match import CandidateEdge, RoadMapMatcher
from src.spatial.routing import RouteResult, find_shortest_route
from src.spatial.service import SpatialService

__all__ = [
    "CandidateEdge",
    "RoadMapMatcher",
    "LAND_COVER_FRICTION",
    "RouteResult",
    "SpatialService",
    "compute_friction_surface",
    "find_shortest_route",
    "land_cover_multiplier",
    "tobler_seconds_per_meter",
    "tobler_speed_kmh",
    "vehicle_slope_impedance_mask",
]
