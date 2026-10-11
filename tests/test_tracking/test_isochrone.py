"""Dynamic isochrone wavefront tests."""

from __future__ import annotations

import math
from uuid import uuid4

import numpy as np
import pytest
from shapely.geometry import Point, shape

from src.common.schemas import EntityClass
from src.spatial.raster import compute_friction_surface, tobler_seconds_per_meter
from src.tracking.isochrone import (
    build_local_friction_window,
    compute_reachability_geojson,
    extract_reachability_contour,
    solve_travel_time_grid,
)


def test_travel_time_grid_expands_with_elapsed():
    side = 41
    friction = build_local_friction_window(side, EntityClass.PERSON, slope_deg=0.0)
    times = solve_travel_time_grid(friction, cell_size_m=10.0)
    center = times[side // 2, side // 2]
    assert center == 0.0

    contour_short = extract_reachability_contour(
        times, -119.82, 36.745, elapsed_seconds=60.0
    )
    contour_long = extract_reachability_contour(
        times, -119.82, 36.745, elapsed_seconds=600.0
    )
    assert contour_short is not None and contour_long is not None
    assert contour_long.area > contour_short.area


def test_vehicle_isochrone_channels_along_road_column():
    side = 51
    road_col = side // 2
    friction = build_local_friction_window(
        side,
        EntityClass.CAR,
        slope_deg=5.0,
        land_cover="scrub",
        road_corridor_col=road_col,
    )
    times = solve_travel_time_grid(friction, cell_size_m=10.0)
    center_r, center_c = side // 2, side // 2
    on_road = times[center_r + 8, road_col]
    off_road = times[center_r, road_col + 12]
    assert np.isfinite(on_road)
    assert not np.isfinite(off_road) or off_road > on_road * 2


def test_steep_slope_slows_pedestrian_isochrone():
    side = 31
    flat = build_local_friction_window(side, EntityClass.PERSON, slope_deg=0.0)
    steep = build_local_friction_window(side, EntityClass.PERSON, slope_deg=25.0)
    t_flat = solve_travel_time_grid(flat, cell_size_m=10.0)
    t_steep = solve_travel_time_grid(steep, cell_size_m=10.0)
    r, c = side // 2, side // 2 + 5
    assert t_steep[r, c] > t_flat[r, c]


def test_geojson_polygon_rfc7946():
    side = 31
    friction = build_local_friction_window(side, EntityClass.PERSON)
    feature = compute_reachability_geojson(
        friction,
        -119.82,
        36.745,
        elapsed_seconds=300.0,
        entity_class=EntityClass.PERSON,
        track_id=uuid4(),
        track_code="TRK-TEST-001",
    )
    assert feature is not None
    assert feature["type"] == "Feature"
    geom = feature["geometry"]
    assert geom["type"] in ("Polygon", "MultiPolygon")
    poly = shape(geom)
    assert poly.is_valid
    assert poly.contains(Point(-119.82, 36.745)) or poly.distance(Point(-119.82, 36.745)) < 0.01


def test_tobler_ratio_matches_spec_within_tolerance():
    """Hill vs flat travel time ratio aligns with Tobler speeds (pedestrian hike scenario)."""
    imp_flat = float(tobler_seconds_per_meter(0.0))
    imp_hill = float(tobler_seconds_per_meter(25.0))
    measured_ratio = imp_hill / imp_flat
    v_flat = 6.0 * math.exp(-3.5 * 0.05)
    v_hill = 6.0 * math.exp(-3.5 * abs(math.tan(math.radians(25.0)) + 0.05))
    expected_ratio = v_flat / v_hill
    error_pct = abs(measured_ratio - expected_ratio) / expected_ratio * 100.0
    assert error_pct <= 10.0
