"""Unit tests for synthetic GIS generation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tools.kinematics import elevation_m
from tools.synthetic_gis import (
    DOMAIN_SIZE_M,
    GRID_SIZE,
    RESOLUTION_M,
    generate_elevation_array,
    generate_slope_array,
    export_dem_geotiff,
    local_m_to_lonlat,
    sw_corner_lonlat,
)


def test_grid_dimensions():
    dem = generate_elevation_array()
    assert dem.shape == (GRID_SIZE, GRID_SIZE)


def test_resolution_and_extent():
    assert GRID_SIZE * RESOLUTION_M == DOMAIN_SIZE_M


def test_elevation_formula_center_hill():
    dem = generate_elevation_array()
    ix, iy = 500, 500
    expected = elevation_m(5000.0, 5000.0)
    assert abs(dem[iy, ix] - expected) < 1e-6


def test_elevation_formula_sample_points():
    for x, y in [(0, 0), (2500, 7500), (9990, 9990)]:
        ix = int(x / RESOLUTION_M)
        iy = int(y / RESOLUTION_M)
        ix = min(ix, GRID_SIZE - 1)
        iy = min(iy, GRID_SIZE - 1)
        assert abs(dem_at(x, y) - elevation_m(x, y)) < 1e-4


def dem_at(x: float, y: float) -> float:
    dem = generate_elevation_array()
    ix = min(int(x / RESOLUTION_M), GRID_SIZE - 1)
    iy = min(int(y / RESOLUTION_M), GRID_SIZE - 1)
    return float(dem[iy, ix])


def test_slope_non_negative_and_bounded():
    slope = generate_slope_array()
    assert slope.min() >= 0.0
    assert slope.max() <= 90.0


def test_slope_matches_manual_gradient():
    dem = generate_elevation_array()
    slope = generate_slope_array(dem)
    dz_dy, dz_dx = np.gradient(dem, RESOLUTION_M, RESOLUTION_M)
    manual = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    np.testing.assert_allclose(slope, manual, rtol=1e-5, atol=1e-5)


def test_local_to_lonlat_monotonic():
    sw = sw_corner_lonlat()
    ne = local_m_to_lonlat(DOMAIN_SIZE_M, DOMAIN_SIZE_M)
    assert ne[0] > sw[0]
    assert ne[1] > sw[1]


def test_export_dem_geotiff(tmp_path: Path):
    out = export_dem_geotiff(output_path=tmp_path / "dem.tif")
    assert out.exists()
    assert out.stat().st_size > 1000
    import rasterio

    with rasterio.open(out) as src:
        assert src.width == GRID_SIZE
        assert src.height == GRID_SIZE
        band = src.read(1)
        assert band.dtype == np.float32
        center_val = band[500, 500]
        assert math.isfinite(center_val)
