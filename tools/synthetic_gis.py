"""
Synthetic GIS generator: 10 km × 10 km ranch DEM, road network, and DB seeding.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import psycopg
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.common.config import settings
from tools.kinematics import elevation_m, slope_degrees_grid

logger = logging.getLogger(__name__)

GRID_SIZE = 1000
RESOLUTION_M = 10.0
DOMAIN_SIZE_M = 10_000.0

# Ranch centroid (Central California solar / cattle ranch testbed)
RANCH_CENTER_LON = -119.8200
RANCH_CENTER_LAT = 36.7450

DEM_OUTPUT_PATH = Path("data/rasters/synthetic_dem.tif")

M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def sw_corner_lonlat() -> Tuple[float, float]:
    half = DOMAIN_SIZE_M / 2.0
    lon = RANCH_CENTER_LON - half / _m_per_deg_lon(RANCH_CENTER_LAT)
    lat = RANCH_CENTER_LAT - half / M_PER_DEG_LAT
    return lon, lat


def local_m_to_lonlat(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert local estate meters (east, north from SW corner) to WGS84 lon/lat."""
    sw_lon, sw_lat = sw_corner_lonlat()
    lon = sw_lon + x_m / _m_per_deg_lon(RANCH_CENTER_LAT)
    lat = sw_lat + y_m / M_PER_DEG_LAT
    return lon, lat


def linestring_wkt(points_xy: Sequence[Tuple[float, float]]) -> str:
    coords = ", ".join(
        f"{local_m_to_lonlat(x, y)[0]:.8f} {local_m_to_lonlat(x, y)[1]:.8f}"
        for x, y in points_xy
    )
    return f"LINESTRING({coords})"


def point_wkt(x_m: float, y_m: float) -> str:
    lon, lat = local_m_to_lonlat(x_m, y_m)
    return f"POINT({lon:.8f} {lat:.8f})"


@dataclass(frozen=True)
class SensorSeed:
    node_name: str
    sender_email: str
    x_m: float
    y_m: float
    azimuth_deg: float
    fov_angle: float = 85.0
    detection_radius_m: float = 25.0


SENSOR_NODES: Tuple[SensorSeed, ...] = (
    SensorSeed("cam-gate-north", "cam-gate-north@ranch.domain", 5000, 10000, 180.0),
    SensorSeed("cam-gate-south", "cam-gate-south@ranch.domain", 5000, 0, 0.0),
    SensorSeed("cam-gate-east", "cam-gate-east@ranch.domain", 10000, 5000, 270.0),
    SensorSeed("cam-gate-west", "cam-gate-west@ranch.domain", 0, 5000, 90.0),
    SensorSeed("cam-intersection", "cam-intersection@ranch.domain", 5000, 5000, 45.0),
    SensorSeed("cam-corridor-4k", "cam-corridor-4k@ranch.domain", 5000, 6000, 180.0),
)

CHOKEPOINTS: Tuple[Tuple[str, float, float], ...] = (
    ("North Gate", 5000, 10000),
    ("South Gate", 5000, 0),
    ("East Gate", 10000, 5000),
    ("West Gate", 0, 5000),
)


def generate_elevation_array() -> np.ndarray:
    xs = np.arange(GRID_SIZE, dtype=np.float64) * RESOLUTION_M
    ys = np.arange(GRID_SIZE, dtype=np.float64) * RESOLUTION_M
    xx, yy = np.meshgrid(xs, ys)
    return (
        400.0
        + 150.0 * np.sin(0.001 * xx) * np.cos(0.001 * yy)
        + 80.0
        * np.exp(-((xx - 5000.0) ** 2 + (yy - 5000.0) ** 2) / (2.0 * 1200.0 ** 2))
    )


def generate_slope_array(elevation: np.ndarray | None = None) -> np.ndarray:
    dem = elevation if elevation is not None else generate_elevation_array()
    return slope_degrees_grid(dem)


def export_dem_geotiff(
    elevation: np.ndarray | None = None,
    output_path: Path | None = None,
) -> Path:
    """Write synthetic DEM as GeoTIFF in local meter CRS (EPSG:32611-ish via WKT)."""
    import rasterio

    dem = elevation if elevation is not None else generate_elevation_array()
    out = output_path or DEM_OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    sw_lon, sw_lat = sw_corner_lonlat()
    pixel_w = RESOLUTION_M / _m_per_deg_lon(RANCH_CENTER_LAT)
    pixel_h = RESOLUTION_M / M_PER_DEG_LAT
    north_lat = sw_lat + DOMAIN_SIZE_M / M_PER_DEG_LAT
    transform = from_origin(sw_lon, north_lat, pixel_w, -pixel_h)

    profile = {
        "driver": "GTiff",
        "height": GRID_SIZE,
        "width": GRID_SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(dem.astype(np.float32), 1)
    return out


def _road_edges_local() -> List[Tuple[Sequence[Tuple[float, float]], str, float]]:
    """Interior ranch roads + perimeter connector segments (local meters)."""
    max_speed = 40.0
    return [
        ([(5000, 0), (5000, 5000), (5000, 10000)], "gravel", max_speed),
        ([(0, 5000), (5000, 5000), (10000, 5000)], "gravel", max_speed),
        ([(0, 5000), (0, 10000), (5000, 10000), (10000, 10000), (10000, 5000), (10000, 0), (5000, 0), (0, 0), (0, 5000)], "gravel", 25.0),
    ]


def _edge_length_m(points: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _segment_edges(
    points: Sequence[Tuple[float, float]], surface: str, max_speed_kmh: float
) -> Iterable[Tuple[str, float, float, str, float]]:
    for i in range(len(points) - 1):
        seg = (points[i], points[i + 1])
        length_m = _edge_length_m(seg)
        cost_s = length_m / (max_speed_kmh / 3.6)
        wkt = linestring_wkt(seg)
        yield wkt, cost_s, cost_s, surface, max_speed_kmh


def init_schema(conn: psycopg.Connection) -> None:
    sql_path = Path(__file__).resolve().parents[1] / "src" / "common" / "init_db.sql"
    conn.execute(sql_path.read_text())
    conn.commit()


def clear_simulation_tables(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        TRUNCATE TABLE tactical_directives, detection_events, entity_tracks,
            strategic_chokepoints, property_network_edges, sensor_nodes,
            elevation_rasters
        RESTART IDENTITY CASCADE;
        """
    )
    conn.commit()


def seed_road_network(conn: psycopg.Connection) -> int:
    count = 0
    for polyline, surface, speed in _road_edges_local():
        for wkt, cost, rev_cost, surf, max_kmh in _segment_edges(polyline, surface, speed):
            conn.execute(
                """
                INSERT INTO property_network_edges
                    (source, target, cost, reverse_cost, surface_type, max_speed_kmh, geom)
                VALUES (0, 0, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
                """,
                (cost, rev_cost, surf, max_kmh, wkt),
            )
            count += 1
    conn.execute(
        """
        SELECT pgr_createTopology(
            'property_network_edges', 0.0001, 'geom', 'id', 'source', 'target', clean := true
        );
        """
    )
    conn.execute(
        """
        SELECT pgr_analyzeGraph(
            'property_network_edges', 0.0001, 'geom', 'id', 'source', 'target'
        );
        """
    )
    conn.commit()
    return count


def seed_chokepoints(conn: psycopg.Connection) -> None:
    for name, x_m, y_m in CHOKEPOINTS:
        conn.execute(
            """
            INSERT INTO strategic_chokepoints (name, chokepoint_type, location)
            VALUES (%s, 'gate', ST_GeomFromText(%s, 4326))
            """,
            (name, point_wkt(x_m, y_m)),
        )
    conn.commit()


def seed_sensor_nodes(conn: psycopg.Connection) -> None:
    for cam in SENSOR_NODES:
        conn.execute(
            """
            INSERT INTO sensor_nodes
                (node_name, sender_email, location, azimuth_deg, fov_angle, detection_radius_m,
                 battery_level, solar_charging, is_active)
            VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s, 92, TRUE, TRUE)
            """,
            (
                cam.node_name,
                cam.sender_email,
                point_wkt(cam.x_m, cam.y_m),
                cam.azimuth_deg,
                cam.fov_angle,
                cam.detection_radius_m,
            ),
        )
    conn.commit()


def seed_elevation_raster(conn: psycopg.Connection, tif_path: Path | None = None) -> None:
    path = tif_path or export_dem_geotiff()
    data = path.read_bytes()
    conn.execute(
        """
        INSERT INTO elevation_rasters (rast, filename)
        VALUES (ST_FromGDALRaster(%s), %s)
        """,
        (data, path.name),
    )
    conn.commit()


def seed_all(
    conn: psycopg.Connection | None = None,
    *,
    init_schema_if_needed: bool = True,
    clear_existing: bool = True,
) -> dict:
    """Full synthetic GIS + network seed for simulation."""
    own_conn = conn is None
    if own_conn:
        conn = psycopg.connect(settings.sync_database_url)

    try:
        if init_schema_if_needed:
            init_schema(conn)
        if clear_existing:
            clear_simulation_tables(conn)
        tif = export_dem_geotiff()
        seed_elevation_raster(conn, tif)
        edge_count = seed_road_network(conn)
        seed_chokepoints(conn)
        seed_sensor_nodes(conn)
        return {
            "dem_path": str(tif),
            "network_edges": edge_count,
            "chokepoints": len(CHOKEPOINTS),
            "sensor_nodes": len(SENSOR_NODES),
        }
    finally:
        if own_conn:
            conn.close()


def get_sensor_by_name(conn: psycopg.Connection, node_name: str) -> Tuple[str, str, float, float]:
    row = conn.execute(
        """
        SELECT id::text, sender_email,
               ST_X(location::geometry), ST_Y(location::geometry)
        FROM sensor_nodes WHERE node_name = %s
        """,
        (node_name,),
    ).fetchone()
    if not row:
        raise KeyError(f"Unknown sensor node: {node_name}")
    return row[0], row[1], row[2], row[3]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = seed_all()
    logger.info("Synthetic GIS seed complete: %s", summary)
