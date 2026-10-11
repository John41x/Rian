"""Anisotropic cost-distance isochrone wavefront solver (spec §4)."""

from __future__ import annotations

import heapq
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy import special
from shapely import geometry as geom
from shapely.ops import unary_union

from src.common.schemas import EntityClass, PointLonLat
from src.tracking.constants import CLASS_HEADING_KAPPA, ISOCHRONE_CELL_SIZE_M


def _meters_to_lonlat_offset(
    center_lon: float,
    center_lat: float,
    east_m: float,
    north_m: float,
) -> PointLonLat:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))
    lon = center_lon + east_m / m_per_deg_lon
    lat = center_lat + north_m / m_per_deg_lat
    return (lon, lat)


def _grid_index_to_lonlat(
    center_lon: float,
    center_lat: float,
    row: int,
    col: int,
    side: int,
    cell_size_m: float,
) -> PointLonLat:
    half = (side - 1) / 2.0
    east_m = (col - half) * cell_size_m
    north_m = (half - row) * cell_size_m
    return _meters_to_lonlat_offset(center_lon, center_lat, east_m, north_m)


def von_mises_direction_weight(
    step_azimuth_rad: float,
    heading_deg: float,
    kappa: float,
) -> float:
    """Directional weight W(theta) from spec §4.2."""
    if kappa <= 0:
        return 1.0
    theta0 = math.radians(heading_deg)
    return math.exp(kappa * math.cos(step_azimuth_rad - theta0)) / (
        2.0 * math.pi * special.iv(0, kappa)
    )


def solve_travel_time_grid(
    friction_s_per_m: np.ndarray,
    cell_size_m: float = ISOCHRONE_CELL_SIZE_M,
    heading_deg: Optional[float] = None,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Solve |grad T| = S(x,y) on a regular grid via Dijkstra (Eikonal discretization).

    Returns arrival time T in seconds from the grid center cell.
    """
    friction = np.asarray(friction_s_per_m, dtype=np.float64)
    if friction.ndim != 2 or friction.shape[0] < 2 or friction.shape[1] < 2:
        raise ValueError("friction grid must be at least 2x2")

    rows, cols = friction.shape
    center_r, center_c = rows // 2, cols // 2
    times = np.full(friction.shape, np.inf, dtype=np.float64)
    times[center_r, center_c] = 0.0

    heap: List[Tuple[float, int, int]] = [(0.0, center_r, center_c)]
    neighbors = ((0, 1), (0, -1), (1, 0), (-1, 0))

    while heap:
        current_t, r, c = heapq.heappop(heap)
        if current_t > times[r, c]:
            continue
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            s = friction[nr, nc]
            if not np.isfinite(s) or s <= 0:
                continue
            edge_cost = s * cell_size_m
            if heading_deg is not None:
                azimuth = math.atan2(dc, -dr)
                w = von_mises_direction_weight(azimuth, heading_deg, kappa)
                edge_cost /= max(w, 1e-9)
            new_t = current_t + edge_cost
            if new_t < times[nr, nc]:
                times[nr, nc] = new_t
                heapq.heappush(heap, (new_t, nr, nc))

    return times


def travel_time_between_cells(
    friction_s_per_m: np.ndarray,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
    cell_size_m: float = ISOCHRONE_CELL_SIZE_M,
    heading_deg: Optional[float] = None,
    kappa: float = 1.0,
) -> float:
    """Least-cost travel time from one grid cell to another on the same friction surface."""
    friction = np.asarray(friction_s_per_m, dtype=np.float64)
    rows, cols = friction.shape
    if not (0 <= from_row < rows and 0 <= from_col < cols):
        return float("inf")
    if not (0 <= to_row < rows and 0 <= to_col < cols):
        return float("inf")

    times = np.full(friction.shape, np.inf, dtype=np.float64)
    times[from_row, from_col] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, from_row, from_col)]
    neighbors = ((0, 1), (0, -1), (1, 0), (-1, 0))

    while heap:
        current_t, r, c = heapq.heappop(heap)
        if r == to_row and c == to_col:
            return current_t
        if current_t > times[r, c]:
            continue
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            s = friction[nr, nc]
            if not np.isfinite(s) or s <= 0:
                continue
            edge_cost = s * cell_size_m
            if heading_deg is not None:
                azimuth = math.atan2(dc, -dr)
                w = von_mises_direction_weight(azimuth, heading_deg, kappa)
                edge_cost /= max(w, 1e-9)
            new_t = current_t + edge_cost
            if new_t < times[nr, nc]:
                times[nr, nc] = new_t
                heapq.heappush(heap, (new_t, nr, nc))

    return float("inf")


def lonlat_to_grid_cell(
    center_lon: float,
    center_lat: float,
    lon: float,
    lat: float,
    side: int,
    cell_size_m: float = ISOCHRONE_CELL_SIZE_M,
) -> Tuple[int, int]:
    """Map WGS84 point to row/col on a grid centered at center_lon/lat."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))
    east_m = (lon - center_lon) * m_per_deg_lon
    north_m = (lat - center_lat) * m_per_deg_lat
    half = (side - 1) / 2.0
    col = int(round(half + east_m / cell_size_m))
    row = int(round(half - north_m / cell_size_m))
    row = max(0, min(side - 1, row))
    col = max(0, min(side - 1, col))
    return row, col


def extract_reachability_contour(
    travel_times: np.ndarray,
    center_lon: float,
    center_lat: float,
    elapsed_seconds: float,
    cell_size_m: float = ISOCHRONE_CELL_SIZE_M,
) -> Optional[geom.base.BaseGeometry]:
    """Extract polygon(s) where T(x,y) <= elapsed_seconds."""
    if elapsed_seconds <= 0:
        lon, lat = center_lon, center_lat
        return geom.Point(lon, lat).buffer(1e-6)

    side = travel_times.shape[0]
    mask = np.isfinite(travel_times) & (travel_times <= elapsed_seconds)
    if not np.any(mask):
        return None

    polygons: List[geom.Polygon] = []
    rows, cols = travel_times.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = (
                mask[r, c],
                mask[r, c + 1],
                mask[r + 1, c + 1],
                mask[r + 1, c],
            )
            if not any(corners):
                continue
            if all(corners):
                ring = [
                    _grid_index_to_lonlat(center_lon, center_lat, r, c, side, cell_size_m),
                    _grid_index_to_lonlat(center_lon, center_lat, r, c + 1, side, cell_size_m),
                    _grid_index_to_lonlat(center_lon, center_lat, r + 1, c + 1, side, cell_size_m),
                    _grid_index_to_lonlat(center_lon, center_lat, r + 1, c, side, cell_size_m),
                    _grid_index_to_lonlat(center_lon, center_lat, r, c, side, cell_size_m),
                ]
                polygons.append(geom.Polygon(ring))
                continue

            # Partial cell: approximate with center point if any corner is reachable
            if any(corners):
                buff_deg = cell_size_m / 111_320.0
                for rr, cc in ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)):
                    if mask[rr, cc]:
                        lon, lat = _grid_index_to_lonlat(
                            center_lon, center_lat, rr, cc, side, cell_size_m
                        )
                        polygons.append(geom.Point(lon, lat).buffer(buff_deg * 0.45))

    if not polygons:
        return None

    merged = unary_union(polygons)
    if merged.is_empty:
        return None
    return merged


def geometry_to_geojson_coordinates(
    shape: geom.base.BaseGeometry,
) -> Dict[str, Any]:
    """Convert shapely geometry to GeoJSON geometry dict (EPSG:4326 lon/lat)."""
    if shape.geom_type == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [
                [[float(x), float(y)] for x, y in shape.exterior.coords],
            ],
        }
    if shape.geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[[float(x), float(y)] for x, y in poly.exterior.coords]]
                for poly in shape.geoms
            ],
        }
    if shape.geom_type == "Point":
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [shape.x, shape.y],
                    [shape.x, shape.y],
                    [shape.x, shape.y],
                    [shape.x, shape.y],
                ]
            ],
        }
    buffered = shape.buffer(1e-5)
    return geometry_to_geojson_coordinates(buffered)


def compute_reachability_geojson(
    friction_s_per_m: np.ndarray,
    center_lon: float,
    center_lat: float,
    elapsed_seconds: float,
    entity_class: EntityClass,
    track_id: UUID,
    track_code: str,
    confidence: float = 1.0,
    heading_deg: Optional[float] = None,
    cell_size_m: float = ISOCHRONE_CELL_SIZE_M,
    generated_at: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Build RFC 7946 reachability Feature for elapsed time since last detection."""
    kappa = CLASS_HEADING_KAPPA.get(entity_class, 1.0)
    travel_times = solve_travel_time_grid(
        friction_s_per_m,
        cell_size_m=cell_size_m,
        heading_deg=heading_deg,
        kappa=kappa,
    )
    contour = extract_reachability_contour(
        travel_times,
        center_lon,
        center_lat,
        elapsed_seconds,
        cell_size_m=cell_size_m,
    )
    if contour is None or contour.is_empty:
        return None

    generated_at = generated_at or datetime.now(timezone.utc)
    return {
        "type": "Feature",
        "geometry": geometry_to_geojson_coordinates(contour),
        "properties": {
            "track_id": str(track_id),
            "track_code": track_code,
            "target_class": entity_class.value,
            "elapsed_seconds": elapsed_seconds,
            "confidence": confidence,
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        },
    }


def build_local_friction_window(
    side: int,
    entity_class: EntityClass,
    slope_deg: float = 0.0,
    land_cover: str = "pasture",
    road_corridor_col: Optional[int] = None,
) -> np.ndarray:
    """
    Build a synthetic friction window for tests and offline association.

    When ``road_corridor_col`` is set, that column uses low vehicle impedance (road).
    """
    from src.spatial.raster import compute_friction_surface

    elev = np.zeros((side, side), dtype=np.float64)
    slope = np.full((side, side), slope_deg, dtype=np.float64)
    travel_mode = "foot"
    if entity_class in (EntityClass.CAR, EntityClass.TRUCK):
        travel_mode = entity_class.value

    if road_corridor_col is not None:
        land = np.full((side, side), land_cover, dtype=object)
        land[:, road_corridor_col] = "road_graded"
        friction = compute_friction_surface(
            elev, slope, land_cover=land, travel_mode=travel_mode, entity_class=entity_class
        )
    else:
        friction = compute_friction_surface(
            elev,
            slope,
            land_cover=land_cover,
            travel_mode=travel_mode,
            entity_class=entity_class,
        )
    return friction
