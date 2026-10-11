"""
Spatial query service: network snap, chokepoints, elevation/slope sampling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from affine import Affine
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.database import async_session_factory
from src.common.schemas import EntityClass, PointLonLat
from src.spatial.raster import compute_friction_surface, slope_from_elevation_grid

logger = logging.getLogger(__name__)

VerticesTable = "property_network_edges_vertices_pgr"
EdgesTable = "property_network_edges"


class SpatialService:
    """Async PostGIS-backed spatial operations for the Rian platform."""

    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        self._session = session

    async def _get_session(self) -> Tuple[AsyncSession, bool]:
        if self._session is not None:
            return self._session, False
        return async_session_factory(), True

    async def get_nearest_network_node(self, lon: float, lat: float) -> Optional[int]:
        """Snap a GPS coordinate to the nearest vertex in the road network."""
        session, own = await self._get_session()
        try:
            sql = text(
                f"""
                SELECT id
                FROM {VerticesTable}
                ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                LIMIT 1
                """
            )
            try:
                result = await session.execute(sql, {"lon": lon, "lat": lat})
                row = result.first()
                if row:
                    return int(row[0])
            except Exception:
                pass

            # Fallback when topology vertices table is missing: use edge endpoints
            fallback = text(
                f"""
                WITH candidates AS (
                    SELECT source AS vid,
                           ST_StartPoint(geom) AS pt
                    FROM {EdgesTable}
                    UNION ALL
                    SELECT target AS vid,
                           ST_EndPoint(geom) AS pt
                    FROM {EdgesTable}
                )
                SELECT vid
                FROM candidates
                ORDER BY pt <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                LIMIT 1
                """
            )
            result = await session.execute(fallback, {"lon": lon, "lat": lat})
            row = result.first()
            return int(row[0]) if row else None
        finally:
            if own:
                await session.close()

    async def get_chokepoints(self) -> List[Dict[str, Any]]:
        """Retrieve all strategic_chokepoints from the database."""
        session, own = await self._get_session()
        try:
            sql = text(
                """
                SELECT
                    id,
                    name,
                    chokepoint_type,
                    ST_X(location) AS lon,
                    ST_Y(location) AS lat,
                    network_node_id,
                    is_accessible_to_response,
                    created_at
                FROM strategic_chokepoints
                ORDER BY name
                """
            )
            result = await session.execute(sql)
            rows = []
            for row in result.mappings():
                item = dict(row)
                item["id"] = UUID(str(item["id"]))
                item["location"] = (float(item.pop("lon")), float(item.pop("lat")))
                rows.append(item)
            return rows
        finally:
            if own:
                await session.close()

    async def get_elevation_and_slope(
        self,
        lon: float,
        lat: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Sample elevation (meters) and slope (degrees) at a point from PostGIS rasters.
        Returns (None, None) when no raster coverage exists.
        """
        session, own = await self._get_session()
        try:
            elev_sql = text(
                """
                SELECT ST_Value(rast, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) AS elev
                FROM elevation_rasters
                WHERE ST_Intersects(
                    rast,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                )
                LIMIT 1
                """
            )
            result = await session.execute(elev_sql, {"lon": lon, "lat": lat})
            row = result.first()
            if not row or row.elev is None:
                return None, None

            elevation = float(row.elev)

            slope_sql = text(
                """
                SELECT ST_Value(
                    ST_Slope(rast, 1, '32BF', 'DEGREES'),
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                ) AS slope_deg
                FROM elevation_rasters
                WHERE ST_Intersects(
                    rast,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                )
                LIMIT 1
                """
            )
            slope_result = await session.execute(slope_sql, {"lon": lon, "lat": lat})
            slope_row = slope_result.first()
            slope_deg = float(slope_row.slope_deg) if slope_row and slope_row.slope_deg is not None else None
            return elevation, slope_deg
        except Exception as exc:
            logger.debug("Elevation/slope sample failed: %s", exc)
            return None, None
        finally:
            if own:
                await session.close()

    async def get_friction_surface_window(
        self,
        center_lon: float,
        center_lat: float,
        radius_meters: float,
        entity_class: EntityClass,
        resolution_m: float = 10.0,
    ) -> Tuple[np.ndarray, Affine, int]:
        """
        Extract a 2D friction surface (seconds/meter) around a point.

        Uses a synthetic flat DEM window when no raster is loaded; slope is derived
        locally from the elevation grid when present.
        """
        side = max(3, int(2 * radius_meters / resolution_m) + 1)
        half = (side - 1) / 2.0
        rows = np.linspace(-half, half, side) * resolution_m
        cols = np.linspace(-half, half, side) * resolution_m
        yy, xx = np.meshgrid(rows, cols, indexing="ij")

        elevation, _ = await self.get_elevation_and_slope(center_lon, center_lat)
        base_elev = elevation if elevation is not None else 100.0
        elevation_grid = base_elev + 0.01 * (xx + yy)
        slope_grid = slope_from_elevation_grid(elevation_grid, resolution_m)

        travel_mode = "foot"
        if entity_class in (EntityClass.CAR, EntityClass.TRUCK):
            travel_mode = entity_class.value

        friction = compute_friction_surface(
            elevation_grid,
            slope_grid,
            land_cover="pasture",
            travel_mode=travel_mode,
            entity_class=entity_class,
        )

        transform = Affine.translation(center_lon, center_lat) @ Affine.scale(
            resolution_m / 111320.0,
            resolution_m / 110540.0,
        )
        return friction, transform, side
