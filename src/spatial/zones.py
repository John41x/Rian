"""
Spatial zone management: default pastures/yards, containment, and headcounts (spec 13).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point, shape
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.domains import get_domain_profile
from src.common.database import async_session_factory

logger = logging.getLogger(__name__)

_SPATIAL_ZONES_DDL = """
CREATE TABLE IF NOT EXISTS spatial_zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(64) NOT NULL,
    zone_code VARCHAR(32) UNIQUE NOT NULL,
    zone_type VARCHAR(32) NOT NULL,
    boundary GEOMETRY(Polygon, 4326) NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_spatial_zones_boundary ON spatial_zones USING GIST(boundary);
"""


class ZoneManager:
    """Loads domain default zones and supports optional PostGIS persistence."""

    def __init__(self, profile_name: Optional[str] = None) -> None:
        self._profile = get_domain_profile(profile_name)
        self._zones: List[Dict[str, Any]] = []
        self._polygons: List[Tuple[Dict[str, Any], Any]] = []
        self._load_defaults()

    def _load_defaults(self) -> None:
        self._zones = []
        self._polygons = []
        for raw in self._profile.default_zones:
            feature = self._zone_dict_to_feature(raw, headcount=0)
            self._zones.append(feature)
            geom = shape(feature["geometry"])
            self._polygons.append((feature, geom))

    @staticmethod
    def _zone_dict_to_feature(raw: Dict[str, Any], headcount: int = 0) -> Dict[str, Any]:
        geometry = raw.get("geometry")
        if geometry is None and "coordinates" in raw:
            geometry = {"type": "Polygon", "coordinates": raw["coordinates"]}
        props = {
            "name": raw["name"],
            "zone_code": raw["zone_code"],
            "zone_type": raw.get("zone_type", "zone"),
            "headcount": headcount,
            "color": raw.get("color", "#38bdf8"),
        }
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": props,
        }

    def list_zones(self) -> List[Dict[str, Any]]:
        """Return GeoJSON features for all spatial zones."""
        return [dict(z) for z in self._zones]

    def as_feature_collection(self) -> Dict[str, Any]:
        return {"type": "FeatureCollection", "features": self.list_zones()}

    def find_zone_for_point(self, lon: float, lat: float) -> Optional[Dict[str, Any]]:
        """Return the zone feature containing the given WGS84 coordinate."""
        pt = Point(lon, lat)
        for feature, poly in self._polygons:
            if poly.contains(pt):
                return dict(feature)
        return None

    def update_zone_headcounts(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Count active entities per zone and refresh headcount properties."""
        counts: Dict[str, int] = {z["properties"]["zone_code"]: 0 for z in self._zones}

        for track in tracks:
            status = str(track.get("status") or track.get("current_status") or "ACTIVE").upper()
            if status not in ("ACTIVE",):
                continue
            point = track.get("last_known_point")
            if not point or len(point) < 2:
                lon = track.get("lon")
                lat = track.get("lat")
                if lon is None or lat is None:
                    continue
            else:
                lon, lat = float(point[0]), float(point[1])
            zone = self.find_zone_for_point(lon, lat)
            if zone:
                code = zone["properties"]["zone_code"]
                counts[code] = counts.get(code, 0) + 1

        for feature in self._zones:
            code = feature["properties"]["zone_code"]
            feature["properties"]["headcount"] = counts.get(code, 0)

        self._polygons = [(z, shape(z["geometry"])) for z in self._zones]
        return self.list_zones()

    async def ensure_table(self, session: Optional[AsyncSession] = None) -> None:
        own = session is None
        if own:
            session = async_session_factory()
        try:
            await session.execute(text(_SPATIAL_ZONES_DDL))
            await session.commit()
        except Exception as exc:
            logger.warning("spatial_zones DDL skipped: %s", exc)
            if own:
                await session.rollback()
        finally:
            if own:
                await session.close()

    async def load_from_database(self, session: Optional[AsyncSession] = None) -> bool:
        """Load zones from PostGIS when the table is populated; return True if loaded."""
        own = session is None
        if own:
            session = async_session_factory()
        try:
            result = await session.execute(
                text(
                    """
                    SELECT
                        name,
                        zone_code,
                        zone_type,
                        ST_AsGeoJSON(boundary) AS geom_json,
                        properties
                    FROM spatial_zones
                    ORDER BY name
                    """
                )
            )
            rows = result.mappings().all()
            if not rows:
                return False

            self._zones = []
            self._polygons = []
            for row in rows:
                geom = json.loads(row["geom_json"])
                props = row["properties"] or {}
                if isinstance(props, str):
                    props = json.loads(props)
                headcount = int(props.get("headcount", 0))
                raw = {
                    "name": row["name"],
                    "zone_code": row["zone_code"],
                    "zone_type": row["zone_type"],
                    "color": props.get("color", "#38bdf8"),
                    "geometry": geom,
                }
                feature = self._zone_dict_to_feature(raw, headcount=headcount)
                self._zones.append(feature)
                self._polygons.append((feature, shape(geom)))
            return True
        except Exception as exc:
            logger.debug("load_from_database failed, using defaults: %s", exc)
            return False
        finally:
            if own:
                await session.close()

    async def seed_database_if_empty(self, session: Optional[AsyncSession] = None) -> None:
        """Persist default profile zones when spatial_zones has no rows."""
        own = session is None
        if own:
            session = async_session_factory()
        try:
            await self.ensure_table(session)
            count_row = await session.execute(text("SELECT COUNT(*) FROM spatial_zones"))
            if int(count_row.scalar() or 0) > 0:
                return

            for raw in self._profile.default_zones:
                geom_json = json.dumps(raw["geometry"])
                props = json.dumps({"headcount": 0, "color": raw.get("color", "#38bdf8")})
                await session.execute(
                    text(
                        """
                        INSERT INTO spatial_zones (name, zone_code, zone_type, boundary, properties)
                        VALUES (
                            :name,
                            :zone_code,
                            :zone_type,
                            ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326),
                            CAST(:props AS jsonb)
                        )
                        ON CONFLICT (zone_code) DO NOTHING
                        """
                    ),
                    {
                        "name": raw["name"],
                        "zone_code": raw["zone_code"],
                        "zone_type": raw.get("zone_type", "zone"),
                        "geom": geom_json,
                        "props": props,
                    },
                )
            await session.commit()
        except Exception as exc:
            logger.warning("seed_database_if_empty failed: %s", exc)
            if own:
                await session.rollback()
        finally:
            if own:
                await session.close()
