"""
Spatial zone management: default pastures/yards, containment, and headcounts (spec 13).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from geopy.distance import geodesic
from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon, shape
from shapely.ops import nearest_points
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

_LIVESTOCK_CLASSES = frozenset(
    {"cow", "cattle", "sheep", "goat", "horse", "livestock", "bison", "calf", "heifer"}
)
_GEOFENCE_PROXIMITY_M = 100.0


class ZoneCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    zone_code: str = Field(..., min_length=2, max_length=32)
    zone_type: str = "pasture"
    color: str = "#22c55e"
    coordinates: List[List[float]]
    properties: Dict[str, Any] = Field(default_factory=dict)


class ZoneUpdateRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    coordinates: Optional[List[List[float]]] = None
    properties: Optional[Dict[str, Any]] = None


class GeofenceBreachEvent(BaseModel):
    event_id: UUID
    track_id: UUID
    track_code: str
    entity_class: str
    assigned_zone_name: str
    current_location: List[float]
    distance_to_boundary_m: float
    breach_type: str
    timestamp: datetime


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
            "zone_id": raw.get("zone_id") or str(uuid4()),
            "name": raw["name"],
            "zone_code": raw["zone_code"],
            "zone_type": raw.get("zone_type", "zone"),
            "headcount": headcount,
            "color": raw.get("color", "#38bdf8"),
        }
        extra = raw.get("properties") or {}
        if isinstance(extra, dict):
            props.update({k: v for k, v in extra.items() if k not in props})
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": props,
        }

    @staticmethod
    def _normalize_polygon_ring(coordinates: List[List[float]]) -> List[List[float]]:
        if len(coordinates) < 3:
            raise ValueError("Polygon requires at least 3 vertices")
        ring = [[float(pt[0]), float(pt[1])] for pt in coordinates]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            raise ValueError("Invalid polygon geometry")
        return ring

    @staticmethod
    def _distance_to_boundary_m(lon: float, lat: float, poly: Any) -> float:
        pt = Point(lon, lat)
        nearest = nearest_points(pt, poly.boundary)[1]
        return float(geodesic((lat, lon), (nearest.y, nearest.x)).meters)

    def _find_zone_index(self, zone_id_or_code: str) -> Optional[int]:
        key = zone_id_or_code.strip()
        for idx, feature in enumerate(self._zones):
            props = feature.get("properties") or {}
            if props.get("zone_id") == key or props.get("zone_code") == key:
                return idx
        return None

    def create_zone(self, request: ZoneCreateRequest) -> Dict[str, Any]:
        """Validate polygon, persist in memory, and return a GeoJSON feature."""
        for feature in self._zones:
            if feature["properties"]["zone_code"] == request.zone_code:
                raise ValueError(f"Zone code already exists: {request.zone_code}")

        ring = self._normalize_polygon_ring(request.coordinates)
        geometry = {"type": "Polygon", "coordinates": [ring]}
        zone_id = str(uuid4())
        raw = {
            "zone_id": zone_id,
            "name": request.name,
            "zone_code": request.zone_code,
            "zone_type": request.zone_type,
            "color": request.color,
            "geometry": geometry,
            "properties": dict(request.properties),
        }
        feature = self._zone_dict_to_feature(raw, headcount=0)
        feature["properties"]["zone_id"] = zone_id
        feature["properties"]["color"] = request.color
        self._zones.append(feature)
        self._polygons.append((feature, shape(geometry)))
        return dict(feature)

    def update_zone(
        self, zone_id_or_code: str, update: ZoneUpdateRequest
    ) -> Optional[Dict[str, Any]]:
        idx = self._find_zone_index(zone_id_or_code)
        if idx is None:
            return None
        feature = self._zones[idx]
        props = feature["properties"]
        if update.name is not None:
            props["name"] = update.name
        if update.color is not None:
            props["color"] = update.color
        if update.properties is not None:
            props.update(update.properties)
        if update.coordinates is not None:
            ring = self._normalize_polygon_ring(update.coordinates)
            feature["geometry"] = {"type": "Polygon", "coordinates": [ring]}
        self._polygons[idx] = (feature, shape(feature["geometry"]))
        return dict(feature)

    def delete_zone(self, zone_id_or_code: str) -> bool:
        idx = self._find_zone_index(zone_id_or_code)
        if idx is None:
            return False
        del self._zones[idx]
        del self._polygons[idx]
        return True

    def detect_geofence_breaches(
        self, tracks: List[Dict[str, Any]]
    ) -> List[GeofenceBreachEvent]:
        """Flag livestock outside pastures or within proximity of a pasture boundary."""
        pasture_items = [
            (feature, geom)
            for feature, geom in self._polygons
            if (feature.get("properties") or {}).get("zone_type") == "pasture"
        ]
        restricted_items = [
            (feature, geom)
            for feature, geom in self._polygons
            if (feature.get("properties") or {}).get("zone_type")
            in ("restricted_zone", "container_bay")
        ]

        events: List[GeofenceBreachEvent] = []
        now = datetime.now(timezone.utc)

        for track in tracks:
            status = str(track.get("status") or track.get("current_status") or "ACTIVE").upper()
            if status != "ACTIVE":
                continue
            entity_class = str(
                track.get("entity_class") or track.get("object_class") or ""
            ).lower()
            if entity_class not in _LIVESTOCK_CLASSES:
                continue

            lon, lat = self._track_lon_lat(track)
            if lon is None or lat is None:
                continue

            track_id = self._track_uuid(track)
            track_code = str(track.get("track_code") or "UNKNOWN")
            assigned_name = str(
                track.get("assigned_zone_name")
                or track.get("home_zone_name")
                or "Unassigned"
            )

            for feature, geom in restricted_items:
                if geom.contains(Point(lon, lat)):
                    events.append(
                        GeofenceBreachEvent(
                            event_id=uuid4(),
                            track_id=track_id,
                            track_code=track_code,
                            entity_class=entity_class,
                            assigned_zone_name=assigned_name,
                            current_location=[lon, lat],
                            distance_to_boundary_m=self._distance_to_boundary_m(
                                lon, lat, geom
                            ),
                            breach_type="RESTRICTED_ZONE_ENTRY",
                            timestamp=now,
                        )
                    )
                    break
            else:
                containing_pasture: Optional[Tuple[Dict[str, Any], Any]] = None
                for feature, geom in pasture_items:
                    if geom.contains(Point(lon, lat)):
                        containing_pasture = (feature, geom)
                        break

                if containing_pasture is None:
                    min_dist = math.inf
                    for _feature, geom in pasture_items:
                        dist = self._distance_to_boundary_m(lon, lat, geom)
                        min_dist = min(min_dist, dist)
                    events.append(
                        GeofenceBreachEvent(
                            event_id=uuid4(),
                            track_id=track_id,
                            track_code=track_code,
                            entity_class=entity_class,
                            assigned_zone_name=assigned_name,
                            current_location=[lon, lat],
                            distance_to_boundary_m=(
                                min_dist if min_dist != math.inf else 0.0
                            ),
                            breach_type="STRAY_OUT_OF_BOUNDS",
                            timestamp=now,
                        )
                    )
                else:
                    _feature, geom = containing_pasture
                    dist_m = self._distance_to_boundary_m(lon, lat, geom)
                    if dist_m <= _GEOFENCE_PROXIMITY_M:
                        zone_name = (_feature.get("properties") or {}).get(
                            "name", assigned_name
                        )
                        events.append(
                            GeofenceBreachEvent(
                                event_id=uuid4(),
                                track_id=track_id,
                                track_code=track_code,
                                entity_class=entity_class,
                                assigned_zone_name=zone_name,
                                current_location=[lon, lat],
                                distance_to_boundary_m=dist_m,
                                breach_type="STRAY_OUT_OF_BOUNDS",
                                timestamp=now,
                            )
                        )

        return events

    @staticmethod
    def _track_lon_lat(track: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        point = track.get("last_known_point")
        if point and len(point) >= 2:
            return float(point[0]), float(point[1])
        lon = track.get("lon")
        lat = track.get("lat")
        if lon is not None and lat is not None:
            return float(lon), float(lat)
        return None, None

    @staticmethod
    def _track_uuid(track: Dict[str, Any]) -> UUID:
        raw = track.get("track_id") or track.get("id")
        if raw:
            return UUID(str(raw))
        return uuid4()

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
                    "zone_id": props.get("zone_id"),
                    "color": props.get("color", "#38bdf8"),
                    "geometry": geom,
                    "properties": props,
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
