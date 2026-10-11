"""HMM track association and entity_tracks persistence (spec §3.2)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

import numpy as np
from geopy.distance import geodesic
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.common.database import async_session_factory
from src.common.schemas import EntityClass, PointLonLat, TrackState, TrackStatus
from src.spatial.service import SpatialService
from src.tracking.constants import (
    CLASS_HEADING_KAPPA,
    SIGMA_T_SECONDS,
    TAU_ASSOCIATE,
    VISUAL_SIMILAR_CLASS_PROB,
)
from src.tracking.gating import evaluate_spatiotemporal_gate
from src.tracking.reid import cosine_similarity
from src.tracking.isochrone import (
    build_local_friction_window,
    compute_reachability_geojson,
    lonlat_to_grid_cell,
    travel_time_between_cells,
)

logger = logging.getLogger(__name__)

_SIMILAR_CLASS_PAIRS = frozenset(
    {
        frozenset({EntityClass.CAR, EntityClass.TRUCK}),
    }
)

_DISJOINT_CLASS_PAIRS = frozenset(
    {
        frozenset({EntityClass.COW, EntityClass.CAR}),
        frozenset({EntityClass.COW, EntityClass.TRUCK}),
        frozenset({EntityClass.COW, EntityClass.PERSON}),
        frozenset({EntityClass.HORSE, EntityClass.CAR}),
        frozenset({EntityClass.HORSE, EntityClass.TRUCK}),
        frozenset({EntityClass.DOG, EntityClass.CAR}),
        frozenset({EntityClass.DOG, EntityClass.TRUCK}),
    }
)


@dataclass
class DetectionAssociationInput:
    detection_lon: float
    detection_lat: float
    detection_time: datetime
    object_class: EntityClass
    sensor_id: Optional[UUID] = None
    confidence: float = 1.0
    reid_embedding: Optional[List[float]] = None


@dataclass
class ActiveTrackRow:
    track_id: UUID
    track_code: str
    object_class: EntityClass
    last_detected_at: datetime
    last_lon: float
    last_lat: float
    estimated_heading_deg: Optional[float]
    estimated_velocity_ms: Optional[float]
    confidence_score: float
    last_reid_embedding: Optional[List[float]] = None


_REID_CLASS_WEIGHT = 0.3
_REID_APPEARANCE_WEIGHT = 0.7


def _class_association_probability(
    detection_class: EntityClass,
    track_class: EntityClass,
) -> float:
    if detection_class == track_class:
        return 1.0
    pair = frozenset({detection_class, track_class})
    if pair in _DISJOINT_CLASS_PAIRS:
        return 0.0
    if pair in _SIMILAR_CLASS_PAIRS:
        return VISUAL_SIMILAR_CLASS_PROB
    return 0.0


def visual_association_probability(
    detection_class: EntityClass,
    track_class: EntityClass,
    detection_reid: Optional[Sequence[float]] = None,
    track_reid: Optional[Sequence[float]] = None,
) -> float:
    """P_visual(c_k | T_j) with optional hybrid class + Re-ID appearance (spec 07)."""
    p_class = _class_association_probability(detection_class, track_class)
    if detection_reid is not None and track_reid is not None:
        appearance = max(0.0, cosine_similarity(detection_reid, track_reid))
        return _REID_CLASS_WEIGHT * p_class + _REID_APPEARANCE_WEIGHT * appearance
    return p_class


def spatial_association_probability(
    delta_t_observed: float,
    delta_t_least_cost: float,
    sigma_t: float = SIGMA_T_SECONDS,
) -> float:
    """P_spatial = exp(-(delta_t_obs - delta_t_lc) / sigma_t)."""
    if not math.isfinite(delta_t_least_cost) or delta_t_least_cost < 0:
        return 0.0
    if delta_t_observed <= 0:
        return 0.0
    exponent = -(delta_t_observed - delta_t_least_cost) / max(sigma_t, 1e-6)
    return float(math.exp(exponent))


def compute_association_score(
    detection: DetectionAssociationInput,
    track: ActiveTrackRow,
    friction_grid: np.ndarray,
    grid_center_lon: float,
    grid_center_lat: float,
) -> Tuple[float, bool]:
    """
    Return (P(z_k | T_j), passed_gating).
    """
    gate = evaluate_spatiotemporal_gate(
        track.last_lon,
        track.last_lat,
        track.last_detected_at,
        detection.detection_lon,
        detection.detection_lat,
        detection.detection_time,
        detection.object_class,
    )
    if not gate.reachable:
        return 0.0, False

    p_visual = visual_association_probability(
        detection.object_class,
        track.object_class,
        detection.reid_embedding,
        track.last_reid_embedding,
    )
    if p_visual <= 0:
        return 0.0, True

    side = friction_grid.shape[0]
    from_row, from_col = lonlat_to_grid_cell(
        grid_center_lon,
        grid_center_lat,
        track.last_lon,
        track.last_lat,
        side,
    )
    to_row, to_col = lonlat_to_grid_cell(
        grid_center_lon,
        grid_center_lat,
        detection.detection_lon,
        detection.detection_lat,
        side,
    )
    kappa = CLASS_HEADING_KAPPA.get(track.object_class, 1.0)
    delta_t_lc = travel_time_between_cells(
        friction_grid,
        from_row,
        from_col,
        to_row,
        to_col,
        heading_deg=track.estimated_heading_deg,
        kappa=kappa,
    )
    p_spatial = spatial_association_probability(gate.delta_t_seconds, delta_t_lc)
    return p_visual * p_spatial, True


def _bearing_deg(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    lat1, lon1 = math.radians(lat_a), math.radians(lon_a)
    lat2, lon2 = math.radians(lat_b), math.radians(lon_b)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


class TrackAssociator:
    """Associates detections with active tracks and persists to PostgreSQL."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
        spatial_service: Optional[SpatialService] = None,
    ) -> None:
        self._session_factory = session_factory
        self._spatial = spatial_service or SpatialService()

    async def fetch_active_tracks(self, session: AsyncSession) -> List[ActiveTrackRow]:
        sql = text(
            """
            SELECT
                id,
                track_code,
                object_class,
                last_detected_at,
                ST_X(last_known_point::geometry) AS lon,
                ST_Y(last_known_point::geometry) AS lat,
                estimated_heading_deg,
                estimated_velocity_ms,
                confidence_score,
                last_reid_embedding
            FROM entity_tracks
            WHERE current_status = 'ACTIVE'
            """
        )
        result = await session.execute(sql)
        rows: List[ActiveTrackRow] = []
        for row in result.mappings():
            rows.append(
                ActiveTrackRow(
                    track_id=UUID(str(row["id"])),
                    track_code=row["track_code"],
                    object_class=EntityClass(row["object_class"]),
                    last_detected_at=row["last_detected_at"],
                    last_lon=float(row["lon"]),
                    last_lat=float(row["lat"]),
                    estimated_heading_deg=row["estimated_heading_deg"],
                    estimated_velocity_ms=row["estimated_velocity_ms"],
                    confidence_score=float(row["confidence_score"] or 1.0),
                    last_reid_embedding=_pg_array_to_embedding(row.get("last_reid_embedding")),
                )
            )
        return rows

    async def _next_track_code(self, session: AsyncSession, when: datetime) -> str:
        day = when.strftime("%Y%m%d")
        prefix = f"TRK-{day}-"
        sql = text(
            """
            SELECT track_code FROM entity_tracks
            WHERE track_code LIKE :pattern
            ORDER BY track_code DESC
            LIMIT 1
            """
        )
        result = await session.execute(sql, {"pattern": f"{prefix}%"})
        row = result.first()
        if not row:
            return f"{prefix}001"
        last_code = row[0]
        try:
            seq = int(last_code.split("-")[-1]) + 1
        except ValueError:
            seq = 1
        return f"{prefix}{seq:03d}"

    async def _friction_for_pair(
        self,
        track: ActiveTrackRow,
        detection: DetectionAssociationInput,
    ) -> Tuple[np.ndarray, float, float]:
        mid_lon = (track.last_lon + detection.detection_lon) / 2.0
        mid_lat = (track.last_lat + detection.detection_lat) / 2.0
        delta_t = (detection.detection_time - track.last_detected_at).total_seconds()
        max_speed = 35.0
        radius = max(500.0, min(3000.0, max_speed * max(delta_t, 60.0)))
        try:
            friction, _, side = await self._spatial.get_friction_surface_window(
                mid_lon,
                mid_lat,
                radius,
                detection.object_class,
            )
            return friction, mid_lon, mid_lat
        except Exception as exc:
            logger.debug("Friction window fallback: %s", exc)
            side = max(31, int(2 * radius / 10.0) + 1)
            if side % 2 == 0:
                side += 1
            friction = build_local_friction_window(side, detection.object_class)
            return friction, mid_lon, mid_lat

    async def associate(
        self,
        detection: DetectionAssociationInput,
        friction_override: Optional[np.ndarray] = None,
        friction_center: Optional[PointLonLat] = None,
    ) -> TrackState:
        async with self._session_factory() as session:
            tracks = await self.fetch_active_tracks(session)
            best_track: Optional[ActiveTrackRow] = None
            best_score = 0.0

            for track in tracks:
                gate = evaluate_spatiotemporal_gate(
                    track.last_lon,
                    track.last_lat,
                    track.last_detected_at,
                    detection.detection_lon,
                    detection.detection_lat,
                    detection.detection_time,
                    detection.object_class,
                )
                if not gate.reachable:
                    continue

                p_visual = visual_association_probability(
                    detection.object_class,
                    track.object_class,
                    detection.reid_embedding,
                    track.last_reid_embedding,
                )
                if p_visual <= 0:
                    continue

                if friction_override is not None:
                    center_lon, center_lat = friction_center or (
                        (track.last_lon + detection.detection_lon) / 2.0,
                        (track.last_lat + detection.detection_lat) / 2.0,
                    )
                    friction = friction_override
                else:
                    friction, center_lon, center_lat = await self._friction_for_pair(
                        track, detection
                    )

                score, gated = compute_association_score(
                    detection, track, friction, center_lon, center_lat
                )
                if gated and score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None and best_score >= TAU_ASSOCIATE:
                return await self._update_track(
                    session,
                    best_track,
                    detection,
                    association_score=best_score,
                    friction_override=friction_override,
                    friction_center=friction_center,
                )

            return await self._create_track(
                session,
                detection,
                friction_override=friction_override,
                friction_center=friction_center,
            )

    async def _update_track(
        self,
        session: AsyncSession,
        track: ActiveTrackRow,
        detection: DetectionAssociationInput,
        association_score: float,
        friction_override: Optional[np.ndarray] = None,
        friction_center: Optional[PointLonLat] = None,
    ) -> TrackState:
        gate = evaluate_spatiotemporal_gate(
            track.last_lon,
            track.last_lat,
            track.last_detected_at,
            detection.detection_lon,
            detection.detection_lat,
            detection.detection_time,
            detection.object_class,
        )
        heading = _bearing_deg(
            track.last_lon,
            track.last_lat,
            detection.detection_lon,
            detection.detection_lat,
        )
        velocity = gate.implied_speed_ms if gate.delta_t_seconds > 0 else track.estimated_velocity_ms

        geojson = await self._reachability_for_point(
            detection.detection_lon,
            detection.detection_lat,
            detection.detection_time,
            detection.detection_time,
            detection.object_class,
            track.track_id,
            track.track_code,
            detection.confidence,
            heading_deg=heading,
            friction_override=friction_override,
            friction_center=friction_center,
        )

        poly_wkt = _geojson_to_wkt(geojson) if geojson else None
        params = {
            "track_id": str(track.track_id),
            "last_detected_at": detection.detection_time,
            "sensor_id": str(detection.sensor_id) if detection.sensor_id else None,
            "lon": detection.detection_lon,
            "lat": detection.detection_lat,
            "prev_lon": track.last_lon,
            "prev_lat": track.last_lat,
            "heading": heading,
            "velocity": velocity,
            "confidence": min(1.0, max(detection.confidence, association_score)),
            "last_reid_embedding": detection.reid_embedding,
        }
        if poly_wkt:
            sql = text(
                """
                UPDATE entity_tracks
                SET
                    last_detected_at = :last_detected_at,
                    last_sensor_id = :sensor_id,
                    last_known_point = ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    estimated_heading_deg = :heading,
                    estimated_velocity_ms = :velocity,
                    confidence_score = :confidence,
                    last_reid_embedding = :last_reid_embedding,
                    history_linestring = CASE
                        WHEN history_linestring IS NULL THEN
                            ST_SetSRID(ST_MakeLine(
                                ST_MakePoint(:prev_lon, :prev_lat),
                                ST_MakePoint(:lon, :lat)
                            ), 4326)
                        ELSE ST_SetSRID(ST_AddPoint(
                            history_linestring::geometry,
                            ST_MakePoint(:lon, :lat)
                        ), 4326)
                    END,
                    reachability_polygon = ST_GeomFromText(:poly_wkt, 4326),
                    updated_at = NOW()
                WHERE id = :track_id
                RETURNING first_detected_at, track_code, object_class, current_status
                """
            )
            params["poly_wkt"] = poly_wkt
        else:
            sql = text(
                """
                UPDATE entity_tracks
                SET
                    last_detected_at = :last_detected_at,
                    last_sensor_id = :sensor_id,
                    last_known_point = ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    estimated_heading_deg = :heading,
                    estimated_velocity_ms = :velocity,
                    confidence_score = :confidence,
                    last_reid_embedding = :last_reid_embedding,
                    history_linestring = CASE
                        WHEN history_linestring IS NULL THEN
                            ST_SetSRID(ST_MakeLine(
                                ST_MakePoint(:prev_lon, :prev_lat),
                                ST_MakePoint(:lon, :lat)
                            ), 4326)
                        ELSE ST_SetSRID(ST_AddPoint(
                            history_linestring::geometry,
                            ST_MakePoint(:lon, :lat)
                        ), 4326)
                    END,
                    updated_at = NOW()
                WHERE id = :track_id
                RETURNING first_detected_at, track_code, object_class, current_status
                """
            )
        result = await session.execute(sql, params)
        await session.commit()
        row = result.mappings().first()
        return TrackState(
            track_id=track.track_id,
            track_code=row["track_code"],
            object_class=EntityClass(row["object_class"]),
            status=TrackStatus(row["current_status"]),
            first_detected_at=row["first_detected_at"],
            last_detected_at=detection.detection_time,
            last_known_point=(detection.detection_lon, detection.detection_lat),
            estimated_heading_deg=heading,
            estimated_velocity_ms=velocity,
            confidence_score=min(1.0, max(detection.confidence, association_score)),
            reachability_geojson=geojson,
            last_reid_embedding=detection.reid_embedding,
        )

    async def _create_track(
        self,
        session: AsyncSession,
        detection: DetectionAssociationInput,
        friction_override: Optional[np.ndarray] = None,
        friction_center: Optional[PointLonLat] = None,
    ) -> TrackState:
        track_id = uuid4()
        track_code = await self._next_track_code(session, detection.detection_time)
        geojson = await self._reachability_for_point(
            detection.detection_lon,
            detection.detection_lat,
            detection.detection_time,
            detection.detection_time,
            detection.object_class,
            track_id,
            track_code,
            detection.confidence,
            heading_deg=None,
            friction_override=friction_override,
            friction_center=friction_center,
        )
        poly_wkt = _geojson_to_wkt(geojson) if geojson else None

        base_params = {
            "id": str(track_id),
            "track_code": track_code,
            "object_class": detection.object_class.value,
            "first_detected_at": detection.detection_time,
            "last_detected_at": detection.detection_time,
            "sensor_id": str(detection.sensor_id) if detection.sensor_id else None,
            "lon": detection.detection_lon,
            "lat": detection.detection_lat,
            "confidence": detection.confidence,
            "last_reid_embedding": detection.reid_embedding,
        }
        if poly_wkt:
            sql = text(
                """
                INSERT INTO entity_tracks (
                    id, track_code, object_class, current_status,
                    first_detected_at, last_detected_at, last_sensor_id,
                    last_known_point, estimated_heading_deg, estimated_velocity_ms,
                    history_linestring, reachability_polygon, confidence_score,
                    last_reid_embedding
                ) VALUES (
                    :id, :track_code, :object_class, 'ACTIVE',
                    :first_detected_at, :last_detected_at, :sensor_id,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    NULL, NULL,
                    ST_SetSRID(ST_MakeLine(ST_MakePoint(:lon, :lat), ST_MakePoint(:lon, :lat)), 4326),
                    ST_GeomFromText(:poly_wkt, 4326),
                    :confidence,
                    :last_reid_embedding
                )
                """
            )
            base_params["poly_wkt"] = poly_wkt
        else:
            sql = text(
                """
                INSERT INTO entity_tracks (
                    id, track_code, object_class, current_status,
                    first_detected_at, last_detected_at, last_sensor_id,
                    last_known_point, estimated_heading_deg, estimated_velocity_ms,
                    history_linestring, confidence_score, last_reid_embedding
                ) VALUES (
                    :id, :track_code, :object_class, 'ACTIVE',
                    :first_detected_at, :last_detected_at, :sensor_id,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    NULL, NULL,
                    ST_SetSRID(ST_MakeLine(ST_MakePoint(:lon, :lat), ST_MakePoint(:lon, :lat)), 4326),
                    :confidence,
                    :last_reid_embedding
                )
                """
            )
        await session.execute(sql, base_params)
        await session.commit()

        return TrackState(
            track_id=track_id,
            track_code=track_code,
            object_class=detection.object_class,
            status=TrackStatus.ACTIVE,
            first_detected_at=detection.detection_time,
            last_detected_at=detection.detection_time,
            last_known_point=(detection.detection_lon, detection.detection_lat),
            confidence_score=detection.confidence,
            reachability_geojson=geojson,
            last_reid_embedding=detection.reid_embedding,
        )

    async def _reachability_for_point(
        self,
        lon: float,
        lat: float,
        last_detected_at: datetime,
        now: datetime,
        entity_class: EntityClass,
        track_id: UUID,
        track_code: str,
        confidence: float,
        heading_deg: Optional[float],
        friction_override: Optional[np.ndarray] = None,
        friction_center: Optional[PointLonLat] = None,
    ) -> Optional[Dict[str, Any]]:
        elapsed = max(0.0, (now - last_detected_at).total_seconds())
        if friction_override is not None:
            friction = friction_override
            center_lon, center_lat = friction_center or (lon, lat)
        else:
            try:
                friction, transform, _ = await self._spatial.get_friction_surface_window(
                    lon, lat, 500.0, entity_class
                )
                center_lon, center_lat = lon, lat
            except Exception:
                friction = build_local_friction_window(51, entity_class)
                center_lon, center_lat = lon, lat

        return compute_reachability_geojson(
            friction,
            center_lon,
            center_lat,
            max(elapsed, 1.0),
            entity_class,
            track_id,
            track_code,
            confidence=confidence,
            heading_deg=heading_deg,
        )


def _pg_array_to_embedding(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(x) for x in value]


def _geojson_to_wkt(geojson: Dict[str, Any]) -> Optional[str]:
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    geometry = geojson.get("geometry")
    if not geometry:
        return None
    geom = shapely_shape(geometry)
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = unary_union(geom)
    if geom.geom_type != "Polygon":
        geom = geom.convex_hull
    gtype = "Polygon"
    coords = [[[float(x), float(y)] for x, y in geom.exterior.coords]]
    if not coords:
        return None
    rings = []
    for ring in coords:
        ring_str = ", ".join(f"{pt[0]} {pt[1]}" for pt in ring)
        rings.append(f"({ring_str})")
    return f"POLYGON({', '.join(rings)})"
