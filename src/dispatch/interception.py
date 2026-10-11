"""
Tactical interception geometry: compare target wavefront arrival at chokepoints
against interceptor Dijkstra transit times (spec 04 §3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.database import async_session_factory
from src.common.schemas import EntityClass, TacticalDirective
from src.spatial.routing import find_shortest_route
from src.spatial.service import SpatialService
from src.tracking.isochrone import (
    ISOCHRONE_CELL_SIZE_M,
    lonlat_to_grid_cell,
    solve_travel_time_grid,
)
from src.tracking.constants import CLASS_HEADING_KAPPA

logger = logging.getLogger(__name__)

TACTICAL_MARGIN_SECONDS = 180.0
ISOCHRONE_RADIUS_M = 2500.0


@dataclass
class _TrackRow:
    track_id: UUID
    track_code: str
    object_class: EntityClass
    last_known_point: Tuple[float, float]
    estimated_heading_deg: Optional[float]
    estimated_velocity_ms: Optional[float]
    last_detected_at: datetime


@dataclass
class _ChokepointEval:
    chokepoint_id: UUID
    name: str
    location: Tuple[float, float]
    target_arrival_s: float
    interceptor_arrival_s: float
    margin_s: float
    route_coordinates: List[Tuple[float, float]]


class InterceptionSolver:
    """Selects an actionable chokepoint and persists a tactical directive."""

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        spatial_service: Optional[SpatialService] = None,
        margin_seconds: float = TACTICAL_MARGIN_SECONDS,
    ) -> None:
        self._session = session
        self._spatial = spatial_service or SpatialService(session)
        self.margin_seconds = margin_seconds

    async def _get_session(self) -> Tuple[AsyncSession, bool]:
        if self._session is not None:
            return self._session, False
        return async_session_factory(), True

    async def load_track(self, track_id: UUID) -> Optional[_TrackRow]:
        session, own = await self._get_session()
        try:
            sql = text(
                """
                SELECT
                    id,
                    track_code,
                    object_class,
                    ST_X(last_known_point) AS lon,
                    ST_Y(last_known_point) AS lat,
                    estimated_heading_deg,
                    estimated_velocity_ms,
                    last_detected_at
                FROM entity_tracks
                WHERE id = :track_id AND current_status = 'ACTIVE'
                """
            )
            result = await session.execute(sql, {"track_id": str(track_id)})
            row = result.mappings().first()
            if not row:
                return None
            return _TrackRow(
                track_id=UUID(str(row["id"])),
                track_code=row["track_code"],
                object_class=EntityClass(row["object_class"]),
                last_known_point=(float(row["lon"]), float(row["lat"])),
                estimated_heading_deg=row["estimated_heading_deg"],
                estimated_velocity_ms=row["estimated_velocity_ms"],
                last_detected_at=row["last_detected_at"],
            )
        finally:
            if own:
                await session.close()

    async def target_travel_time_seconds(
        self,
        track: _TrackRow,
        chokepoint_lonlat: Tuple[float, float],
    ) -> float:
        """
        Eikonal arrival time from the track's last known point to the chokepoint
        on a local friction surface (wavefront / isochrone model).
        """
        lon, lat = track.last_known_point
        cp_lon, cp_lat = chokepoint_lonlat
        try:
            friction, _transform, side = await self._spatial.get_friction_surface_window(
                lon,
                lat,
                ISOCHRONE_RADIUS_M,
                track.object_class,
            )
        except Exception as exc:
            logger.debug("Friction window failed, using velocity fallback: %s", exc)
            return self._velocity_fallback_eta_s(track, chokepoint_lonlat)

        kappa = CLASS_HEADING_KAPPA.get(track.object_class, 1.0)
        times = solve_travel_time_grid(
            friction,
            cell_size_m=ISOCHRONE_CELL_SIZE_M,
            heading_deg=track.estimated_heading_deg,
            kappa=kappa,
        )
        center_r, center_c = lonlat_to_grid_cell(
            lon, lat, lon, lat, side, ISOCHRONE_CELL_SIZE_M
        )
        cp_r, cp_c = lonlat_to_grid_cell(
            lon, lat, cp_lon, cp_lat, side, ISOCHRONE_CELL_SIZE_M
        )
        t_from_center = float(times[cp_r, cp_c])
        if not (t_from_center < float("inf")):
            return self._velocity_fallback_eta_s(track, chokepoint_lonlat)
        t_center = float(times[center_r, center_c])
        return max(0.0, t_from_center - t_center)

    def _velocity_fallback_eta_s(
        self,
        track: _TrackRow,
        chokepoint_lonlat: Tuple[float, float],
    ) -> float:
        from geopy.distance import geodesic

        lon, lat = track.last_known_point
        cp_lon, cp_lat = chokepoint_lonlat
        distance_m = geodesic((lat, lon), (cp_lat, cp_lon)).meters
        v = track.estimated_velocity_ms
        if v is None or v <= 0:
            from src.common.schemas import CLASS_MAX_VELOCITY

            v = CLASS_MAX_VELOCITY[track.object_class] * 0.5
        return distance_m / v

    async def evaluate_chokepoints(
        self,
        track: _TrackRow,
        interceptor_position: Tuple[float, float],
        chokepoints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[_ChokepointEval]:
        if chokepoints is None:
            chokepoints = await self._spatial.get_chokepoints()

        evaluations: List[_ChokepointEval] = []
        for cp in chokepoints:
            if not cp.get("is_accessible_to_response", True):
                continue
            cp_id = cp["id"] if isinstance(cp["id"], UUID) else UUID(str(cp["id"]))
            location = cp["location"]
            target_s = await self.target_travel_time_seconds(track, location)
            route = await find_shortest_route(
                interceptor_position,
                location,
                travel_mode="vehicle",
                session=self._session,
            )
            resp_s = route.transit_time_s
            margin = target_s - resp_s
            evaluations.append(
                _ChokepointEval(
                    chokepoint_id=cp_id,
                    name=cp["name"],
                    location=location,
                    target_arrival_s=target_s,
                    interceptor_arrival_s=resp_s,
                    margin_s=margin,
                    route_coordinates=route.coordinates,
                )
            )
        return evaluations

    def select_best_actionable(
        self, evaluations: List[_ChokepointEval]
    ) -> Optional[_ChokepointEval]:
        actionable = [e for e in evaluations if e.margin_s >= self.margin_seconds]
        if not actionable:
            return None
        return min(actionable, key=lambda e: e.target_arrival_s)

    async def solve(
        self,
        track_id: UUID,
        interceptor_position: Tuple[float, float],
        reference_time: Optional[datetime] = None,
    ) -> Optional[TacticalDirective]:
        """
        Compute interception geometry, persist directive, and return the contract model.
        """
        reference_time = reference_time or datetime.now(timezone.utc)
        track = await self.load_track(track_id)
        if track is None:
            logger.warning("Track %s not found or not ACTIVE", track_id)
            return None

        evaluations = await self.evaluate_chokepoints(track, interceptor_position)
        best = self.select_best_actionable(evaluations)
        if best is None:
            logger.info("No actionable chokepoint for track %s", track.track_code)
            return None

        target_eta = reference_time + timedelta(seconds=best.target_arrival_s)
        interceptor_eta = reference_time + timedelta(seconds=best.interceptor_arrival_s)
        directive_id = await self._persist_directive(
            track_id=track.track_id,
            chokepoint_id=best.chokepoint_id,
            target_eta=target_eta,
            interceptor_eta=interceptor_eta,
            margin_s=best.margin_s,
            route_coordinates=best.route_coordinates,
        )

        action_message = (
            f"INTERCEPT AT {best.name} — margin {best.margin_s / 60.0:.1f} min"
        )
        return TacticalDirective(
            directive_id=directive_id,
            track_id=track.track_id,
            track_code=track.track_code,
            target_class=track.object_class,
            target_chokepoint_id=best.chokepoint_id,
            chokepoint_name=best.name,
            chokepoint_location=best.location,
            target_arrival_time=target_eta,
            interceptor_arrival_time=interceptor_eta,
            safety_margin_seconds=best.margin_s,
            action_message=action_message,
        )

    async def _persist_directive(
        self,
        track_id: UUID,
        chokepoint_id: UUID,
        target_eta: datetime,
        interceptor_eta: datetime,
        margin_s: float,
        route_coordinates: List[Tuple[float, float]],
    ) -> UUID:
        session, own = await self._get_session()
        try:
            payload = {
                "response_route": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in route_coordinates],
                },
                "status": "ISSUED",
            }
            sql = text(
                """
                INSERT INTO tactical_directives
                    (track_id, chokepoint_id, target_eta, interceptor_eta,
                     tactical_margin_seconds, dispatch_payload)
                VALUES
                    (:track_id, :chokepoint_id, :target_eta, :interceptor_eta,
                     :margin_s, CAST(:payload AS jsonb))
                RETURNING id
                """
            )
            result = await session.execute(
                sql,
                {
                    "track_id": str(track_id),
                    "chokepoint_id": str(chokepoint_id),
                    "target_eta": target_eta,
                    "interceptor_eta": interceptor_eta,
                    "margin_s": margin_s,
                    "payload": json.dumps(payload),
                },
            )
            await session.commit()
            row = result.first()
            return UUID(str(row[0]))
        finally:
            if own:
                await session.close()
