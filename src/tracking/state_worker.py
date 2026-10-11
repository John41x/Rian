"""Periodic track state daemon (spec §5)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.common.database import async_session_factory
from src.common.schemas import EntityClass, TrackState, TrackStatus
from src.spatial.service import SpatialService
from src.tracking.association import TrackAssociator, _geojson_to_wkt
from src.tracking.constants import STALE_THRESHOLD_SECONDS, WORKER_INTERVAL_SECONDS
logger = logging.getLogger(__name__)

TrackUpdateCallback = Callable[[TrackState], Awaitable[None]]


class TrackStateWorker:
    """
    Advances reachability polygons for ACTIVE tracks and marks stale tracks.

    Runs on a 15-second cadence; publishes ``TrackState`` updates via optional callback.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
        spatial_service: Optional[SpatialService] = None,
        on_track_update: Optional[TrackUpdateCallback] = None,
        interval_seconds: float = WORKER_INTERVAL_SECONDS,
        stale_threshold_seconds: float = STALE_THRESHOLD_SECONDS,
    ) -> None:
        self._session_factory = session_factory
        self._spatial = spatial_service or SpatialService()
        self._associator = TrackAssociator(session_factory, self._spatial)
        self._on_track_update = on_track_update
        self._interval = interval_seconds
        self._stale_threshold = stale_threshold_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def subscribe(self, callback: TrackUpdateCallback) -> None:
        self._on_track_update = callback

    async def tick(self, now: Optional[datetime] = None) -> List[TrackState]:
        """Single worker iteration."""
        now = now or datetime.now(timezone.utc)
        updated: List[TrackState] = []

        async with self._session_factory() as session:
            sql = text(
                """
                SELECT
                    id,
                    track_code,
                    object_class,
                    current_status,
                    first_detected_at,
                    last_detected_at,
                    ST_X(last_known_point::geometry) AS lon,
                    ST_Y(last_known_point::geometry) AS lat,
                    estimated_heading_deg,
                    estimated_velocity_ms,
                    confidence_score
                FROM entity_tracks
                WHERE current_status = 'ACTIVE'
                """
            )
            result = await session.execute(sql)
            rows = list(result.mappings())

            for row in rows:
                track_id = row["id"] if isinstance(row["id"], UUID) else UUID(str(row["id"]))
                last_detected = row["last_detected_at"]
                if last_detected.tzinfo is None:
                    last_detected = last_detected.replace(tzinfo=timezone.utc)
                elapsed = (now - last_detected).total_seconds()

                if elapsed > self._stale_threshold:
                    await session.execute(
                        text(
                            """
                            UPDATE entity_tracks
                            SET current_status = 'STALE', updated_at = NOW()
                            WHERE id = :id
                            """
                        ),
                        {"id": str(track_id)},
                    )
                    state = TrackState(
                        track_id=track_id,
                        track_code=row["track_code"],
                        object_class=EntityClass(row["object_class"]),
                        status=TrackStatus.STALE,
                        first_detected_at=row["first_detected_at"],
                        last_detected_at=last_detected,
                        last_known_point=(float(row["lon"]), float(row["lat"])),
                        estimated_heading_deg=row["estimated_heading_deg"],
                        estimated_velocity_ms=row["estimated_velocity_ms"],
                        confidence_score=float(row["confidence_score"] or 1.0),
                    )
                    updated.append(state)
                    continue

                lon, lat = float(row["lon"]), float(row["lat"])
                entity_class = EntityClass(row["object_class"])
                geojson = await self._associator._reachability_for_point(
                    lon,
                    lat,
                    last_detected,
                    now,
                    entity_class,
                    track_id,
                    row["track_code"],
                    float(row["confidence_score"] or 1.0),
                    row["estimated_heading_deg"],
                )
                poly_wkt = _geojson_to_wkt(geojson) if geojson else None
                if poly_wkt:
                    await session.execute(
                        text(
                            """
                            UPDATE entity_tracks
                            SET reachability_polygon = ST_GeomFromText(:wkt, 4326),
                                updated_at = NOW()
                            WHERE id = :id
                            """
                        ),
                        {"id": str(track_id), "wkt": poly_wkt},
                    )

                state = TrackState(
                    track_id=track_id,
                    track_code=row["track_code"],
                    object_class=entity_class,
                    status=TrackStatus.ACTIVE,
                    first_detected_at=row["first_detected_at"],
                    last_detected_at=last_detected,
                    last_known_point=(lon, lat),
                    estimated_heading_deg=row["estimated_heading_deg"],
                    estimated_velocity_ms=row["estimated_velocity_ms"],
                    confidence_score=float(row["confidence_score"] or 1.0),
                    reachability_geojson=geojson,
                )
                updated.append(state)

            await session.commit()

        for state in updated:
            if self._on_track_update:
                await self._on_track_update(state)

        return updated

    async def run(self) -> None:
        """Run until ``stop()`` is called."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                await self.tick()
            except Exception as exc:
                logger.exception("Track state worker tick failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = asyncio.create_task(self.run())
        return self._task

    def stop(self) -> None:
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
