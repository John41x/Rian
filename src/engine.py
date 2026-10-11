"""
Rian Autonomous Spatial State Engine.

Orchestrates the full event-driven lifecycle:
  1. Edge Ingestion (aiosmtpd / HTTP Webhook)
  2. YOLOv11 Computer Vision inference
  3. Dynamic HMM Kinematic Track Association & Isochrone Wavefront Propagation
  4. Real-time Chokepoint Interception & Tactical Route Optimization
  5. Automated Twilio SMS/WhatsApp Directive Dispatch
  6. Live Common Operating Picture (COP) SSE state broadcasting
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.common.config import settings
from src.common.database import async_session_factory
from src.common.schemas import (
    DetectionEventProcessed,
    EntityClass,
    PointLonLat,
    TacticalDirective,
    TrackState,
    TrackStatus,
)
from src.dispatch.alerter import AlertResult, TwilioAlerter
from src.dispatch.interception import InterceptionSolver
from src.ingestion.inference import YOLOInferenceProcessor
from src.ingestion.queue import get_ingestion_queue
from src.ingestion.service import IngestionService
from src.ingestion.smtp_server import SMTPIngestionController
from src.spatial.service import SpatialService
from src.tracking.association import DetectionAssociationInput, TrackAssociator
from src.tracking.state_worker import TrackStateWorker

logger = logging.getLogger("rian.engine")

InterceptorPositionProvider = Callable[[], Optional[PointLonLat]]
CopBroadcastCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class RianEngine:
    """Unified autonomous coordinator for the Rian Spatial State Engine."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
        spatial_service: Optional[SpatialService] = None,
        inference: Optional[YOLOInferenceProcessor] = None,
        alerter: Optional[TwilioAlerter] = None,
        interceptor_position_provider: Optional[InterceptorPositionProvider] = None,
        cop_broadcast: Optional[CopBroadcastCallback] = None,
        enable_smtp: bool = True,
        enable_worker: bool = True,
        worker_interval_seconds: float = 15.0,
    ) -> None:
        self.session_factory = session_factory
        self.spatial = spatial_service or SpatialService()
        self.inference = inference or YOLOInferenceProcessor()
        self.alerter = alerter or TwilioAlerter()
        self.interceptor_position_provider = interceptor_position_provider
        self.cop_broadcast = cop_broadcast

        self.associator = TrackAssociator(
            session_factory=self.session_factory,
            spatial_service=self.spatial,
        )
        self.solver = InterceptionSolver(
            spatial_service=self.spatial,
        )
        self.ingestion = IngestionService(
            session_factory=self.session_factory,
            inference=self.inference,
        )
        self.ingestion.add_listener(self.handle_detection_event)

        self.enable_worker = enable_worker
        self.worker = (
            TrackStateWorker(
                session_factory=self.session_factory,
                spatial_service=self.spatial,
                on_track_update=self.handle_worker_track_update,
                interval_seconds=worker_interval_seconds,
            )
            if enable_worker
            else None
        )

        self.enable_smtp = enable_smtp
        self.smtp_controller: Optional[SMTPIngestionController] = None
        if enable_smtp:
            self.smtp_controller = SMTPIngestionController(
                host=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                queue=self.ingestion.queue,
            )

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def get_interceptor_position(self) -> Optional[PointLonLat]:
        if self.interceptor_position_provider:
            pos = self.interceptor_position_provider()
            if pos is not None:
                return pos
        # Default fallback to ranch center if unpositioned
        return (-119.8200, 36.7450)

    async def start(self) -> None:
        """Start all autonomous background services."""
        if self._running:
            return
        logger.info("Starting Rian Autonomous Spatial State Engine...")
        self._running = True

        await self.ingestion.start()
        if self.worker:
            self.worker.start()

        if self.smtp_controller:
            try:
                self.smtp_controller.start()
                logger.info(
                    "SMTP Ingestion listener active on %s:%d",
                    self.smtp_controller.host,
                    self.smtp_controller.port,
                )
            except Exception as exc:
                logger.warning("Could not bind SMTP listener: %s (continuing with webhook/queue)", exc)

        logger.info("Rian Engine online.")

    async def stop(self) -> None:
        """Gracefully stop all services."""
        if not self._running:
            return
        logger.info("Stopping Rian Engine...")
        self._running = False

        if self.smtp_controller:
            try:
                self.smtp_controller.stop()
            except Exception as exc:
                logger.debug("Error stopping SMTP controller: %s", exc)

        if self.worker:
            await self.worker.stop()

        await self.ingestion.stop()
        logger.info("Rian Engine stopped.")

    async def handle_detection_event(
        self,
        event: DetectionEventProcessed,
    ) -> List[TrackState]:
        """
        Processes a detection event from ingestion through association,
        interception calculation, and alert dispatch.
        """
        logger.info(
            "Engine processing event %s from %s (%d detections)",
            event.event_id,
            event.node_name,
            len(event.detections),
        )
        updated_tracks: List[TrackState] = []
        interceptor_pos = self.get_interceptor_position()

        for det in event.detections:
            assoc_input = DetectionAssociationInput(
                detection_lon=event.sensor_location[0],
                detection_lat=event.sensor_location[1],
                detection_time=event.event_timestamp,
                object_class=det.object_class,
                sensor_id=event.sensor_id,
                confidence=det.confidence,
                reid_embedding=det.reid_embedding,
            )

            try:
                track_state = await self.associator.associate(assoc_input)
                updated_tracks.append(track_state)
                logger.info(
                    "Associated detection to track %s (status=%s, class=%s)",
                    track_state.track_code,
                    track_state.status.value,
                    track_state.object_class.value,
                )

                try:
                    from src.ingestion.vlm_inspector import get_vlm_inspector
                    get_vlm_inspector().link_track_snapshot(track_state.track_id, event.event_id)
                except Exception:
                    pass

                if track_state.status == TrackStatus.ACTIVE and interceptor_pos:
                    directive = await self.solver.solve(
                        track_id=track_state.track_id,
                        interceptor_position=interceptor_pos,
                        reference_time=event.event_timestamp,
                    )
                    if directive is not None:
                        logger.info(
                            "Tactical directive issued for track %s at chokepoint %s (margin=%.1fs)",
                            track_state.track_code,
                            directive.chokepoint_name,
                            directive.safety_margin_seconds,
                        )
                        alert_res = await self.alerter.send_alert(directive)
                        logger.info("Alert dispatched: %s", alert_res)

                if self.cop_broadcast:
                    await self.cop_broadcast(
                        {
                            "type": "detection_associated",
                            "track_id": str(track_state.track_id),
                            "track_code": track_state.track_code,
                            "object_class": track_state.object_class.value,
                            "status": track_state.status.value,
                            "last_known_point": list(track_state.last_known_point),
                        }
                    )
            except Exception:
                logger.exception("Error associating detection for %s", det.object_class)

        return updated_tracks

    async def handle_worker_track_update(self, track_state: TrackState) -> None:
        """Callback from TrackStateWorker when reachability polygons expand or status changes."""
        logger.debug(
            "Worker updated track %s (status=%s)",
            track_state.track_code,
            track_state.status.value,
        )
        if track_state.status == TrackStatus.ACTIVE:
            interceptor_pos = self.get_interceptor_position()
            if interceptor_pos:
                try:
                    directive = await self.solver.solve(
                        track_id=track_state.track_id,
                        interceptor_position=interceptor_pos,
                    )
                    if directive is not None:
                        await self.alerter.send_alert(directive)
                except Exception:
                    logger.exception("Worker error solving interception for %s", track_state.track_code)

        if self.cop_broadcast:
            try:
                await self.cop_broadcast(
                    {
                        "type": "track_wavefront_update",
                        "track_id": str(track_state.track_id),
                        "status": track_state.status.value,
                    }
                )
            except Exception:
                logger.debug("Failed broadcasting worker update")


_engine_instance: Optional[RianEngine] = None


def get_engine() -> RianEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RianEngine()
    return _engine_instance
