"""
FastAPI Common Operating Picture (COP) dashboard and REST/SSE APIs (spec 04 §5).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.database import async_session_factory
from src.common.schemas import PointLonLat
from src.ingestion.webhook_server import router as webhook_router
from src.web.auth_routes import router as auth_router
from src.spatial.routing import find_shortest_route, RouteResult
from src.spatial.service import SpatialService

logger = logging.getLogger(__name__)

# In-memory interceptor telemetry and directive status overlays (no DDL changes).
_interceptor_position: Optional[PointLonLat] = None
_interceptor_routes: Dict[str, RouteResult] = {}
_directive_status: Dict[str, str] = {}
_sse_subscribers: List[asyncio.Queue] = []


class InterceptorTelemetry(BaseModel):
    lon: float
    lat: float
    unit_id: Optional[str] = "default"


class DirectiveStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(ACKNOWLEDGED|ON_SCENE|RESOLVED|ISSUED)$")


class CopState(BaseModel):
    sensors: List[Dict[str, Any]]
    tracks: List[Dict[str, Any]]
    reachability: List[Dict[str, Any]]
    chokepoints: List[Dict[str, Any]]
    directives: List[Dict[str, Any]]
    interceptor: Optional[Dict[str, Any]] = None
    generated_at: datetime


async def _init_auth_storage() -> None:
    from src.common.auth import bootstrap_initial_admin, ensure_users_table

    async with async_session_factory() as session:
        await ensure_users_table(session)
        await bootstrap_initial_admin(session)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_auth_storage()
    if os.environ.get("RIAN_START_ENGINE", "0") == "1":
        from src.engine import get_engine

        engine = get_engine()
        engine.interceptor_position_provider = lambda: _interceptor_position
        engine.cop_broadcast = _broadcast_sse
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()
    else:
        yield


def create_app() -> FastAPI:
    application = FastAPI(title="Rian COP", version="1.0.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(webhook_router)
    application.include_router(auth_router)
    from src.web.camera_routes import router as camera_router

    application.include_router(camera_router)
    from src.web.sla_service import sla_router

    application.include_router(sla_router)
    from src.web.zone_routes import router as zone_router

    application.include_router(zone_router)
    from src.web.snapshot_routes import router as snapshot_router

    application.include_router(snapshot_router)
    from src.web.predator_routes import router as predator_router

    application.include_router(predator_router)
    from src.web.rollcall_routes import router as rollcall_router

    application.include_router(rollcall_router)

    @application.get("/", response_class=HTMLResponse)
    @application.get("/cop", response_class=HTMLResponse)
    async def cop_dashboard() -> str:
        return _COP_HTML

    @application.get("/api/v1/cop/state", response_model=CopState)
    async def cop_state() -> CopState:
        return await build_cop_state()

    @application.get("/api/v1/cop/stream")
    async def cop_stream(max_events: Optional[int] = None) -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        _sse_subscribers.append(queue)

        async def event_generator() -> AsyncGenerator[bytes, None]:
            try:
                initial = await build_cop_state()
                yield _sse_pack("cop_state", initial.model_dump(mode="json"))
                emitted = 1
                if max_events is not None and emitted >= max_events:
                    return
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield _sse_pack("cop_update", payload)
                        emitted += 1
                        if max_events is not None and emitted >= max_events:
                            return
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
            finally:
                if queue in _sse_subscribers:
                    _sse_subscribers.remove(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @application.post("/api/v1/interceptor/telemetry")
    async def interceptor_telemetry(body: InterceptorTelemetry) -> Dict[str, Any]:
        global _interceptor_position
        _interceptor_position = (body.lon, body.lat)
        routes: Dict[str, Any] = {}
        spatial = SpatialService()
        chokepoints = await spatial.get_chokepoints()
        for cp in chokepoints:
            if not cp.get("is_accessible_to_response", True):
                continue
            cp_id = str(cp["id"])
            route = await find_shortest_route(
                _interceptor_position,
                cp["location"],
                travel_mode="vehicle",
            )
            _interceptor_routes[cp_id] = route
            routes[cp_id] = {
                "chokepoint_name": cp["name"],
                "transit_time_s": route.transit_time_s,
                "distance_m": route.distance_m,
                "geojson": {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[c[0], c[1]] for c in route.coordinates],
                    },
                    "properties": {"chokepoint_id": cp_id},
                },
            }
        payload = {
            "interceptor": {"lon": body.lon, "lat": body.lat, "unit_id": body.unit_id},
            "routes": routes,
        }
        await _broadcast_sse(payload)
        return payload

    @application.post("/api/v1/directives/{directive_id}/status")
    async def update_directive_status(
        directive_id: UUID,
        body: DirectiveStatusUpdate,
    ) -> Dict[str, Any]:
        async with async_session_factory() as session:
            row = await session.execute(
                text(
                    """
                    SELECT dispatch_payload
                    FROM tactical_directives
                    WHERE id = :id
                    """
                ),
                {"id": str(directive_id)},
            )
            found = row.first()
            if not found:
                raise HTTPException(status_code=404, detail="Directive not found")

            payload = found[0] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload["status"] = body.status
            await session.execute(
                text(
                    """
                    UPDATE tactical_directives
                    SET dispatch_payload = CAST(:payload AS jsonb)
                    WHERE id = :id
                    """
                ),
                {"id": str(directive_id), "payload": json.dumps(payload)},
            )
            await session.commit()

        _directive_status[str(directive_id)] = body.status
        await _broadcast_sse({"directive_id": str(directive_id), "status": body.status})
        return {"directive_id": str(directive_id), "status": body.status}

    @application.post("/api/v1/sim/run-scenario")
    async def run_scenario_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
        scenario_name = body.get("name", "vehicle_intrusion")
        from tools.scenarios import SCENARIOS

        if scenario_name not in SCENARIOS:
            raise HTTPException(status_code=400, detail=f"Unknown scenario {scenario_name}")
        res = SCENARIOS[scenario_name](emit_smtp=False)
        updated_state = await build_cop_state()
        await _broadcast_sse(
            {"type": "scenario_executed", "scenario": scenario_name, "state": updated_state.model_dump(mode="json")}
        )
        return {"status": "success", "result": res}

    @application.post("/api/v1/sim/reset")
    async def reset_simulation() -> Dict[str, str]:
        async with async_session_factory() as session:
            try:
                await session.execute(text("DELETE FROM tactical_directives"))
                await session.execute(text("DELETE FROM entity_tracks"))
                await session.commit()
            except Exception as e:
                logger.warning("Error clearing test tracks: %s", e)
        global _interceptor_position, _interceptor_routes, _directive_status
        _interceptor_routes.clear()
        _directive_status.clear()
        updated_state = await build_cop_state()
        await _broadcast_sse({"type": "simulation_reset", "state": updated_state.model_dump(mode="json")})
        return {"status": "reset_complete"}

    @application.post("/api/v1/sim/inject")
    async def inject_detection_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
        camera_name = body.get("camera_name", "cam-gate-north")
        entity_class_str = body.get("entity_class", "truck")
        from uuid import uuid4
        from src.engine import get_engine
        from src.common.schemas import BoundingBox, DetectionEventProcessed, DetectionItem, EntityClass
        from src.ingestion.repository import resolve_sensor_by_email
        from tools.synthetic_gis import SENSOR_NODES

        target_email = None
        for cam in SENSOR_NODES:
            if cam.node_name == camera_name:
                target_email = cam.sender_email
                break
        if not target_email:
            target_email = f"{camera_name}@ranch.domain"

        async with async_session_factory() as session:
            sensor = await resolve_sensor_by_email(session, target_email)
        if not sensor:
            raise HTTPException(status_code=404, detail=f"Camera {camera_name} not found in DB")

        try:
            e_class = EntityClass(entity_class_str.lower())
        except ValueError:
            e_class = EntityClass.TRUCK

        now = datetime.now(timezone.utc)
        event = DetectionEventProcessed(
            event_id=uuid4(),
            sensor_id=sensor.id,
            node_name=sensor.node_name,
            sensor_location=sensor.location,
            sensor_azimuth_deg=sensor.azimuth_deg,
            event_timestamp=now,
            detections=[
                DetectionItem(
                    object_class=e_class,
                    confidence=0.94,
                    bounding_box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
                )
            ],
            raw_image_path="simulated_injection.jpg",
            processed_latency_ms=25,
        )

        engine = get_engine()
        engine.cop_broadcast = _broadcast_sse
        updated_tracks = await engine.handle_detection_event(event)
        updated_state = await build_cop_state()
        await _broadcast_sse({"type": "sim_injected", "state": updated_state.model_dump(mode="json")})
        return {"status": "injected", "tracks": [t.track_code for t in updated_tracks]}

    @application.post("/api/v1/cop/ai-sitrep")
    async def ai_sitrep_endpoint() -> Dict[str, Any]:
        from src.dispatch.llm_copilot import TacticalCopilot
        state = await build_cop_state()
        copilot = TacticalCopilot()
        return await copilot.generate_sitrep(state.model_dump(mode="json"))

    @application.post("/api/v1/cop/ai-query")
    async def ai_query_endpoint(body: Dict[str, str]) -> Dict[str, Any]:
        query = body.get("query", "")
        from src.dispatch.llm_copilot import TacticalCopilot
        state = await build_cop_state()
        copilot = TacticalCopilot()
        answer = await copilot.answer_operator_query(query, state.model_dump(mode="json"))
        return {"query": query, "response": answer, "provider": copilot.provider}

    @application.post("/api/v1/cop/entity/{track_id}/insight")
    async def entity_insight_endpoint(track_id: str):
        from src.dispatch.llm_copilot import EntityTrackNotFoundError, TacticalCopilot

        state = await build_cop_state()
        copilot = TacticalCopilot()
        try:
            return await copilot.generate_entity_insight(
                track_id, state.model_dump(mode="json")
            )
        except EntityTrackNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return application


app = create_app()


def _sse_pack(event: str, data: Any) -> bytes:
    body = json.dumps(data, default=str)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


async def _broadcast_sse(payload: Dict[str, Any]) -> None:
    for queue in list(_sse_subscribers):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def build_cop_state() -> CopState:
    async with async_session_factory() as session:
        sensors = await _fetch_sensors(session)
        tracks, reachability = await _fetch_tracks(session)
        chokepoints = await _fetch_chokepoints(session)
        directives = await _fetch_directives(session)

    interceptor: Optional[Dict[str, Any]] = None
    if _interceptor_position is not None:
        interceptor = {
            "location": list(_interceptor_position),
            "routes": {
                cp_id: {
                    "transit_time_s": r.transit_time_s,
                    "coordinates": r.coordinates,
                }
                for cp_id, r in _interceptor_routes.items()
            },
        }

    return CopState(
        sensors=sensors,
        tracks=tracks,
        reachability=reachability,
        chokepoints=chokepoints,
        directives=directives,
        interceptor=interceptor,
        generated_at=datetime.now(timezone.utc),
    )


async def _fetch_sensors(session: AsyncSession) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT
            id,
            node_name,
            ST_X(location) AS lon,
            ST_Y(location) AS lat,
            azimuth_deg,
            fov_angle,
            is_active,
            battery_level,
            last_heartbeat
        FROM sensor_nodes
        ORDER BY node_name
        """
    )
    try:
        result = await session.execute(sql)
    except Exception:
        return []
    rows = []
    for row in result.mappings():
        active = bool(row["is_active"])
        rows.append(
            {
                "id": str(row["id"]),
                "node_name": row["node_name"],
                "location": [float(row["lon"]), float(row["lat"])],
                "azimuth_deg": float(row["azimuth_deg"]),
                "fov_angle": float(row["fov_angle"] or 85.0),
                "status": "active" if active else "offline",
                "battery_level": row["battery_level"],
                "fov_wedge": _fov_wedge_geojson(
                    float(row["lon"]),
                    float(row["lat"]),
                    float(row["azimuth_deg"]),
                    float(row["fov_angle"] or 85.0),
                    25.0,
                ),
            }
        )
    return rows


def _fov_wedge_geojson(
    lon: float,
    lat: float,
    azimuth_deg: float,
    fov_deg: float,
    radius_m: float,
) -> Dict[str, Any]:
    import math

    points: List[List[float]] = [[lon, lat]]
    half = fov_deg / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat))
    for bearing in (azimuth_deg - half, azimuth_deg + half):
        br = math.radians(bearing)
        east = radius_m * math.sin(br)
        north = radius_m * math.cos(br)
        points.append(
            [
                lon + east / m_per_deg_lon,
                lat + north / m_per_deg_lat,
            ]
        )
    points.append([lon, lat])
    return {"type": "Polygon", "coordinates": [points]}


async def _fetch_tracks(
    session: AsyncSession,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sql = text(
        """
        SELECT
            t.id,
            t.track_code,
            t.object_class,
            t.current_status,
            ST_X(t.last_known_point) AS lon,
            ST_Y(t.last_known_point) AS lat,
            t.estimated_heading_deg,
            t.estimated_velocity_ms,
            t.confidence_score,
            t.last_detected_at,
            sn.node_name AS last_camera_name,
            ST_AsGeoJSON(t.history_linestring) AS history_json,
            ST_AsGeoJSON(t.reachability_polygon) AS reach_json
        FROM entity_tracks t
        LEFT JOIN sensor_nodes sn ON sn.id = t.last_sensor_id
        WHERE t.current_status = 'ACTIVE'
        """
    )
    try:
        result = await session.execute(sql)
    except Exception:
        return [], []

    tracks: List[Dict[str, Any]] = []
    reachability: List[Dict[str, Any]] = []
    for row in result.mappings():
        tid = str(row["id"])
        history = json.loads(row["history_json"]) if row["history_json"] else None
        tracks.append(
            {
                "track_id": tid,
                "track_code": row["track_code"],
                "object_class": row["object_class"],
                "status": row["current_status"],
                "last_known_point": [float(row["lon"]), float(row["lat"])],
                "estimated_heading_deg": row["estimated_heading_deg"],
                "estimated_velocity_ms": row["estimated_velocity_ms"],
                "confidence_score": row["confidence_score"],
                "last_detected_at": (
                    row["last_detected_at"].isoformat()
                    if row["last_detected_at"] is not None
                    else None
                ),
                "last_camera_name": row["last_camera_name"],
                "history": history,
            }
        )
        if row["reach_json"]:
            reach_geom = json.loads(row["reach_json"])
            reachability.append(
                {
                    "type": "Feature",
                    "geometry": reach_geom,
                    "properties": {
                        "track_id": tid,
                        "track_code": row["track_code"],
                        "target_class": row["object_class"],
                    },
                }
            )
    return tracks, reachability


async def _fetch_chokepoints(session: AsyncSession) -> List[Dict[str, Any]]:
    spatial = SpatialService(session)
    cps = await spatial.get_chokepoints()
    return [
        {
            "id": str(cp["id"]),
            "name": cp["name"],
            "chokepoint_type": cp.get("chokepoint_type"),
            "location": list(cp["location"]),
            "is_accessible_to_response": cp.get("is_accessible_to_response", True),
        }
        for cp in cps
    ]


async def _fetch_directives(session: AsyncSession) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT
            d.id,
            d.track_id,
            d.chokepoint_id,
            d.target_eta,
            d.interceptor_eta,
            d.tactical_margin_seconds,
            d.notification_dispatched,
            d.dispatch_payload,
            t.track_code,
            c.name AS chokepoint_name,
            ST_X(c.location) AS cp_lon,
            ST_Y(c.location) AS cp_lat
        FROM tactical_directives d
        JOIN entity_tracks t ON t.id = d.track_id
        JOIN strategic_chokepoints c ON c.id = d.chokepoint_id
        ORDER BY d.created_at DESC
        LIMIT 50
        """
    )
    try:
        result = await session.execute(sql)
    except Exception:
        return []

    directives: List[Dict[str, Any]] = []
    for row in result.mappings():
        payload = row["dispatch_payload"] or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        did = str(row["id"])
        status = _directive_status.get(did) or payload.get("status", "ISSUED")
        directives.append(
            {
                "directive_id": did,
                "track_id": str(row["track_id"]),
                "track_code": row["track_code"],
                "chokepoint_id": str(row["chokepoint_id"]),
                "chokepoint_name": row["chokepoint_name"],
                "chokepoint_location": [float(row["cp_lon"]), float(row["cp_lat"])],
                "target_eta": row["target_eta"].isoformat(),
                "interceptor_eta": row["interceptor_eta"].isoformat(),
                "tactical_margin_seconds": float(row["tactical_margin_seconds"]),
                "notification_dispatched": row["notification_dispatched"],
                "status": status,
                "response_route": payload.get("response_route"),
            }
        )
    return directives


_COP_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <title>Rian COP // Autonomous Spatial State Engine</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    body, html { font-family: 'Inter', sans-serif; height: 100%; margin: 0; background: #030712; color: #f3f4f6; overflow: hidden; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    #map { height: 100%; width: 100%; z-index: 1; }
    .glass-panel { background: rgba(10, 15, 30, 0.90); backdrop-filter: blur(14px); border: 1px solid rgba(56, 189, 248, 0.22); }
    .pulse-target { animation: pulseRing 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite; }
    @keyframes pulseRing {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { transform: scale(1.05); box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .leaflet-popup-content-wrapper { background: rgba(15, 23, 42, 0.95) !important; color: #f8fafc !important; border: 1px solid #38bdf8; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .leaflet-popup-tip { background: #0f172a !important; }
    .dark-tiles { filter: brightness(0.65) invert(1) contrast(2.8) hue-rotate(200deg) saturate(0.35) brightness(0.75); }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.6); }
    ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.3); border-radius: 4px; }
    .modal-overlay { background: rgba(3, 7, 18, 0.82); backdrop-filter: blur(8px); }
    .admin-glass { background: rgba(8, 12, 28, 0.92); backdrop-filter: blur(18px); border: 1px solid rgba(56, 189, 248, 0.28); }
    .admin-tab-btn.active { background: rgba(8, 145, 178, 0.35); color: #67e8f9; border-color: rgba(34, 211, 238, 0.45); }
    #login-modal.open, #admin-modal.open, #rollcall-modal.open { display: flex; }
    #login-modal, #admin-modal, #rollcall-modal { display: none; }
    .rollcall-status-verified { background: rgba(6, 78, 59, 0.55); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.45); }
    .rollcall-status-overdue { background: rgba(120, 53, 15, 0.55); color: #fcd34d; border: 1px solid rgba(251, 191, 36, 0.45); }
    .rollcall-status-missing { background: rgba(127, 29, 29, 0.6); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.5); }
    .compass-arrow { display: inline-block; transition: transform 0.12s ease-out; transform-origin: center center; }
    .map-pick-cursor { cursor: crosshair !important; }
    .map-draw-zone-cursor { cursor: crosshair !important; }
    #draw-zone-card { display: none; }
    #draw-zone-card.open { display: block; }
    .blinking-entity-dot {
      width: 14px; height: 14px; border-radius: 50%;
      background: radial-gradient(circle, #fecaca 0%, #ef4444 55%, #991b1b 100%);
      border: 2px solid #fca5a5;
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.85);
      animation: entityPulse 1.2s ease-in-out infinite;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px; line-height: 1;
    }
    @keyframes entityPulse {
      0%, 100% { transform: scale(0.92); opacity: 1; }
      50% { transform: scale(1.15); opacity: 0.82; }
    }
    .uncertainty-halo-svg {
      animation: haloBreath 2.4s ease-in-out infinite;
      pointer-events: none;
    }
    #predator-alert-banner {
      background: linear-gradient(90deg, rgba(127,29,29,0.95), rgba(153,27,27,0.92));
      border: 1px solid rgba(248,113,113,0.6);
      box-shadow: 0 8px 32px rgba(220,38,38,0.35);
      animation: predatorPulse 2s ease-in-out infinite;
    }
    @keyframes predatorPulse {
      0%, 100% { box-shadow: 0 0 12px rgba(248,113,113,0.45); }
      50% { box-shadow: 0 0 22px rgba(248,113,113,0.75); }
    }
    .predator-threat-line path {
      stroke: #dc2626 !important;
      stroke-width: 3 !important;
      stroke-dasharray: 8 10 !important;
      filter: drop-shadow(0 0 6px rgba(220,38,38,0.85));
    }
    @keyframes haloBreath {
      0%, 100% { opacity: 0.55; }
      50% { opacity: 0.9; }
    }
    .entity-marker-wrap { position: relative; width: 28px; height: 28px; }
    .entity-rich-tooltip {
      background: rgba(15, 23, 42, 0.96); border: 1px solid #38bdf8;
      color: #e2e8f0; padding: 8px 10px; border-radius: 8px;
      font-family: 'JetBrains Mono', monospace; font-size: 10px;
      line-height: 1.45; max-width: 260px; box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    }
    .entity-rich-tooltip .tt-title { color: #67e8f9; font-weight: 700; margin-bottom: 4px; }
    #entity-modal.open { display: flex; }
    #entity-modal { display: none; }
    .zone-label-icon {
      background: transparent; border: none;
      font-family: 'JetBrains Mono', monospace; font-size: 9px;
      color: #e2e8f0; text-shadow: 0 1px 3px #000;
      white-space: nowrap; text-align: center;
    }
  </style>
</head>
<body class="flex flex-col h-screen select-none">

  <!-- TACTICAL SIGN-IN OVERLAY -->
  <div id="login-modal" class="open fixed inset-0 z-[100] modal-overlay items-center justify-center p-4">
    <div class="w-full max-w-md admin-glass rounded-2xl shadow-2xl p-6 border border-cyan-800/50">
      <div class="text-center mb-6">
        <span class="mono font-bold text-xl tracking-wider text-cyan-400">[ RIAN ]</span>
        <p class="text-xs text-slate-400 mt-2 mono">TACTICAL OPERATOR AUTHENTICATION</p>
      </div>
      <form id="login-form" class="space-y-4" onsubmit="submitLogin(event)">
        <div>
          <label class="text-[10px] mono text-slate-400 uppercase">Email</label>
          <input id="login-email" type="email" required autocomplete="username"
                 class="mt-1 w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:border-cyan-500 outline-none mono" />
        </div>
        <div>
          <label class="text-[10px] mono text-slate-400 uppercase">Password</label>
          <input id="login-password" type="password" required autocomplete="current-password"
                 class="mt-1 w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:border-cyan-500 outline-none mono" />
        </div>
        <p id="login-error" class="hidden text-xs mono text-red-400"></p>
        <button type="submit" id="login-submit-btn"
                class="w-full py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-sm mono transition-colors">
          Authenticate
        </button>
      </form>
      <div class="flex flex-col gap-2 mt-4">
        <button type="button" onclick="quickFillLogin()"
                class="w-full py-1.5 text-[11px] mono text-cyan-400 border border-cyan-900/60 rounded-lg hover:bg-cyan-950/50">
          Quick-fill: admin@rian.internal
        </button>
        <button type="button" onclick="continueAsGuest()"
                class="w-full py-1.5 text-[11px] mono text-slate-400 hover:text-slate-200">
          Continue as Guest / Local Dev
        </button>
      </div>
    </div>
  </div>

  <!-- ADMIN & FLEET MODAL -->
  <div id="admin-modal" class="fixed inset-0 z-[90] modal-overlay items-center justify-center p-3 md:p-6">
    <div class="admin-glass rounded-2xl shadow-2xl w-full max-w-5xl max-h-[92vh] flex flex-col overflow-hidden">
      <div class="flex items-center justify-between px-4 py-3 border-b border-cyan-900/50">
        <span class="mono text-sm font-bold text-cyan-300 tracking-wider">🛡 ADMIN &amp; FLEET COMMAND</span>
        <button type="button" onclick="closeAdminModal()" class="text-slate-400 hover:text-white text-lg px-2">✕</button>
      </div>
      <div class="flex border-b border-slate-800 text-[11px] mono">
        <button type="button" data-admin-tab="cameras" onclick="switchAdminTab('cameras')"
                class="admin-tab-btn active flex-1 py-2.5 border-b-2 border-transparent text-slate-400 hover:text-cyan-200">Camera Provisioning</button>
        <button type="button" data-admin-tab="users" onclick="switchAdminTab('users')"
                class="admin-tab-btn flex-1 py-2.5 border-b-2 border-transparent text-slate-400 hover:text-cyan-200">User Accounts (RBAC)</button>
        <button type="button" data-admin-tab="sla" onclick="switchAdminTab('sla')"
                class="admin-tab-btn flex-1 py-2.5 border-b-2 border-transparent text-slate-400 hover:text-cyan-200">SLA &amp; Benchmarking</button>
      </div>
      <div class="flex-1 overflow-y-auto p-4">
        <!-- Tab: Cameras -->
        <div id="admin-tab-cameras" class="admin-tab-panel space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="mono text-xs font-bold text-slate-300">REGISTERED CAMERA FLEET</h3>
            <button type="button" onclick="loadAdminCameras()" class="text-[10px] mono px-2 py-1 rounded border border-cyan-800 text-cyan-400 hover:bg-cyan-950">↻ Refresh</button>
          </div>
          <div id="admin-cameras-error" class="hidden text-xs mono text-red-400"></div>
          <div class="overflow-x-auto rounded-lg border border-slate-800">
            <table class="w-full text-[11px] mono text-left">
              <thead class="bg-slate-950/80 text-slate-400">
                <tr>
                  <th class="px-2 py-2">Node</th>
                  <th class="px-2 py-2">Sender</th>
                  <th class="px-2 py-2">Az°</th>
                  <th class="px-2 py-2">Battery</th>
                  <th class="px-2 py-2">Status</th>
                  <th class="px-2 py-2">Actions</th>
                </tr>
              </thead>
              <tbody id="admin-cameras-tbody" class="divide-y divide-slate-800 text-slate-300">
                <tr><td colspan="6" class="px-2 py-4 text-center text-slate-500">Loading fleet...</td></tr>
              </tbody>
            </table>
          </div>
          <div class="rounded-xl border border-cyan-900/40 bg-slate-950/50 p-4">
            <h3 class="mono text-xs font-bold text-cyan-400 mb-3">ADD NEW CAMERA NODE</h3>
            <form id="camera-provision-form" class="grid grid-cols-1 md:grid-cols-2 gap-3" onsubmit="submitNewCamera(event)">
              <div>
                <label class="text-[10px] text-slate-400">Node Name</label>
                <input id="cam-node-name" required placeholder="cam-ridge-west"
                       class="mt-0.5 w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              </div>
              <div>
                <label class="text-[10px] text-slate-400">Sender Email</label>
                <input id="cam-sender-email" type="email" required placeholder="cam-ridge-west@ranch.domain"
                       class="mt-0.5 w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              </div>
              <div>
                <label class="text-[10px] text-slate-400">Longitude</label>
                <input id="cam-lon" type="number" step="any" required value="-119.82"
                       class="mt-0.5 w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              </div>
              <div>
                <label class="text-[10px] text-slate-400">Latitude</label>
                <input id="cam-lat" type="number" step="any" required value="36.745"
                       class="mt-0.5 w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              </div>
              <div class="md:col-span-2 flex flex-wrap items-center gap-3">
                <button type="button" id="btn-map-pick" onclick="toggleMapPickMode()"
                        class="px-3 py-1.5 text-[10px] mono font-bold rounded border border-amber-700/50 bg-amber-950/60 text-amber-300 hover:bg-amber-900/60">
                  Click Map to Pick Pin
                </button>
                <div class="flex items-center gap-2 flex-1 min-w-[200px]">
                  <span class="text-[10px] text-slate-400">Azimuth</span>
                  <input id="cam-azimuth" type="range" min="0" max="360" value="0" class="flex-1"
                         oninput="updateAzimuthUI(this.value)" />
                  <span id="cam-azimuth-deg" class="mono text-cyan-300 text-xs w-10">0°</span>
                  <span id="cam-compass" class="text-lg" title="Compass"><span class="compass-arrow" id="compass-arrow">↑</span></span>
                </div>
              </div>
              <p id="camera-form-error" class="hidden md:col-span-2 text-xs mono text-red-400"></p>
              <div class="md:col-span-2">
                <button type="submit" class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs mono font-bold rounded-lg">
                  Provision Camera Node
                </button>
              </div>
            </form>
          </div>
        </div>
        <!-- Tab: Users -->
        <div id="admin-tab-users" class="admin-tab-panel hidden space-y-4">
          <p id="admin-users-auth-hint" class="hidden text-xs mono text-amber-400">Admin role required — sign in with an ADMIN account.</p>
          <div id="admin-users-error" class="hidden text-xs mono text-red-400"></div>
          <div class="overflow-x-auto rounded-lg border border-slate-800">
            <table class="w-full text-[11px] mono text-left">
              <thead class="bg-slate-950/80 text-slate-400">
                <tr><th class="px-2 py-2">Email</th><th class="px-2 py-2">Name</th><th class="px-2 py-2">Role</th><th class="px-2 py-2">Active</th></tr>
              </thead>
              <tbody id="admin-users-tbody" class="divide-y divide-slate-800 text-slate-300">
                <tr><td colspan="4" class="px-2 py-4 text-center text-slate-500">—</td></tr>
              </tbody>
            </table>
          </div>
          <div class="rounded-xl border border-cyan-900/40 bg-slate-950/50 p-4">
            <h3 class="mono text-xs font-bold text-cyan-400 mb-3">CREATE USER</h3>
            <form id="user-create-form" class="grid grid-cols-1 md:grid-cols-2 gap-3" onsubmit="submitNewUser(event)">
              <input id="user-email" type="email" required placeholder="Email" class="bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              <input id="user-password" type="password" required placeholder="Password" class="bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              <input id="user-full-name" placeholder="Full Name" class="bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono" />
              <select id="user-role" class="bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs mono">
                <option value="ADMIN">ADMIN</option>
                <option value="OPERATOR">OPERATOR</option>
                <option value="RESPONDER">RESPONDER</option>
              </select>
              <p id="user-form-error" class="hidden md:col-span-2 text-xs mono text-red-400"></p>
              <div class="md:col-span-2">
                <button type="submit" class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs mono font-bold rounded-lg">Create User</button>
              </div>
            </form>
          </div>
        </div>
        <!-- Tab: SLA -->
        <div id="admin-tab-sla" class="admin-tab-panel hidden space-y-4">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div class="flex items-center gap-3 whitespace-nowrap">
              <h3 class="mono text-xs font-bold text-slate-300 whitespace-nowrap">REAL-TIME SLA DASHBOARD</h3>
              <span id="sla-timestamp" class="text-[10px] mono text-slate-500 whitespace-nowrap">—</span>
            </div>
            <div class="flex items-center gap-2 shrink-0 whitespace-nowrap">
              <span id="sla-overall-badge" class="mono text-[10px] px-2 py-0.5 rounded border whitespace-nowrap">—</span>
              <button type="button" id="btn-run-benchmark" onclick="runSlaBenchmark()"
                      class="px-3 py-1.5 text-[10px] mono font-bold rounded bg-violet-950 text-violet-300 border border-violet-700/50 hover:bg-violet-900/60 whitespace-nowrap">
                Run 25-Iteration Benchmark
              </button>
            </div>
          </div>
          <div id="sla-benchmark-status" class="hidden text-xs mono text-cyan-400 flex items-center gap-2">
            <span class="inline-block w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></span>
            Running benchmark iterations...
          </div>
          <div>
            <h4 class="mono text-[10px] text-cyan-500 mb-2 uppercase">Tier 1 — Hot-Path Latency</h4>
            <div id="sla-tier1-grid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2"></div>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
              <h4 class="mono text-[10px] text-emerald-500 mb-2 uppercase">Tier 2 — Sensor Fleet Availability</h4>
              <div id="sla-tier2" class="text-xs mono text-slate-300">—</div>
            </div>
            <div class="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
              <h4 class="mono text-[10px] text-amber-500 mb-2 uppercase">Tier 3 — Alert Delivery SLA</h4>
              <div id="sla-tier3" class="text-xs mono text-slate-300">—</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- DAILY HERD ROLL-CALL MODAL -->
  <div id="rollcall-modal" class="fixed inset-0 z-[92] modal-overlay items-center justify-center p-3 md:p-6">
    <div class="admin-glass rounded-2xl shadow-2xl w-full max-w-6xl max-h-[92vh] flex flex-col overflow-hidden">
      <div class="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-cyan-900/50">
        <div>
          <h2 class="mono text-sm font-bold text-cyan-300 tracking-wider">Daily Herd Roll-Call &amp; Pasture Census</h2>
          <p id="rollcall-generated-at" class="text-[10px] mono text-slate-500 mt-0.5">—</p>
        </div>
        <div class="flex items-center gap-2 shrink-0">
          <button type="button" onclick="exportRollcallCsv()"
                  class="px-3 py-1.5 text-[11px] mono font-bold rounded bg-emerald-950 text-emerald-300 border border-emerald-600/50 hover:bg-emerald-900/70 whitespace-nowrap">
            📥 Export CSV
          </button>
          <button type="button" onclick="closeRollcallModal()" class="text-slate-400 hover:text-white text-lg px-2">✕</button>
        </div>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-2 px-4 py-3 border-b border-slate-800/80 text-[11px] mono">
        <div class="rounded-lg bg-slate-950/70 border border-slate-800 px-3 py-2">
          <div class="text-slate-500 uppercase text-[9px]">Total Head</div>
          <div id="rollcall-metric-total" class="text-lg font-bold text-cyan-300">0</div>
        </div>
        <div class="rounded-lg bg-slate-950/70 border border-emerald-900/40 px-3 py-2">
          <div class="text-emerald-600 uppercase text-[9px]">Verified Active</div>
          <div id="rollcall-metric-verified" class="text-lg font-bold text-emerald-400">0</div>
        </div>
        <div class="rounded-lg bg-slate-950/70 border border-amber-900/40 px-3 py-2">
          <div class="text-amber-600 uppercase text-[9px]">Overdue Check</div>
          <div id="rollcall-metric-overdue" class="text-lg font-bold text-amber-400">0</div>
        </div>
        <div class="rounded-lg bg-slate-950/70 border border-red-900/40 px-3 py-2">
          <div class="text-red-500 uppercase text-[9px]">Missing / Critical</div>
          <div id="rollcall-metric-missing" class="text-lg font-bold text-red-400">0</div>
        </div>
      </div>
      <div id="rollcall-loading" class="hidden px-4 py-6 text-xs mono text-cyan-400 flex items-center gap-2 justify-center">
        <span class="inline-block w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></span>
        Compiling pasture census...
      </div>
      <div class="flex-1 overflow-auto px-4 py-3">
        <table class="w-full text-[11px] mono text-left min-w-[720px]">
          <thead class="bg-slate-950/90 text-slate-400 sticky top-0 z-10">
            <tr>
              <th class="px-2 py-2">Tag</th>
              <th class="px-2 py-2">Class</th>
              <th class="px-2 py-2">Pasture / Zone</th>
              <th class="px-2 py-2">Last Camera</th>
              <th class="px-2 py-2">Hours Ago</th>
              <th class="px-2 py-2">Status</th>
              <th class="px-2 py-2">VLM Observations</th>
            </tr>
          </thead>
          <tbody id="rollcall-tbody" class="divide-y divide-slate-800 text-slate-300">
            <tr><td colspan="7" class="px-2 py-6 text-center text-slate-500">Open roll-call to load inventory.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TOP HUD BAR -->
  <header class="h-14 bg-slate-950/95 border-b border-cyan-900/40 px-3 lg:px-4 flex items-center justify-between z-50 whitespace-nowrap overflow-x-auto overflow-y-hidden gap-2 lg:gap-3 flex-nowrap">
    <div class="flex items-center gap-2.5 shrink-0 whitespace-nowrap flex-nowrap">
      <div class="flex items-center gap-2 shrink-0 whitespace-nowrap flex-nowrap">
        <span class="mono font-bold text-base md:text-lg tracking-wider text-cyan-400 bg-cyan-950/60 border border-cyan-500/40 px-2.5 py-0.5 rounded shadow-sm whitespace-nowrap shrink-0">[ RIAN ]</span>
        <span id="header-app-title" class="text-xs font-semibold text-slate-300 tracking-wider whitespace-nowrap hidden sm:inline shrink-0">Rian Ranch Operations</span>
        <span class="text-[10px] mono text-cyan-500/70 border-l border-cyan-800/40 pl-2 whitespace-nowrap hidden 2xl:inline shrink-0">v1.0.0 // TACTICAL C2</span>
      </div>
      <div id="live-indicator" class="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 text-[11px] mono shrink-0 whitespace-nowrap">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
        <span id="conn-status" class="whitespace-nowrap">LIVE SSE STREAM</span>
      </div>
    </div>

    <!-- STATS COUNTERS, CONTROLS, CLOCK & ACTIONS (SINGLE LINE GUARANTEED) -->
    <div class="flex items-center gap-2 lg:gap-2.5 text-xs mono shrink-0 whitespace-nowrap flex-nowrap">
      <!-- SLEEK UNIFIED TELEMETRY CLUSTER -->
      <div class="hidden sm:flex items-center gap-2.5 bg-slate-900/80 border border-slate-800 px-2.5 py-1 rounded text-[11px] whitespace-nowrap shrink-0">
        <span title="Active Intruder Tracks" class="flex items-center gap-1"><span class="text-slate-400">TRK:</span><strong id="kpi-tracks" class="text-red-400 font-bold">0</strong></span>
        <span class="text-slate-700">·</span>
        <span title="Surveillance Sensors" class="flex items-center gap-1"><span class="text-slate-400">CAM:</span><strong id="kpi-sensors" class="text-emerald-400 font-bold">0</strong></span>
        <span class="text-slate-700 hidden md:inline">·</span>
        <span title="Strategic Chokepoints" class="hidden md:flex items-center gap-1"><span class="text-slate-400">CP:</span><strong id="kpi-chokepoints" class="text-sky-400 font-bold">0</strong></span>
        <span class="text-slate-700 hidden lg:inline">·</span>
        <span title="Tactical Directives" class="hidden lg:flex items-center gap-1"><span class="text-slate-400">DIR:</span><strong id="kpi-directives" class="text-amber-400 font-bold">0</strong></span>
      </div>

      <!-- BASEMAP SWITCHER -->
      <div class="flex items-center bg-slate-900 border border-slate-700 rounded p-0.5 text-[10px] shrink-0 whitespace-nowrap">
        <button id="btn-satellite" onclick="switchBasemap('satellite')" class="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-bold border border-cyan-700/50 whitespace-nowrap">🛰 Sat</button>
        <button id="btn-dark" onclick="switchBasemap('dark')" class="px-2 py-0.5 rounded text-slate-400 hover:text-slate-200 whitespace-nowrap">⬛ Dark</button>
      </div>

      <span id="user-badge" class="hidden px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-700 text-[10px] mono shrink-0 whitespace-nowrap max-w-[150px] truncate">[ GUEST ]</span>
      <button type="button" onclick="openRollcallModal()" class="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-[10px] mono text-emerald-300 border border-emerald-700/50 font-bold shrink-0 whitespace-nowrap">📋 Roll-Call</button>
      <button type="button" onclick="openAdminModal()" class="px-2 py-1 rounded bg-cyan-950 hover:bg-cyan-900 text-[10px] mono text-cyan-300 border border-cyan-700/50 font-bold shrink-0 whitespace-nowrap">🛡 Admin</button>
      <button type="button" id="btn-logout" onclick="logout()" class="hidden px-2 py-1 rounded bg-slate-900 hover:bg-red-950 text-[10px] mono text-slate-400 hover:text-red-300 border border-slate-700 shrink-0 whitespace-nowrap">Logout</button>

      <!-- CLOCK (STRICTLY ON ONE SINGLE LINE) -->
      <div class="flex items-center gap-2 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 shrink-0 whitespace-nowrap text-[11px]">
        <span id="clock-utc" class="font-bold text-cyan-300 whitespace-nowrap">00:00:00 UTC</span>
        <span class="text-slate-600 whitespace-nowrap">|</span>
        <span id="clock-local" class="text-slate-400 whitespace-nowrap">00:00:00 LOCAL</span>
      </div>

      <button onclick="promptApiUrl()" title="Configure Backend API URL" class="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-[10px] mono text-cyan-400 border border-cyan-800/50 shrink-0 whitespace-nowrap">⚙ API</button>
    </div>
  </header>

  <!-- MAIN VIEWPORT -->
  <div class="relative flex-1 flex overflow-hidden">

    <div id="predator-alert-banner" class="hidden absolute top-2 left-1/2 -translate-x-1/2 z-[55] max-w-[95vw] px-4 py-2 rounded-lg text-xs mono text-red-50 font-semibold flex flex-wrap items-center justify-center gap-2 text-center"></div>

    <!-- LEFT SIDEBAR: FLEET & PERIMETER -->
    <aside id="left-sidebar" class="absolute top-3 left-3 bottom-16 w-80 glass-panel rounded-xl shadow-2xl z-40 flex flex-col transition-all duration-300">
      <div class="p-3 border-b border-cyan-900/40 flex items-center justify-between">
        <span id="label-fleet-header" class="mono text-xs font-bold text-cyan-400 tracking-wider">ESTATE SURVEILLANCE FLEET</span>
        <span id="fleet-count" class="text-[10px] mono bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800/40">6 NODES</span>
      </div>
      <div id="sensors-list" class="flex-1 overflow-y-auto p-2.5 space-y-2 text-xs max-h-[38%]">
        <div class="text-slate-500 text-center py-6">Connecting to sensor fabric...</div>
      </div>

      <div class="p-3 border-t border-cyan-900/40 bg-slate-950/50">
        <div class="text-[11px] mono font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
          <span id="label-zones-header">Pastures &amp; Paddocks</span>
          <span class="text-[10px] text-emerald-400">LIVE HEADCOUNT</span>
        </div>
        <div id="zones-sidebar-list" class="space-y-1 text-[10px] mono text-slate-300 max-h-28 overflow-y-auto"></div>
      </div>

      <div class="p-3 border-t border-cyan-900/40 bg-slate-950/60 rounded-b-xl">
        <div class="text-[11px] mono font-semibold text-slate-300 mb-1 flex items-center justify-between">
          <span id="label-chokepoints-header">Gates &amp; Water Troughs</span>
          <span class="text-[10px] text-sky-400">PERIMETER</span>
        </div>
        <div id="chokepoints-mini" class="grid grid-cols-2 gap-1 text-[10px] mono text-slate-400"></div>
      </div>
    </aside>

    <!-- RIGHT SIDEBAR: AI COPILOT & THREAT DIRECTIVES -->
    <aside id="right-sidebar" class="absolute top-3 right-3 bottom-16 w-96 glass-panel rounded-xl shadow-2xl z-40 flex flex-col transition-all duration-300">

      <!-- ENTITY INSPECTOR (DeepSeek targeted diagnostic) -->
      <div id="entity-modal" class="absolute inset-0 z-[60] bg-slate-950/96 backdrop-blur-md rounded-xl border border-cyan-600/40 flex-col p-3 overflow-y-auto">
        <div class="flex items-center justify-between mb-2">
          <span class="mono text-xs font-bold text-cyan-300 tracking-wider">ENTITY INSPECTOR</span>
          <button type="button" onclick="closeEntityModal()" class="text-slate-400 hover:text-white text-lg px-1">✕</button>
        </div>
        <div id="entity-modal-loading" class="hidden flex items-center gap-2 text-xs mono text-cyan-400 py-6 justify-center">
          <span class="inline-block w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></span>
          Synthesizing targeted diagnostic...
        </div>
        <div id="entity-modal-body" class="space-y-2 text-xs">
          <div id="entity-risk-badge" class="mono text-[10px] font-bold px-2 py-0.5 rounded border inline-block">—</div>
          <div id="entity-headline" class="mono font-bold text-slate-100 text-sm leading-snug">Select an entity on the map</div>
          <p id="entity-insight" class="text-slate-300 text-[11px] leading-relaxed"></p>
          <div id="entity-actions" class="space-y-1 text-[10px] mono text-cyan-300"></div>
          <div id="entity-telemetry" class="grid grid-cols-2 gap-1 text-[10px] mono text-slate-400 border-t border-slate-800 pt-2"></div>
          <div id="entity-provider" class="text-[9px] mono text-slate-500 text-right pt-1"></div>
          <div id="entity-snapshot-section" class="border-t border-slate-800 pt-2 mt-2 space-y-2">
            <div class="flex items-center justify-between gap-2">
              <span class="mono text-[10px] font-bold text-cyan-300">SNAPSHOT VISUAL INSPECTION</span>
              <span id="entity-snapshot-meta" class="text-[9px] mono text-slate-500 truncate"></span>
            </div>
            <div id="entity-snapshot-placeholder" class="hidden text-center py-3 text-slate-500 text-[10px]">
              No forensic snapshot linked to this track yet.
              <button type="button" onclick="generateTestSnapshot()" class="mt-2 block mx-auto px-2 py-1 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 rounded border border-cyan-800 text-[10px] mono">Generate Test Snapshot</button>
            </div>
            <div id="entity-snapshot-viewer" class="hidden relative w-full rounded border border-slate-700 overflow-hidden bg-slate-900">
              <img id="entity-snapshot-img" alt="Forensic snapshot" class="w-full block" />
              <canvas id="entity-snapshot-canvas" class="absolute top-0 left-0 w-full h-full pointer-events-none"></canvas>
            </div>
            <button type="button" id="entity-vlm-btn" onclick="runVlmAnalysis()" class="hidden w-full py-1.5 mono text-[10px] font-bold bg-indigo-950 hover:bg-indigo-900 text-indigo-200 border border-indigo-700/50 rounded">🔍 Deep AI Vision Analysis</button>
            <div id="entity-vlm-loading" class="hidden text-[10px] mono text-cyan-400 flex items-center gap-2 justify-center py-2">
              <span class="inline-block w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></span>
              Running multimodal forensic analysis...
            </div>
            <div id="entity-vlm-report" class="hidden space-y-1 text-[10px] text-slate-300 border border-slate-800 rounded p-2 bg-slate-900/80">
              <div id="entity-vlm-risk" class="mono font-bold text-amber-300 text-[10px]"></div>
              <p id="entity-vlm-visual" class="leading-relaxed text-[10px]"></p>
              <p id="entity-vlm-env" class="text-slate-400 leading-relaxed text-[10px]"></p>
              <ul id="entity-vlm-findings" class="list-disc pl-4 space-y-0.5 text-cyan-200/90 text-[10px]"></ul>
              <p id="entity-vlm-action" class="mono text-emerald-300 text-[10px]"></p>
              <div id="entity-vlm-provider" class="text-[9px] text-slate-500 text-right"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- AI SITREP CARD -->
      <div class="p-3 border-b border-cyan-900/40 bg-slate-950/40">
        <div class="flex items-center justify-between mb-1.5">
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            <span class="mono text-xs font-bold text-cyan-300 tracking-wider">AI TACTICAL COPILOT</span>
          </div>
          <button onclick="fetchAiSitrep()" title="Regenerate SITREP" class="text-[10px] mono bg-cyan-950 hover:bg-cyan-900 text-cyan-300 px-2 py-0.5 rounded border border-cyan-700/50">⚡ SITREP</button>
        </div>

        <div id="ai-sitrep-card" class="bg-slate-900/90 rounded-lg p-2.5 border border-cyan-900/40 text-xs">
          <div class="flex items-center justify-between gap-2 mb-1.5 whitespace-nowrap">
            <span id="sitrep-badge" class="mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-700/50 truncate shrink-0 max-w-[210px] whitespace-nowrap">DEFCON 5 // SECURE</span>
            <span id="sitrep-time" class="mono text-[10px] text-slate-400 shrink-0 whitespace-nowrap">--:--:-- UTC</span>
          </div>
          <div id="sitrep-headline" class="mono font-bold text-slate-200 mt-1.5 text-[11px] leading-snug break-words">Perimeter Secure</div>
          <p id="sitrep-summary" class="text-slate-300 text-[11px] leading-relaxed mt-1">
            Analyzing spatial telemetry and evaluating kinematic gating envelopes across acreage...
          </p>
          <div id="sitrep-actions" class="mt-2 space-y-1 text-[10px] mono text-cyan-300"></div>
          <div id="sitrep-provider" class="text-[9px] mono text-slate-500 mt-2 border-t border-slate-800 pt-1 text-right">Engine: Local Tactical Reasoner</div>
        </div>

        <!-- NATURAL LANGUAGE OPERATOR COMMS -->
        <div class="mt-2 flex gap-1">
          <input id="ai-query-input" type="text" placeholder="Ask Tactical Copilot (e.g. Risk at North Gate?)..." 
                 onkeydown="if(event.key==='Enter') sendAiQuery()"
                 class="flex-1 bg-slate-900 text-slate-200 text-xs px-2.5 py-1 rounded border border-slate-700 outline-none focus:border-cyan-500 mono" />
          <button onclick="sendAiQuery()" class="px-2 py-1 bg-cyan-700 hover:bg-cyan-600 text-white mono text-xs font-bold rounded">Ask</button>
        </div>
        <div id="ai-query-output" class="hidden mt-1.5 p-2 bg-slate-900/95 border border-cyan-600/50 rounded text-[11px] mono text-cyan-200"></div>
      </div>

      <!-- DIRECTIVES FEED -->
      <div class="p-2 border-b border-cyan-900/30">
        <div class="flex items-center justify-between px-1 mb-1">
          <span id="label-directives-header" class="mono text-[11px] font-bold text-amber-400">Herd Roundups &amp; Checks</span>
          <span id="threat-indicator" class="text-[9px] mono bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700">STANDBY</span>
        </div>
        <div id="directives-container" class="space-y-1.5 max-h-40 overflow-y-auto">
          <div class="text-slate-500 text-xs text-center py-2">No active intercept directives</div>
        </div>
      </div>

      <!-- ACTIVE TRACKS -->
      <div class="flex-1 overflow-y-auto p-2.5 flex flex-col">
        <div class="px-1 mb-1 text-[11px] mono text-slate-400 font-semibold flex items-center justify-between">
          <span id="label-tracks-header">Livestock / Wildlife Tracks</span>
          <span id="active-tracks-count" class="text-red-400">0 ACTIVE</span>
        </div>
        <div id="tracks-list" class="flex-1 overflow-y-auto space-y-1.5 text-xs">
          <div class="text-slate-500 text-center py-4">Perimeter secure. Zero unassociated radar gates.</div>
        </div>
      </div>
    </aside>

    <!-- MAP CONTAINER -->
    <div id="map"></div>

    <div id="map-control-bar" class="absolute top-[4.25rem] left-1/2 -translate-x-1/2 z-[45] flex items-center gap-2 glass-panel rounded-lg px-2 py-1 shadow-lg">
      <button type="button" id="btn-draw-zone" onclick="toggleZoneDrawMode()"
        class="px-2.5 py-1 text-[11px] mono font-semibold bg-emerald-950/90 hover:bg-emerald-900 text-emerald-200 border border-emerald-600/50 rounded whitespace-nowrap">
        [ ✏️ Draw Zone ]
      </button>
    </div>

    <div id="draw-zone-card" class="absolute top-28 right-[25rem] z-[50] w-72 glass-panel rounded-xl shadow-2xl p-3 space-y-2">
      <div class="mono text-xs font-bold text-emerald-300 tracking-wider">Save Pasture / Zone</div>
      <p class="text-[10px] text-slate-400">Click the map to place vertices. Need at least 3 points.</p>
      <label class="block text-[10px] mono text-slate-400">Zone Name</label>
      <input id="draw-zone-name" type="text" placeholder="Alfalfa South 40"
        class="w-full bg-slate-900 text-slate-200 text-xs px-2 py-1 rounded border border-slate-700 mono" />
      <label class="block text-[10px] mono text-slate-400">Code</label>
      <input id="draw-zone-code" type="text" placeholder="PASTURE-ALFALFA-S"
        class="w-full bg-slate-900 text-slate-200 text-xs px-2 py-1 rounded border border-slate-700 mono" />
      <label class="block text-[10px] mono text-slate-400">Type</label>
      <select id="draw-zone-type" class="w-full bg-slate-900 text-slate-200 text-xs px-2 py-1 rounded border border-slate-700 mono">
        <option value="pasture">Pasture</option>
        <option value="paddock">Paddock</option>
        <option value="wash">Wash</option>
        <option value="restricted_zone">Restricted</option>
        <option value="container_bay">Container Bay</option>
      </select>
      <label class="block text-[10px] mono text-slate-400">Color</label>
      <input id="draw-zone-color" type="color" value="#22c55e" class="w-full h-8 rounded border border-slate-700 bg-slate-900" />
      <div class="flex gap-2 pt-1">
        <button type="button" onclick="completeZoneDraw()" class="flex-1 py-1.5 mono text-[10px] font-bold bg-emerald-700 hover:bg-emerald-600 text-white rounded">Complete &amp; Save</button>
        <button type="button" onclick="cancelZoneDraw()" class="flex-1 py-1.5 mono text-[10px] font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-600">Cancel</button>
      </div>
    </div>

    <!-- BOTTOM FLOATING SIMULATION & CONTROL BAR -->
    <div class="absolute bottom-3 left-3 right-3 h-12 glass-panel rounded-xl shadow-2xl z-40 flex items-center justify-between px-3 md:px-4 gap-2 whitespace-nowrap overflow-x-auto flex-nowrap">
      <div class="flex items-center gap-2 shrink-0 whitespace-nowrap flex-nowrap">
        <span class="text-[11px] mono text-slate-400 font-bold mr-1 shrink-0 whitespace-nowrap">SIMULATION:</span>
        <button onclick="runScenario('stray_cow')" class="px-2.5 py-1 text-xs mono font-semibold bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-700/50 rounded transition-colors whitespace-nowrap shrink-0">
          ► Stray Cow Drift
        </button>
        <button onclick="runScenario('vehicle_intrusion')" class="px-2.5 py-1 text-xs mono font-semibold bg-red-950/80 hover:bg-red-900 text-red-300 border border-red-700/50 rounded transition-colors whitespace-nowrap shrink-0">
          ► Vehicle Intrusion
        </button>
        <button onclick="runScenario('pedestrian_hike')" class="px-2.5 py-1 text-xs mono font-semibold bg-amber-950/80 hover:bg-amber-900 text-amber-300 border border-amber-700/50 rounded transition-colors whitespace-nowrap shrink-0">
          ► Pedestrian Hike
        </button>
        <button onclick="simulatePredatorIncursion()" class="px-2.5 py-1 text-xs mono font-semibold bg-rose-950/80 hover:bg-rose-900 text-rose-200 border border-rose-600/50 rounded transition-colors whitespace-nowrap shrink-0">
          ► Predator Incursion
        </button>
        <button onclick="resetSimulation()" class="px-2.5 py-1 text-xs mono font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 rounded transition-colors whitespace-nowrap shrink-0">
          ↺ Reset All
        </button>
      </div>

      <div class="flex items-center gap-2.5 text-xs mono shrink-0 whitespace-nowrap flex-nowrap">
        <!-- Manual Camera Trigger Form -->
        <div class="flex items-center gap-1.5 bg-slate-900/80 px-2 py-1 rounded border border-slate-700 shrink-0 whitespace-nowrap">
          <label class="text-[10px] text-slate-400 whitespace-nowrap">Inject:</label>
          <select id="sel-camera" class="bg-slate-950 text-cyan-300 text-xs rounded border border-slate-700 px-1 py-0.5 outline-none shrink-0">
            <option value="cam-gate-north">cam-gate-north</option>
            <option value="cam-gate-south">cam-gate-south</option>
            <option value="cam-gate-east">cam-gate-east</option>
            <option value="cam-gate-west">cam-gate-west</option>
            <option value="cam-intersection">cam-intersection</option>
          </select>
          <select id="sel-class" class="bg-slate-950 text-amber-300 text-xs rounded border border-slate-700 px-1 py-0.5 outline-none shrink-0">
            <option value="cow">Cow</option>
            <option value="truck">Truck</option>
            <option value="person">Person</option>
            <option value="car">Car</option>
          </select>
          <button onclick="injectDetection()" class="px-2 py-0.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded text-[11px] transition-colors whitespace-nowrap shrink-0">
            Emit
          </button>
        </div>

        <span class="text-[10px] text-slate-400 hidden xl:inline shrink-0 whitespace-nowrap">Click map to relocate Response Unit</span>
      </div>
    </div>

  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const API_BASE = (window.location.origin.includes('vercel.app') || window.location.protocol === 'file:')
      ? (localStorage.getItem('RIAN_BACKEND_URL') || 'http://localhost:8000')
      : '';

    const AUTH_TOKEN_KEY = 'rian_access_token';
    const AUTH_USER_KEY = 'rian_user';
    const AUTH_GUEST_KEY = 'rian_guest_mode';
    let mapPickMode = false;
    let zoneDrawMode = false;
    let zoneDrawVertices = [];
    let zoneDrawLayer = null;
    let zoneDrawGuides = null;
    let authSession = { authenticated: false, guest: false, user: null };

    const SLA_COMPONENT_LABELS = {
      ingestion_decode: 'Ingestion Decode',
      yolo_reid_inference: 'YOLO Re-ID Inference',
      isochrone_wavefront: 'Isochrone Wavefront',
      intercept_routing: 'Intercept Routing',
      hotpath_end_to_end: 'Hot-Path E2E'
    };

    function getAccessToken() {
      return localStorage.getItem(AUTH_TOKEN_KEY);
    }

    function getStoredUser() {
      try {
        const raw = localStorage.getItem(AUTH_USER_KEY);
        return raw ? JSON.parse(raw) : null;
      } catch (_) {
        return null;
      }
    }

    function authHeaders(json = true) {
      const headers = {};
      if (json) headers['Content-Type'] = 'application/json';
      const token = getAccessToken();
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return headers;
    }

    function slaStatusClass(status) {
      if (status === 'HEALTHY') return 'bg-emerald-950 text-emerald-300 border-emerald-700/50';
      if (status === 'DEGRADED') return 'bg-amber-950 text-amber-300 border-amber-700/50';
      return 'bg-red-950 text-red-300 border-red-700/50';
    }

    function refreshAuthUI() {
      const badge = document.getElementById('user-badge');
      const logoutBtn = document.getElementById('btn-logout');
      const user = getStoredUser();
      const token = getAccessToken();
      authSession.authenticated = !!(token && user);
      authSession.user = user;
      authSession.guest = localStorage.getItem(AUTH_GUEST_KEY) === '1' && !authSession.authenticated;

      if (authSession.authenticated && user) {
        const role = (user.role || 'OPERATOR').toUpperCase();
        badge.textContent = '[ ' + role + ' ] ' + user.email;
        badge.className = 'px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 text-[10px] mono';
        badge.classList.remove('hidden');
        logoutBtn.classList.remove('hidden');
      } else if (authSession.guest) {
        badge.textContent = '[ GUEST ]';
        badge.className = 'px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-700 text-[10px] mono';
        badge.classList.remove('hidden');
        logoutBtn.classList.add('hidden');
      } else {
        badge.classList.add('hidden');
        logoutBtn.classList.add('hidden');
      }
    }

    function showLoginModal() {
      const el = document.getElementById('login-modal');
      el.classList.add('open');
    }

    function hideLoginModal() {
      const el = document.getElementById('login-modal');
      el.classList.remove('open');
    }

    function quickFillLogin() {
      document.getElementById('login-email').value = 'admin@rian.internal';
      document.getElementById('login-password').value = 'rian_admin_secret';
    }

    function continueAsGuest() {
      localStorage.setItem(AUTH_GUEST_KEY, '1');
      authSession.guest = true;
      refreshAuthUI();
      hideLoginModal();
    }

    async function submitLogin(ev) {
      ev.preventDefault();
      const errEl = document.getElementById('login-error');
      errEl.classList.add('hidden');
      const btn = document.getElementById('login-submit-btn');
      btn.disabled = true;
      btn.textContent = 'Authenticating...';
      try {
        const res = await fetch(API_BASE + '/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: document.getElementById('login-email').value.trim(),
            password: document.getElementById('login-password').value
          })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          errEl.textContent = data.detail || 'Authentication failed';
          errEl.classList.remove('hidden');
          return;
        }
        localStorage.setItem(AUTH_TOKEN_KEY, data.access_token);
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
        localStorage.removeItem(AUTH_GUEST_KEY);
        refreshAuthUI();
        hideLoginModal();
      } catch (e) {
        errEl.textContent = 'Unable to reach auth service.';
        errEl.classList.remove('hidden');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Authenticate';
      }
    }

    function logout() {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
      localStorage.removeItem(AUTH_GUEST_KEY);
      refreshAuthUI();
      showLoginModal();
    }

    function openAdminModal() {
      document.getElementById('admin-modal').classList.add('open');
      switchAdminTab('cameras');
    }

    function closeAdminModal() {
      document.getElementById('admin-modal').classList.remove('open');
      if (mapPickMode) toggleMapPickMode(false);
    }

    function switchAdminTab(tab) {
      document.querySelectorAll('.admin-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-admin-tab') === tab);
      });
      document.querySelectorAll('.admin-tab-panel').forEach(panel => panel.classList.add('hidden'));
      const panel = document.getElementById('admin-tab-' + tab);
      if (panel) panel.classList.remove('hidden');
      if (tab === 'cameras') loadAdminCameras();
      if (tab === 'users') loadAdminUsers();
      if (tab === 'sla') loadSlaSummary();
    }

    function updateAzimuthUI(deg) {
      const d = Number(deg);
      document.getElementById('cam-azimuth-deg').textContent = Math.round(d) + '°';
      const arrow = document.getElementById('compass-arrow');
      if (arrow) arrow.style.transform = 'rotate(' + d + 'deg)';
    }

    function toggleMapPickMode(forceOff) {
      const btn = document.getElementById('btn-map-pick');
      if (forceOff === false || (mapPickMode && forceOff !== true)) {
        mapPickMode = false;
        map.getContainer().classList.remove('map-pick-cursor');
        if (btn) {
          btn.textContent = 'Click Map to Pick Pin';
          btn.classList.remove('ring-2', 'ring-amber-400');
        }
        return;
      }
      mapPickMode = !mapPickMode;
      if (mapPickMode) {
        if (zoneDrawMode) toggleZoneDrawMode(false);
        map.getContainer().classList.add('map-pick-cursor');
        btn.textContent = 'Picking on map… (click to place)';
        btn.classList.add('ring-2', 'ring-amber-400');
        closeAdminModal();
      } else {
        toggleMapPickMode(false);
      }
    }

    function clearZoneDrawGraphics() {
      zoneDrawVertices = [];
      if (zoneDrawLayer) {
        map.removeLayer(zoneDrawLayer);
        zoneDrawLayer = null;
      }
      if (zoneDrawGuides) {
        map.removeLayer(zoneDrawGuides);
        zoneDrawGuides = null;
      }
    }

    function refreshZoneDrawGraphics() {
      if (zoneDrawGuides) {
        map.removeLayer(zoneDrawGuides);
        zoneDrawGuides = null;
      }
      if (!zoneDrawVertices.length) return;
      const latlngs = zoneDrawVertices.map(v => [v[1], v[0]]);
      zoneDrawGuides = L.polyline(latlngs, {
        color: '#34d399',
        weight: 2,
        dashArray: '6, 8',
        opacity: 0.9
      }).addTo(map);
      if (zoneDrawVertices.length >= 3) {
        if (zoneDrawLayer) map.removeLayer(zoneDrawLayer);
        zoneDrawLayer = L.polygon(latlngs, {
          color: '#34d399',
          fillColor: '#22c55e',
          fillOpacity: 0.15,
          weight: 1.5,
          dashArray: '4, 6'
        }).addTo(map);
      }
    }

    function toggleZoneDrawMode(forceOff) {
      const btn = document.getElementById('btn-draw-zone');
      const card = document.getElementById('draw-zone-card');
      if (forceOff === false || (zoneDrawMode && forceOff !== true)) {
        zoneDrawMode = false;
        map.getContainer().classList.remove('map-draw-zone-cursor');
        if (btn) btn.classList.remove('ring-2', 'ring-emerald-400');
        if (card) card.classList.remove('open');
        clearZoneDrawGraphics();
        return;
      }
      zoneDrawMode = !zoneDrawMode;
      if (zoneDrawMode) {
        if (mapPickMode) toggleMapPickMode(false);
        clearZoneDrawGraphics();
        map.getContainer().classList.add('map-draw-zone-cursor');
        if (btn) btn.classList.add('ring-2', 'ring-emerald-400');
        if (card) card.classList.add('open');
      } else {
        toggleZoneDrawMode(false);
      }
    }

    function cancelZoneDraw() {
      toggleZoneDrawMode(false);
    }

    async function completeZoneDraw() {
      if (zoneDrawVertices.length < 3) {
        alert('Place at least 3 vertices on the map.');
        return;
      }
      const name = (document.getElementById('draw-zone-name').value || '').trim();
      const code = (document.getElementById('draw-zone-code').value || '').trim();
      const zoneType = document.getElementById('draw-zone-type').value;
      const color = document.getElementById('draw-zone-color').value || '#22c55e';
      if (!name || name.length < 2) {
        alert('Enter a zone name (min 2 characters).');
        return;
      }
      if (!code || code.length < 2) {
        alert('Enter a zone code (min 2 characters).');
        return;
      }
      const ring = zoneDrawVertices.slice();
      const first = ring[0];
      const last = ring[ring.length - 1];
      if (first[0] !== last[0] || first[1] !== last[1]) {
        ring.push([first[0], first[1]]);
      }
      try {
        const res = await fetch(`${API_BASE}/api/v1/zones`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            zone_code: code,
            zone_type: zoneType,
            color,
            coordinates: ring
          })
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Failed to save zone');
        }
        toggleZoneDrawMode(false);
        await loadZones();
      } catch (e) {
        alert('Zone save failed: ' + (e.message || e));
      }
    }

    async function loadAdminCameras() {
      const tbody = document.getElementById('admin-cameras-tbody');
      const errEl = document.getElementById('admin-cameras-error');
      errEl.classList.add('hidden');
      try {
        const res = await fetch(API_BASE + '/api/v1/cameras');
        const cameras = await res.json();
        if (!res.ok) throw new Error(cameras.detail || 'Failed to load cameras');
        if (!Array.isArray(cameras) || !cameras.length) {
          tbody.innerHTML = '<tr><td colspan="6" class="px-2 py-4 text-center text-slate-500">No cameras registered</td></tr>';
          return;
        }
        tbody.innerHTML = cameras.map(c => {
          const batt = c.battery_level != null ? c.battery_level + '%' : '—';
          const st = (c.status || 'unknown').toUpperCase();
          const stCls = c.status === 'active' ? 'text-emerald-400' : 'text-red-400';
          return `<tr>
            <td class="px-2 py-2 font-bold">${escapeHtml(c.node_name)}</td>
            <td class="px-2 py-2 text-slate-400">${escapeHtml(c.sender_email)}</td>
            <td class="px-2 py-2">${c.azimuth_deg}°</td>
            <td class="px-2 py-2">${batt}</td>
            <td class="px-2 py-2 ${stCls}">${st}</td>
            <td class="px-2 py-2 space-x-1">
              <button type="button" onclick="cameraTestProbe('${c.id}')" class="px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800 text-[10px]">Test Probe</button>
              <button type="button" onclick="cameraDecommission('${c.id}')" class="px-1.5 py-0.5 rounded bg-red-950 text-red-300 border border-red-800 text-[10px]">Decommission</button>
            </td>
          </tr>`;
        }).join('');
      } catch (e) {
        errEl.textContent = String(e.message || e);
        errEl.classList.remove('hidden');
        tbody.innerHTML = '<tr><td colspan="6" class="px-2 py-4 text-center text-slate-500">Error loading fleet</td></tr>';
      }
    }

    async function submitNewCamera(ev) {
      ev.preventDefault();
      const errEl = document.getElementById('camera-form-error');
      errEl.classList.add('hidden');
      const body = {
        node_name: document.getElementById('cam-node-name').value.trim(),
        sender_email: document.getElementById('cam-sender-email').value.trim(),
        lon: parseFloat(document.getElementById('cam-lon').value),
        lat: parseFloat(document.getElementById('cam-lat').value),
        azimuth_deg: parseFloat(document.getElementById('cam-azimuth').value)
      };
      try {
        const res = await fetch(API_BASE + '/api/v1/cameras', {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify(body)
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          errEl.textContent = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || 'Provision failed');
          errEl.classList.remove('hidden');
          return;
        }
        document.getElementById('camera-provision-form').reset();
        document.getElementById('cam-lon').value = '-119.82';
        document.getElementById('cam-lat').value = '36.745';
        updateAzimuthUI(0);
        await loadAdminCameras();
        await loadState();
      } catch (e) {
        errEl.textContent = 'Network error provisioning camera.';
        errEl.classList.remove('hidden');
      }
    }

    async function cameraTestProbe(id) {
      try {
        await fetch(API_BASE + '/api/v1/cameras/' + id + '/test-probe', { method: 'POST', headers: authHeaders(false) });
        await loadState();
      } catch (e) {
        console.error(e);
      }
    }

    async function cameraDecommission(id) {
      if (!confirm('Decommission this camera node?')) return;
      try {
        await fetch(API_BASE + '/api/v1/cameras/' + id, { method: 'DELETE', headers: authHeaders(false) });
        await loadAdminCameras();
        await loadState();
      } catch (e) {
        console.error(e);
      }
    }

    async function loadAdminUsers() {
      const tbody = document.getElementById('admin-users-tbody');
      const errEl = document.getElementById('admin-users-error');
      const hint = document.getElementById('admin-users-auth-hint');
      errEl.classList.add('hidden');
      hint.classList.add('hidden');
      if (!getAccessToken()) {
        hint.classList.remove('hidden');
        tbody.innerHTML = '<tr><td colspan="4" class="px-2 py-4 text-center text-slate-500">Sign in required</td></tr>';
        return;
      }
      try {
        const res = await fetch(API_BASE + '/api/v1/admin/users', { headers: authHeaders(false) });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          hint.classList.remove('hidden');
          throw new Error(data.detail || 'Failed to load users');
        }
        if (!data.length) {
          tbody.innerHTML = '<tr><td colspan="4" class="px-2 py-4 text-center text-slate-500">No users</td></tr>';
          return;
        }
        tbody.innerHTML = data.map(u => `<tr>
          <td class="px-2 py-2">${escapeHtml(u.email)}</td>
          <td class="px-2 py-2">${escapeHtml(u.full_name || '—')}</td>
          <td class="px-2 py-2 text-cyan-300">${escapeHtml(u.role)}</td>
          <td class="px-2 py-2">${u.is_active ? 'YES' : 'NO'}</td>
        </tr>`).join('');
      } catch (e) {
        errEl.textContent = String(e.message || e);
        errEl.classList.remove('hidden');
      }
    }

    async function submitNewUser(ev) {
      ev.preventDefault();
      const errEl = document.getElementById('user-form-error');
      errEl.classList.add('hidden');
      if (!getAccessToken()) {
        errEl.textContent = 'Admin sign-in required.';
        errEl.classList.remove('hidden');
        return;
      }
      const body = {
        email: document.getElementById('user-email').value.trim(),
        password: document.getElementById('user-password').value,
        full_name: document.getElementById('user-full-name').value.trim() || null,
        role: document.getElementById('user-role').value
      };
      try {
        const res = await fetch(API_BASE + '/api/v1/admin/users', {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify(body)
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          errEl.textContent = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || 'Create failed');
          errEl.classList.remove('hidden');
          return;
        }
        document.getElementById('user-create-form').reset();
        await loadAdminUsers();
      } catch (e) {
        errEl.textContent = 'Network error creating user.';
        errEl.classList.remove('hidden');
      }
    }

    function renderSlaSummary(summary) {
      if (!summary) return;
      document.getElementById('sla-timestamp').textContent = 'Updated: ' + (summary.timestamp_utc || '—');
      const overall = document.getElementById('sla-overall-badge');
      const oh = summary.overall_health || '—';
      overall.textContent = oh;
      const ohCls = oh === 'OPTIMAL' ? 'bg-emerald-950 text-emerald-300 border-emerald-700/50'
        : oh === 'WARNING' ? 'bg-amber-950 text-amber-300 border-amber-700/50'
        : 'bg-red-950 text-red-300 border-red-700/50';
      overall.className = 'mono text-[10px] px-2 py-0.5 rounded border ' + ohCls;

      const tier1 = summary.tier1_hotpath_sla || {};
      const grid = document.getElementById('sla-tier1-grid');
      grid.innerHTML = Object.keys(tier1).map(key => {
        const m = tier1[key];
        const label = SLA_COMPONENT_LABELS[key] || key;
        const badge = slaStatusClass(m.status);
        return `<div class="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
          <div class="flex items-center justify-between gap-2 mb-1">
            <span class="mono text-[10px] text-slate-200 font-bold">${escapeHtml(label)}</span>
            <span class="mono text-[9px] px-1.5 py-0.5 rounded border ${badge}">${m.status}</span>
          </div>
          <div class="text-[10px] mono text-slate-400">Budget &lt; ${m.target_budget_ms}ms · n=${m.samples_count}</div>
          <div class="grid grid-cols-3 gap-1 mt-1.5 text-[10px] mono">
            <div><span class="text-slate-500">p50</span><br><span class="text-cyan-300">${m.p50_ms.toFixed(2)}</span></div>
            <div><span class="text-slate-500">p95</span><br><span class="text-cyan-300">${m.p95_ms.toFixed(2)}</span></div>
            <div><span class="text-slate-500">p99</span><br><span class="text-cyan-300">${m.p99_ms.toFixed(2)}</span></div>
          </div>
        </div>`;
      }).join('');

      const uptime = summary.tier2_fleet_uptime_pct;
      const lowBat = summary.tier2_low_battery_nodes || [];
      const t2Ok = uptime >= 99;
      document.getElementById('sla-tier2').innerHTML = `
        <div>Uptime: <span class="${t2Ok ? 'text-emerald-400' : 'text-red-400'} font-bold">${uptime.toFixed(2)}%</span> <span class="text-slate-500">(target ≥99%)</span></div>
        <div class="mt-2 text-slate-400">Low battery warnings: ${lowBat.length ? lowBat.map(n => '<span class="text-amber-400">' + escapeHtml(n) + '</span>').join(', ') : '<span class="text-emerald-400">None</span>'}</div>`;

      const alertP95 = summary.tier3_alert_delivery_p95_s;
      const t3Ok = alertP95 < 5.0;
      document.getElementById('sla-tier3').innerHTML = `
        <div>Trigger → Alert p95: <span class="${t3Ok ? 'text-emerald-400' : 'text-red-400'} font-bold">${alertP95.toFixed(3)}s</span></div>
        <div class="text-slate-500 mt-1">Target &lt; 5.0s</div>`;
    }

    async function loadSlaSummary() {
      try {
        const res = await fetch(API_BASE + '/api/v1/sla/summary');
        const data = await res.json();
        if (res.ok) renderSlaSummary(data);
      } catch (e) {
        console.error('SLA load failed', e);
      }
    }

    async function runSlaBenchmark() {
      const statusEl = document.getElementById('sla-benchmark-status');
      const btn = document.getElementById('btn-run-benchmark');
      statusEl.classList.remove('hidden');
      btn.disabled = true;
      try {
        const res = await fetch(API_BASE + '/api/v1/sla/run-benchmark?runs=25', { method: 'POST' });
        const data = await res.json();
        if (res.ok) renderSlaSummary(data);
      } catch (e) {
        console.error(e);
      } finally {
        statusEl.classList.add('hidden');
        btn.disabled = false;
      }
    }

    function initAuthGate() {
      refreshAuthUI();
      const token = getAccessToken();
      const guest = localStorage.getItem(AUTH_GUEST_KEY) === '1';
      if (token && getStoredUser()) {
        hideLoginModal();
      } else if (guest) {
        hideLoginModal();
      } else {
        showLoginModal();
      }
      updateAzimuthUI(document.getElementById('cam-azimuth').value);
    }

    function promptApiUrl() {
      const current = localStorage.getItem('RIAN_BACKEND_URL') || (window.location.origin.includes('vercel.app') ? 'http://localhost:8000' : window.location.origin);
      const next = prompt('Enter Rian Engine Backend URL (e.g. http://localhost:8000 or https://your-domain.com):', current);
      if (next !== null) {
        localStorage.setItem('RIAN_BACKEND_URL', next.replace(/\/$/, ''));
        window.location.reload();
      }
    }

    function escapeHtml(text) {
      return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function formatCopilotText(text) {
      const safe = escapeHtml(text);
      return safe
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\n/g, '<br>');
    }

    let domainProfile = null;
    let spatialZones = [];
    const trackKinematics = new Map();
    const entityRenderState = new Map();
    let kinematicAnimHandle = null;
    let selectedTrackId = null;

    const ZONE_STYLE = {
      'PASTURE-NORTH': { fill: '#10b981', stroke: '#34d399' },
      'ZONE-CREEK': { fill: '#06b6d4', stroke: '#22d3ee' },
      'PASTURE-SOUTH': { fill: '#f59e0b', stroke: '#fbbf24' },
      'PADDOCK-HOME': { fill: '#8b5cf6', stroke: '#a78bfa' }
    };

    function entityClassIcon(cls) {
      const c = String(cls || '').toLowerCase();
      if (c === 'cow' || c === 'horse') return '🐄';
      if (c === 'truck' || c === 'car') return '🚛';
      if (c === 'person') return '🚶';
      return '📍';
    }

    function uncertaintyRadiusM(elapsedS, confidence) {
      const c = Math.min(1.0, Math.max(0.1, confidence ?? 0.5));
      return 15.0 + 3.5 * Math.sqrt(Math.max(0, elapsedS)) * (1.0 - c);
    }

    function pointInRing(lon, lat, ring) {
      let inside = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const xi = ring[i][0], yi = ring[i][1];
        const xj = ring[j][0], yj = ring[j][1];
        const intersect = ((yi > lat) !== (yj > lat))
          && (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi);
        if (intersect) inside = !inside;
      }
      return inside;
    }

    function findZoneNameForPoint(lon, lat) {
      for (const f of spatialZones) {
        const ring = f.geometry && f.geometry.coordinates && f.geometry.coordinates[0];
        if (ring && pointInRing(lon, lat, ring)) {
          return f.properties.name || 'Unknown Zone';
        }
      }
      return '—';
    }

    function applyDomainProfile(profile) {
      if (!profile) return;
      domainProfile = profile;
      const titleEl = document.getElementById('header-app-title');
      if (titleEl) titleEl.textContent = profile.app_title || 'Rian Operations';
      const zh = document.getElementById('label-zones-header');
      if (zh) zh.textContent = profile.zone_label || 'Spatial Zones';
      const ch = document.getElementById('label-chokepoints-header');
      if (ch) ch.textContent = profile.node_label || 'Access Nodes';
      const dh = document.getElementById('label-directives-header');
      if (dh) dh.textContent = profile.task_label || 'Operational Tasks';
      const th = document.getElementById('label-tracks-header');
      if (th) th.textContent = profile.entity_label || 'Entity Tracks';
      document.title = (profile.app_title || 'Rian') + ' // COP';
    }

    async function loadDomainProfile() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/domain-profile`);
        if (res.ok) applyDomainProfile(await res.json());
      } catch (e) {
        console.warn('domain-profile fetch failed', e);
      }
    }

    function renderZonesSidebar() {
      const el = document.getElementById('zones-sidebar-list');
      if (!el) return;
      if (!spatialZones.length) {
        el.innerHTML = '<div class="text-slate-500">Loading zones...</div>';
        return;
      }
      el.innerHTML = spatialZones.map(f => {
        const p = f.properties || {};
        const hc = p.headcount != null ? p.headcount : 0;
        return `<div class="flex justify-between gap-2 py-0.5 border-b border-slate-800/60">
          <span class="truncate">${escapeHtml(p.name || p.zone_code)}</span>
          <span class="text-emerald-400 font-bold shrink-0">${hc} hd</span>
        </div>`;
      }).join('');
    }

    function renderZonePolygons() {
      layers.zones.clearLayers();
      spatialZones.forEach(f => {
        const code = (f.properties && f.properties.zone_code) || '';
        const style = ZONE_STYLE[code] || {
          fill: (f.properties && f.properties.color) || '#38bdf8',
          stroke: '#67e8f9'
        };
        const ring = (f.geometry.coordinates[0] || []).map(c => [c[1], c[0]]);
        if (!ring.length) return;
        L.polygon(ring, {
          color: style.stroke,
          fillColor: style.fill,
          fillOpacity: 0.22,
          weight: 1.5,
          dashArray: '4, 6'
        }).addTo(layers.zones);
        const center = ring.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0, 0])
          .map(v => v / ring.length);
        const hc = (f.properties && f.properties.headcount) || 0;
        L.marker(center, {
          icon: L.divIcon({
            className: 'zone-label-icon',
            html: `<div>${escapeHtml(f.properties.name)}<br><span style="color:#6ee7b7">${hc} head</span></div>`,
            iconSize: [120, 28],
            iconAnchor: [60, 14]
          }),
          interactive: false
        }).addTo(layers.zones);
      });
    }

    async function loadZones() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/zones`);
        const data = await res.json();
        if (res.ok && data.features) {
          spatialZones = data.features;
          renderZonePolygons();
          renderZonesSidebar();
        }
      } catch (e) {
        console.warn('zones fetch failed', e);
      }
    }

    function syncTrackKinematics(tracks) {
      const seen = new Set();
      tracks.forEach(t => {
        const id = t.track_id || t.id;
        if (!id) return;
        seen.add(id);
        const p = t.last_known_point || [0, 0];
        const lastSeenMs = t.last_detected_at ? Date.parse(t.last_detected_at) : Date.now();
        const heading = t.estimated_heading_deg || 0;
        const v = t.estimated_velocity_ms || 0;
        const br = heading * Math.PI / 180;
        const lat0 = p[1];
        const mLat = 111320;
        const mLon = mLat * Math.cos(lat0 * Math.PI / 180);
        const prev = trackKinematics.get(id);
        const snap = !prev || prev.lastSeenIso !== t.last_detected_at;
        trackKinematics.set(id, {
          trackId: id,
          trackCode: t.track_code,
          objectClass: t.object_class,
          baseLon: p[0],
          baseLat: p[1],
          vx: v * Math.sin(br) / mLon,
          vy: v * Math.cos(br) / mLat,
          headingDeg: heading,
          velocityMs: v,
          confidence: t.confidence_score ?? 0.5,
          lastSeenMs: Number.isFinite(lastSeenMs) ? lastSeenMs : Date.now(),
          lastSeenIso: t.last_detected_at,
          lastCamera: t.last_camera_name || '—',
          snap
        });
      });
      for (const id of trackKinematics.keys()) {
        if (!seen.has(id)) {
          trackKinematics.delete(id);
          const ent = entityRenderState.get(id);
          if (ent) {
            layers.entities.removeLayer(ent.marker);
            layers.entities.removeLayer(ent.halo);
            entityRenderState.delete(id);
          }
        }
      }
      ensureEntityMarkers();
    }

    function projectedPosition(kin, nowMs) {
      const elapsedS = Math.max(0, (nowMs - kin.lastSeenMs) / 1000);
      return {
        lon: kin.baseLon + kin.vx * elapsedS,
        lat: kin.baseLat + kin.vy * elapsedS,
        elapsedS
      };
    }

    function buildEntityTooltip(kin, lon, lat, elapsedS) {
      const speedKmh = (kin.velocityMs || 0) * 3.6;
      const zone = findZoneNameForPoint(lon, lat);
      const radius = uncertaintyRadiusM(elapsedS, kin.confidence);
      return `<div class="entity-rich-tooltip">
        <div class="tt-title">${escapeHtml(kin.trackCode)} · ${escapeHtml(String(kin.objectClass || '').toUpperCase())}</div>
        <div>Speed: <span class="text-cyan-300">${speedKmh.toFixed(1)} km/h</span> (${(kin.velocityMs || 0).toFixed(2)} m/s)</div>
        <div>Heading: <span class="text-cyan-300">${(kin.headingDeg || 0).toFixed(0)}°</span></div>
        <div>Zone: <span class="text-emerald-300">${escapeHtml(zone)}</span></div>
        <div>Uncertainty: <span class="text-amber-300">± ${radius.toFixed(1)} m</span></div>
        <div>Last Camera: <span class="text-slate-300">${escapeHtml(kin.lastCamera)}</span></div>
      </div>`;
    }

    function ensureEntityMarkers() {
      trackKinematics.forEach((kin, id) => {
        let ent = entityRenderState.get(id);
        const nowMs = Date.now();
        const pos = projectedPosition(kin, nowMs);
        if (!ent) {
          const icon = L.divIcon({
            className: '',
            html: `<div class="entity-marker-wrap"><div class="blinking-entity-dot">${entityClassIcon(kin.objectClass)}</div></div>`,
            iconSize: [28, 28],
            iconAnchor: [14, 14]
          });
          const marker = L.marker([pos.lat, pos.lon], { icon, zIndexOffset: 800 });
          const halo = L.circle([pos.lat, pos.lon], {
            radius: uncertaintyRadiusM(pos.elapsedS, kin.confidence),
            color: '#f87171',
            weight: 2,
            dashArray: '6, 8',
            fillColor: '#ef4444',
            fillOpacity: 0.08,
            className: 'uncertainty-halo-svg'
          });
          marker.on('mouseover', () => {
            const p = projectedPosition(kin, Date.now());
            marker.bindTooltip(buildEntityTooltip(kin, p.lon, p.lat, p.elapsedS), {
              direction: 'top', offset: [0, -12], className: 'entity-rich-tooltip', sticky: true
            }).openTooltip();
          });
          marker.on('mouseout', () => marker.closeTooltip());
          marker.on('click', () => openEntityInspector(id, pos.lat, pos.lon));
          marker.addTo(layers.entities);
          halo.addTo(layers.entities);
          ent = { marker, halo, kin };
          entityRenderState.set(id, ent);
        }
        ent.kin = kin;
      });
    }

    function updateEntityKinematicsFrame() {
      const nowMs = Date.now();
      entityRenderState.forEach((ent, id) => {
        const kin = trackKinematics.get(id);
        if (!kin) return;
        const pos = projectedPosition(kin, nowMs);
        ent.marker.setLatLng([pos.lat, pos.lon]);
        const r = uncertaintyRadiusM(pos.elapsedS, kin.confidence);
        ent.halo.setLatLng([pos.lat, pos.lon]);
        ent.halo.setRadius(r);
      });
      kinematicAnimHandle = requestAnimationFrame(updateEntityKinematicsFrame);
    }

    function startKinematicLoop() {
      if (kinematicAnimHandle == null) {
        kinematicAnimHandle = requestAnimationFrame(updateEntityKinematicsFrame);
      }
    }

    function openEntityModal() {
      document.getElementById('entity-modal').classList.add('open');
    }

    function closeEntityModal() {
      document.getElementById('entity-modal').classList.remove('open');
      selectedTrackId = null;
      currentEntitySnapshotId = null;
      currentSnapshotBoxes = [];
    }

    let currentEntitySnapshotId = null;
    let currentSnapshotBoxes = [];

    function resetEntitySnapshotUi() {
      currentEntitySnapshotId = null;
      currentSnapshotBoxes = [];
      document.getElementById('entity-snapshot-placeholder').classList.add('hidden');
      document.getElementById('entity-snapshot-viewer').classList.add('hidden');
      document.getElementById('entity-vlm-btn').classList.add('hidden');
      document.getElementById('entity-vlm-loading').classList.add('hidden');
      document.getElementById('entity-vlm-report').classList.add('hidden');
      document.getElementById('entity-snapshot-meta').textContent = '';
      const img = document.getElementById('entity-snapshot-img');
      img.removeAttribute('src');
      img.onload = null;
    }

    function drawSnapshotBoundingBoxes() {
      const img = document.getElementById('entity-snapshot-img');
      const canvas = document.getElementById('entity-snapshot-canvas');
      if (!img.naturalWidth || !currentSnapshotBoxes.length) return;
      const rect = img.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      currentSnapshotBoxes.forEach(box => {
        const x = box.xmin * rect.width;
        const y = box.ymin * rect.height;
        const w = (box.xmax - box.xmin) * rect.width;
        const h = (box.ymax - box.ymin) * rect.height;
        ctx.strokeStyle = box.color || '#22c55e';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        const label = `${box.label} ${Math.round((box.confidence || 0) * 100)}%`;
        ctx.font = '10px monospace';
        const tw = ctx.measureText(label).width + 6;
        ctx.fillStyle = box.color || '#22c55e';
        ctx.fillRect(x, Math.max(0, y - 14), tw, 14);
        ctx.fillStyle = '#0f172a';
        ctx.fillText(label, x + 3, Math.max(10, y - 4));
      });
    }

    function showEntitySnapshotRecord(data) {
      currentEntitySnapshotId = data.snapshot_id;
      currentSnapshotBoxes = data.bounding_boxes || [];
      document.getElementById('entity-snapshot-placeholder').classList.add('hidden');
      document.getElementById('entity-snapshot-viewer').classList.remove('hidden');
      document.getElementById('entity-vlm-btn').classList.remove('hidden');
      const ts = data.timestamp ? new Date(data.timestamp).toISOString().replace('T', ' ').slice(0, 19) + ' UTC' : '—';
      document.getElementById('entity-snapshot-meta').textContent =
        `${data.camera_name || 'Camera'} · ${ts} · ${Math.round((data.confidence || 0) * 100)}%`;
      const img = document.getElementById('entity-snapshot-img');
      img.onload = () => drawSnapshotBoundingBoxes();
      img.src = `${API_BASE}/api/v1/snapshots/${data.snapshot_id}/image?t=${Date.now()}`;
      document.getElementById('entity-vlm-report').classList.add('hidden');
    }

    async function loadEntitySnapshot(trackId) {
      resetEntitySnapshotUi();
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/entity/${trackId}/snapshot`);
        if (res.status === 404) {
          document.getElementById('entity-snapshot-placeholder').classList.remove('hidden');
          return;
        }
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Snapshot load failed');
        showEntitySnapshotRecord(data);
      } catch (e) {
        document.getElementById('entity-snapshot-placeholder').classList.remove('hidden');
      }
    }

    async function generateTestSnapshot() {
      if (!selectedTrackId) return;
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/entity/${selectedTrackId}/snapshot`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Generate failed');
        showEntitySnapshotRecord(data);
      } catch (e) {
        alert(String(e.message || e));
      }
    }

    async function runVlmAnalysis() {
      if (!currentEntitySnapshotId) return;
      document.getElementById('entity-vlm-loading').classList.remove('hidden');
      document.getElementById('entity-vlm-report').classList.add('hidden');
      try {
        const res = await fetch(`${API_BASE}/api/v1/snapshots/${currentEntitySnapshotId}/vlm-analyze`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'VLM analysis failed');
        document.getElementById('entity-vlm-risk').textContent = data.risk_assessment || '—';
        document.getElementById('entity-vlm-visual').textContent = data.visual_description || '';
        document.getElementById('entity-vlm-env').textContent = data.environmental_context || '';
        const findings = data.findings || [];
        document.getElementById('entity-vlm-findings').innerHTML = findings.map(f => `<li>${escapeHtml(f)}</li>`).join('');
        document.getElementById('entity-vlm-action').textContent = data.recommended_action ? '→ ' + data.recommended_action : '';
        document.getElementById('entity-vlm-provider').textContent = data.provider ? 'Vision: ' + data.provider : '';
        document.getElementById('entity-vlm-report').classList.remove('hidden');
      } catch (e) {
        alert(String(e.message || e));
      } finally {
        document.getElementById('entity-vlm-loading').classList.add('hidden');
      }
    }

    window.addEventListener('resize', () => {
      if (currentEntitySnapshotId) drawSnapshotBoundingBoxes();
    });

    async function openEntityInspector(trackId, lat, lon) {
      selectedTrackId = trackId;
      map.flyTo([lat, lon], Math.max(map.getZoom(), 16), { animate: true, duration: 1.2 });
      openEntityModal();
      document.getElementById('entity-modal-loading').classList.remove('hidden');
      document.getElementById('entity-headline').textContent = 'Analyzing entity telemetry...';
      document.getElementById('entity-insight').textContent = '';
      document.getElementById('entity-actions').innerHTML = '';
      document.getElementById('entity-telemetry').innerHTML = '';
      document.getElementById('entity-provider').textContent = '';
      resetEntitySnapshotUi();
      loadEntitySnapshot(trackId);
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/entity/${trackId}/insight`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Insight failed');
        document.getElementById('entity-risk-badge').textContent = data.risk_level || '—';
        document.getElementById('entity-headline').textContent = data.headline || data.track_code || 'Entity Diagnostic';
        document.getElementById('entity-insight').textContent = data.insight || '';
        const actions = data.recommended_actions || [];
        document.getElementById('entity-actions').innerHTML = actions.length
          ? actions.map(a => `<div>• ${escapeHtml(a)}</div>`).join('') : '';
        const tel = data.telemetry || {};
        document.getElementById('entity-telemetry').innerHTML = `
          <div>Speed: <span class="text-cyan-300">${tel.speed_kmh != null ? tel.speed_kmh + ' km/h' : '—'}</span></div>
          <div>Heading: <span class="text-cyan-300">${tel.heading_deg != null ? tel.heading_deg + '°' : '—'}</span></div>
          <div>Elapsed: <span class="text-amber-300">${tel.elapsed_seconds != null ? tel.elapsed_seconds + ' s' : '—'}</span></div>
          <div>Uncertainty: <span class="text-amber-300">± ${tel.uncertainty_radius_m != null ? tel.uncertainty_radius_m + ' m' : '—'}</span></div>
          <div class="col-span-2">Zone: <span class="text-emerald-300">${escapeHtml(data.current_zone || '—')}</span></div>`;
        document.getElementById('entity-provider').textContent = data.provider ? 'AI: ' + data.provider : '';
      } catch (e) {
        document.getElementById('entity-headline').textContent = 'Diagnostic unavailable';
        document.getElementById('entity-insight').textContent = String(e.message || e);
      } finally {
        document.getElementById('entity-modal-loading').classList.add('hidden');
      }
    }

    // Map initialization (compact demo estate)
    const map = L.map('map', { zoomControl: false }).setView([36.7465, -119.82], 15);
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Basemap layers
    const satelliteTile = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
      attribution: '&copy; Esri World Imagery'
    }).addTo(map);

    const darkTile = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      className: 'dark-tiles',
      attribution: '&copy; OpenStreetMap'
    });

    let currentBasemap = 'satellite';
    function switchBasemap(mode) {
      if (mode === 'dark' && currentBasemap !== 'dark') {
        map.removeLayer(satelliteTile);
        map.addLayer(darkTile);
        document.getElementById('btn-dark').className = 'px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-bold border border-cyan-700/50';
        document.getElementById('btn-satellite').className = 'px-2 py-0.5 rounded text-slate-400 hover:text-slate-200';
        currentBasemap = 'dark';
      } else if (mode === 'satellite' && currentBasemap !== 'satellite') {
        map.removeLayer(darkTile);
        map.addLayer(satelliteTile);
        document.getElementById('btn-satellite').className = 'px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-bold border border-cyan-700/50';
        document.getElementById('btn-dark').className = 'px-2 py-0.5 rounded text-slate-400 hover:text-slate-200';
        currentBasemap = 'satellite';
      }
    }

    const layers = {
      zones: L.layerGroup().addTo(map),
      reach: L.layerGroup().addTo(map),
      routes: L.layerGroup().addTo(map),
      choke: L.layerGroup().addTo(map),
      tracks: L.layerGroup().addTo(map),
      sensors: L.layerGroup().addTo(map),
      entities: L.layerGroup().addTo(map),
      predators: L.layerGroup().addTo(map)
    };

    let currentState = null;
    let lastPredatorSummary = null;

    // Clock
    setInterval(() => {
      const now = new Date();
      document.getElementById('clock-utc').textContent = now.toUTCString().slice(17, 25) + ' UTC';
      document.getElementById('clock-local').textContent = now.toTimeString().slice(0, 8) + ' LOCAL';
    }, 1000);

    // Map click: pin pick for camera provision OR relocate response unit
    map.on('click', async (e) => {
      const lon = e.latlng.lng;
      const lat = e.latlng.lat;
      if (zoneDrawMode) {
        zoneDrawVertices.push([lon, lat]);
        refreshZoneDrawGraphics();
        return;
      }
      if (mapPickMode) {
        document.getElementById('cam-lon').value = lon.toFixed(6);
        document.getElementById('cam-lat').value = lat.toFixed(6);
        toggleMapPickMode(false);
        openAdminModal();
        switchAdminTab('cameras');
        return;
      }
      await fetch(`${API_BASE}/api/v1/interceptor/telemetry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lon, lat, unit_id: 'unit-alpha' })
      });
      loadState();
    });

    function renderState(state) {
      currentState = state;
      layers.sensors.clearLayers();
      layers.tracks.clearLayers();
      layers.reach.clearLayers();
      layers.choke.clearLayers();
      layers.routes.clearLayers();

      const sensors = state.sensors || [];
      const tracks = state.tracks || [];
      const reachability = state.reachability || [];
      const chokepoints = state.chokepoints || [];
      const directives = state.directives || [];

      // KPIs
      document.getElementById('kpi-tracks').textContent = tracks.length;
      document.getElementById('kpi-sensors').textContent = sensors.length;
      document.getElementById('kpi-chokepoints').textContent = chokepoints.length;
      document.getElementById('kpi-directives').textContent = directives.length;
      document.getElementById('fleet-count').textContent = sensors.length + ' NODES';
      document.getElementById('active-tracks-count').textContent = tracks.length + ' ACTIVE';

      if (directives.length > 0) {
        document.getElementById('threat-indicator').className = 'text-[9px] mono bg-red-950 text-red-300 px-1.5 py-0.5 rounded border border-red-500 animate-pulse';
        document.getElementById('threat-indicator').textContent = 'DIRECTIVE ACTIVE';
      } else {
        document.getElementById('threat-indicator').className = 'text-[9px] mono bg-slate-900 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700';
        document.getElementById('threat-indicator').textContent = 'STANDBY';
      }

      // Render Sensors
      const sensorsListHtml = sensors.map(s => {
        const active = s.status === 'active';
        return `
          <div class="p-2 rounded bg-slate-900/80 border border-slate-800 hover:border-cyan-500/50 transition-colors flex items-center justify-between">
            <div>
              <div class="font-bold text-slate-200 mono flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full ${active ? 'bg-emerald-400' : 'bg-red-400'}"></span>
                ${s.node_name}
              </div>
              <div class="text-[10px] text-slate-400 mono mt-0.5">Azimuth: ${s.azimuth_deg}° | FOV: ${s.fov_angle}°</div>
            </div>
            <div class="text-right">
              <span class="text-[10px] mono px-1.5 py-0.5 rounded ${active ? 'bg-emerald-950 text-emerald-300' : 'bg-red-950 text-red-300'}">${s.status.toUpperCase()}</span>
              <button onclick="flyToCoord(${s.location[1]}, ${s.location[0]})" class="block text-[10px] text-cyan-400 hover:underline mt-1">Locate</button>
            </div>
          </div>
        `;
      }).join('');
      document.getElementById('sensors-list').innerHTML = sensorsListHtml || '<div class="text-slate-500 text-center py-4">No sensors found</div>';

      sensors.forEach(s => {
        const active = s.status === 'active';
        const marker = L.circleMarker([s.location[1], s.location[0]], {
          radius: 7,
          color: active ? '#10b981' : '#ef4444',
          fillColor: active ? '#34d399' : '#f87171',
          fillOpacity: 0.9,
          weight: 2
        }).addTo(layers.sensors);
        marker.bindPopup(`<b>${s.node_name}</b><br>Status: ${s.status}<br>Azimuth: ${s.azimuth_deg}°`);

        if (s.fov_wedge) {
          const ring = s.fov_wedge.coordinates[0].map(c => [c[1], c[0]]);
          L.polygon(ring, {
            color: '#38bdf8',
            fillColor: '#0284c7',
            fillOpacity: 0.18,
            weight: 1.5,
            dashArray: '3, 4'
          }).addTo(layers.sensors);
        }
      });

      // Render Chokepoints
      const chokeHtml = chokepoints.map(c => `
        <div class="truncate py-0.5 text-slate-300">⚓ ${c.name}</div>
      `).join('');
      document.getElementById('chokepoints-mini').innerHTML = chokeHtml;

      chokepoints.forEach(c => {
        const marker = L.circleMarker([c.location[1], c.location[0]], {
          radius: 8,
          color: '#38bdf8',
          fillColor: '#0284c7',
          fillOpacity: 0.85,
          weight: 2
        }).addTo(layers.choke);
        marker.bindPopup(`<b>CHOKEPOINT: ${c.name}</b><br>Type: ${c.chokepoint_type || 'Perimeter Gate'}`);
      });

      // Render Tracks
      const tracksHtml = tracks.map(t => {
        const speedKmh = t.estimated_velocity_ms ? (t.estimated_velocity_ms * 3.6).toFixed(1) : '—';
        return `
          <div class="p-2.5 rounded bg-red-950/30 border border-red-900/50">
            <div class="flex items-center justify-between">
              <span class="mono font-bold text-red-400">${t.track_code}</span>
              <span class="mono text-[10px] bg-red-900/60 text-red-200 px-1.5 py-0.2 rounded">${(t.object_class || 'TARGET').toUpperCase()}</span>
            </div>
            <div class="text-[10px] mono text-slate-300 mt-1 grid grid-cols-2 gap-1">
              <div>Velocity: <span class="text-cyan-300">${speedKmh} km/h</span></div>
              <div>Heading: <span class="text-cyan-300">${t.estimated_heading_deg != null ? t.estimated_heading_deg.toFixed(0) + '°' : '—'}</span></div>
              <div>Confidence: <span class="text-emerald-300">${((t.confidence_score || 1) * 100).toFixed(0)}%</span></div>
              <div>Status: <span class="text-amber-300">${t.status}</span></div>
            </div>
          </div>
        `;
      }).join('');
      document.getElementById('tracks-list').innerHTML = tracksHtml || '<div class="text-slate-500 text-center py-6">Perimeter secure. Zero unassociated radar gates.</div>';

      syncTrackKinematics(tracks);
      startKinematicLoop();
      loadZones();

      tracks.forEach(t => {
        if (t.history && t.history.coordinates) {
          const latlngs = t.history.coordinates.map(c => [c[1], c[0]]);
          L.polyline(latlngs, { color: '#f59e0b', weight: 3, dashArray: '6, 6' }).addTo(layers.tracks);
        }
      });

      // Isochrone Reachability
      reachability.forEach(f => {
        if (!f.geometry) return;
        const ring = (f.geometry.coordinates[0] || []).map(c => [c[1], c[0]]);
        if (ring.length) {
          L.polygon(ring, {
            color: '#eab308',
            fillColor: '#ca8a04',
            fillOpacity: 0.22,
            weight: 2
          }).addTo(layers.reach);
        }
      });

      // Directives Card
      const directivesHtml = directives.map(d => {
        const marginMin = (d.tactical_margin_seconds / 60.0).toFixed(1);
        return `
          <div class="p-2.5 rounded bg-amber-950/40 border border-amber-600/60 shadow-lg">
            <div class="flex items-center justify-between">
              <span class="mono font-bold text-amber-300 text-xs">INTERCEPT DIRECTIVE</span>
              <span class="mono text-[10px] bg-amber-900/80 text-amber-200 px-1.5 py-0.5 rounded">${d.status}</span>
            </div>
            <div class="mono text-xs font-bold text-cyan-300 mt-1">OBJECTIVE: ${d.chokepoint_name}</div>
            <div class="text-[10px] mono text-slate-300 mt-0.5">
              Tactical Margin: <span class="text-emerald-400 font-bold">+${marginMin} min advantage</span>
            </div>
            <div class="flex items-center gap-1.5 mt-2">
              <button onclick="updateDirectiveStatus('${d.directive_id}', 'ACKNOWLEDGED')" class="flex-1 py-1 text-[10px] mono font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-600">ACK</button>
              <button onclick="updateDirectiveStatus('${d.directive_id}', 'ON_SCENE')" class="flex-1 py-1 text-[10px] mono font-bold bg-cyan-900 hover:bg-cyan-800 text-cyan-200 rounded border border-cyan-600">ON SCENE</button>
              <button onclick="updateDirectiveStatus('${d.directive_id}', 'RESOLVED')" class="flex-1 py-1 text-[10px] mono font-bold bg-emerald-950 hover:bg-emerald-900 text-emerald-300 rounded border border-emerald-600">RESOLVE</button>
            </div>
          </div>
        `;
      }).join('');
      document.getElementById('directives-container').innerHTML = directivesHtml || '<div class="text-slate-500 text-xs text-center py-4">No active intercept directives</div>';

      directives.forEach(d => {
        if (d.response_route && d.response_route.coordinates) {
          const latlngs = d.response_route.coordinates.map(c => [c[1], c[0]]);
          L.polyline(latlngs, { color: '#06b6d4', weight: 4 }).addTo(layers.routes);
        }
      });

      // Interceptor Position
      if (state.interceptor && state.interceptor.location) {
        const p = state.interceptor.location;
        const icon = L.circleMarker([p[1], p[0]], {
          radius: 10,
          color: '#06b6d4',
          fillColor: '#22d3ee',
          fillOpacity: 1.0,
          weight: 3
        }).addTo(layers.routes);
        icon.bindPopup('<b>RESPONSE PATROL UNIT</b><br>Dijkstra Dynamic Router Active');
      }
    }

    function flyToCoord(lat, lon) {
      map.flyTo([lat, lon], 15, { animate: true, duration: 1 });
    }

    async function fetchAiSitrep() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/ai-sitrep`, { method: 'POST' });
        const report = await res.json();
        const threat = report.threat_status || report.threat_level || 'SECURE';
        const defcon = report.defcon_level != null ? report.defcon_level : 5;
        const badgeEl = document.getElementById('sitrep-badge');
        badgeEl.textContent = `DEFCON ${defcon} // ${threat}`;
        const threatUpper = String(threat).toUpperCase();
        if (threatUpper.includes('RED') || defcon <= 2) {
          badgeEl.className = 'mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-red-950 text-red-300 border border-red-700/50 animate-pulse';
        } else if (threatUpper.includes('AMBER') || defcon === 3) {
          badgeEl.className = 'mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-700/50';
        } else {
          badgeEl.className = 'mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-700/50';
        }
        document.getElementById('sitrep-time').textContent = report.timestamp_utc || report.timestamp || '';
        document.getElementById('sitrep-headline').textContent = report.headline || '';
        document.getElementById('sitrep-summary').textContent = report.target_summary || report.summary || '';
        const actions = report.recommended_actions || report.action_items || [];
        if (actions.length) {
          document.getElementById('sitrep-actions').innerHTML = actions.map(a => `<div>• ${escapeHtml(a)}</div>`).join('');
        } else {
          document.getElementById('sitrep-actions').innerHTML = '';
        }
        const provider = report.engine_provider || report.provider;
        if (provider) {
          document.getElementById('sitrep-provider').textContent = 'Engine: ' + provider;
        }
      } catch (e) {
        console.error('Error fetching SITREP:', e);
      }
    }

    async function sendAiQuery() {
      const inp = document.getElementById('ai-query-input');
      const val = inp.value.trim();
      if (!val) return;
      const out = document.getElementById('ai-query-output');
      out.classList.remove('hidden');
      out.textContent = 'Tactical Copilot reasoning...';
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/ai-query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: val })
        });
        const data = await res.json();
        const responseText = typeof data.response === 'string'
          ? data.response
          : JSON.stringify(data.response, null, 2);
        out.innerHTML = `<b>&gt; ${escapeHtml(data.query)}</b><br><span class="text-slate-200 mt-1 block whitespace-pre-wrap break-words leading-relaxed">${formatCopilotText(responseText)}</span>`;
      } catch (e) {
        out.textContent = 'Error connecting to Tactical Copilot.';
      }
    }

    function trackLatLngById(trackId) {
      const kin = trackKinematics.get(trackId);
      if (kin) {
        const pos = projectedPosition(kin, Date.now());
        return [pos.lat, pos.lon];
      }
      const tracks = (currentState && currentState.tracks) || [];
      const row = tracks.find(t => (t.track_id || t.id) === trackId);
      if (row && row.last_known_point) {
        return [row.last_known_point[1], row.last_known_point[0]];
      }
      return null;
    }

    function renderPredatorThreatUi(summary) {
      lastPredatorSummary = summary;
      const banner = document.getElementById('predator-alert-banner');
      layers.predators.clearLayers();
      const threats = (summary && summary.threats) || [];
      if (!threats.length) {
        banner.classList.add('hidden');
        banner.innerHTML = '';
        return;
      }
      const primary = threats[0];
      const clsLabel = (primary.predator_class || 'predator').replace(/^./, c => c.toUpperCase());
      banner.classList.remove('hidden');
      banner.innerHTML = `⚠️ PREDATOR ALERT: ${escapeHtml(clsLabel)} ${primary.distance_meters.toFixed(0)}m from ${escapeHtml(primary.nearest_livestock_code)}
        <button type="button" onclick="triggerDeterrent('${primary.threat_id}')" class="px-2 py-0.5 rounded bg-red-900 hover:bg-red-800 border border-red-400 text-red-50 text-[10px] font-bold">🔊 Sound Siren</button>`;

      threats.forEach(threat => {
        const predId = threat.predator_track_id;
        const herdId = threat.nearest_livestock_id;
        const predLl = trackLatLngById(predId);
        const herdLl = trackLatLngById(herdId);
        if (!predLl || !herdLl) return;
        const line = L.polyline([predLl, herdLl], {
          color: '#dc2626',
          weight: 3,
          dashArray: '8, 10',
          className: 'predator-threat-line'
        }).addTo(layers.predators);
        const midLat = (predLl[0] + herdLl[0]) / 2;
        const midLon = (predLl[1] + herdLl[1]) / 2;
        L.marker([midLat, midLon], {
          icon: L.divIcon({
            className: 'predator-threat-label',
            html: `<div style="background:rgba(127,29,29,0.92);color:#fecaca;padding:2px 6px;border-radius:4px;font-size:10px;font-family:monospace;border:1px solid #f87171;">${threat.distance_meters.toFixed(0)}m</div>`,
            iconSize: [48, 18],
            iconAnchor: [24, 9]
          }),
          interactive: false
        }).addTo(layers.predators);
        line.bringToFront();
      });
    }

    async function refreshPredatorThreats() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/predators/threats`);
        if (!res.ok) return;
        const summary = await res.json();
        renderPredatorThreatUi(summary);
      } catch (e) {
        console.warn('predator threats fetch failed', e);
      }
    }

    async function triggerDeterrent(threatId) {
      await fetch(`${API_BASE}/api/v1/predators/${threatId}/trigger-deterrent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deterrent_type: 'acoustic_siren' })
      });
      await refreshPredatorThreats();
    }

    async function simulatePredatorIncursion() {
      const btn = event && event.target;
      const originalText = btn ? btn.textContent : '';
      if (btn) btn.textContent = 'Seeding...';
      try {
        await fetch(`${API_BASE}/api/v1/predators/simulate-incursion`, { method: 'POST' });
        await loadState();
        await refreshPredatorThreats();
        await fetchAiSitrep();
      } finally {
        if (btn) btn.textContent = originalText;
      }
    }

    function rollcallStatusBadgeClass(status) {
      if (status === 'VERIFIED_ACTIVE') return 'rollcall-status-verified';
      if (status === 'OVERDUE_CHECK') return 'rollcall-status-overdue';
      return 'rollcall-status-missing';
    }

    function rollcallStatusLabel(status) {
      if (status === 'VERIFIED_ACTIVE') return 'Verified Active';
      if (status === 'OVERDUE_CHECK') return 'Overdue Check';
      return 'Missing Critical';
    }

    function openRollcallModal() {
      document.getElementById('rollcall-modal').classList.add('open');
      refreshRollcallReport();
    }

    function closeRollcallModal() {
      document.getElementById('rollcall-modal').classList.remove('open');
    }

    async function refreshRollcallReport() {
      const loading = document.getElementById('rollcall-loading');
      const tbody = document.getElementById('rollcall-tbody');
      loading.classList.remove('hidden');
      try {
        const res = await fetch(`${API_BASE}/api/v1/reports/herd-rollcall`);
        if (!res.ok) throw new Error('Roll-call fetch failed');
        const data = await res.json();
        const items = data.items || [];
        const verified = items.filter(i => i.health_status === 'VERIFIED_ACTIVE').length;
        const overdue = items.filter(i => i.health_status === 'OVERDUE_CHECK').length;
        const missing = items.filter(i => i.health_status === 'MISSING_CRITICAL').length;
        document.getElementById('rollcall-metric-total').textContent = String(data.total_tracked_count ?? items.length);
        document.getElementById('rollcall-metric-verified').textContent = String(verified);
        document.getElementById('rollcall-metric-overdue').textContent = String(overdue);
        document.getElementById('rollcall-metric-missing').textContent = String(missing);
        document.getElementById('rollcall-generated-at').textContent = data.generated_at_utc
          ? `Generated ${data.generated_at_utc} UTC · Profile ${data.profile_type || '—'}`
          : '—';
        if (!items.length) {
          tbody.innerHTML = '<tr><td colspan="7" class="px-2 py-6 text-center text-slate-500">No tracked entities in current COP state.</td></tr>';
        } else {
          tbody.innerHTML = items.map(item => {
            const badgeCls = rollcallStatusBadgeClass(item.health_status);
            const label = rollcallStatusLabel(item.health_status);
            const vlm = (item.vlm_summary || '—').replace(/</g, '&lt;');
            return `<tr>
              <td class="px-2 py-2 font-bold text-cyan-300">${item.track_code}</td>
              <td class="px-2 py-2">${item.entity_class}</td>
              <td class="px-2 py-2">${item.current_zone}</td>
              <td class="px-2 py-2">${item.last_camera_name || '—'}</td>
              <td class="px-2 py-2">${Number(item.hours_since_last_seen).toFixed(1)}</td>
              <td class="px-2 py-2"><span class="px-2 py-0.5 rounded text-[10px] ${badgeCls}">${label}</span></td>
              <td class="px-2 py-2 text-slate-400 max-w-xs">${vlm}</td>
            </tr>`;
          }).join('');
        }
      } catch (e) {
        console.warn('rollcall report failed', e);
        tbody.innerHTML = '<tr><td colspan="7" class="px-2 py-6 text-center text-red-400">Failed to load roll-call report.</td></tr>';
      } finally {
        loading.classList.add('hidden');
      }
    }

    function exportRollcallCsv() {
      window.location.href = `${API_BASE}/api/v1/reports/herd-rollcall/csv`;
    }

    async function loadState() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/state`);
        const state = await res.json();
        renderState(state);
        await refreshPredatorThreats();
      } catch (e) {
        console.error('Error fetching state:', e);
      }
    }

    async function updateDirectiveStatus(id, status) {
      await fetch(`${API_BASE}/api/v1/directives/${id}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      loadState();
      fetchAiSitrep();
    }

    async function runScenario(name) {
      const btn = event.target;
      const originalText = btn.textContent;
      btn.textContent = 'Running...';
      try {
        await fetch(`${API_BASE}/api/v1/sim/run-scenario`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name })
        });
        await loadState();
        await fetchAiSitrep();
      } finally {
        btn.textContent = originalText;
      }
    }

    async function resetSimulation() {
      await fetch(`${API_BASE}/api/v1/sim/reset`, { method: 'POST' });
      await loadState();
      await fetchAiSitrep();
    }

    async function injectDetection() {
      const camera_name = document.getElementById('sel-camera').value;
      const entity_class = document.getElementById('sel-class').value;
      await fetch(`${API_BASE}/api/v1/sim/inject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_name, entity_class })
      });
      await loadState();
      await fetchAiSitrep();
    }

    // Initial load and SSE stream
    initAuthGate();
    loadDomainProfile();
    loadZones();
    loadState().then(() => fetchAiSitrep());
    startKinematicLoop();
    const es = new EventSource(`${API_BASE}/api/v1/cop/stream`);
    es.addEventListener('cop_state', e => {
      renderState(JSON.parse(e.data));
      fetchAiSitrep();
    });
    es.addEventListener('cop_update', () => {
      loadState();
      fetchAiSitrep();
    });
  </script>
</body>
</html>
"""
