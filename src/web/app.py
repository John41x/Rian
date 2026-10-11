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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
            id,
            track_code,
            object_class,
            current_status,
            ST_X(last_known_point) AS lon,
            ST_Y(last_known_point) AS lat,
            estimated_heading_deg,
            estimated_velocity_ms,
            confidence_score,
            ST_AsGeoJSON(history_linestring) AS history_json,
            ST_AsGeoJSON(reachability_polygon) AS reach_json
        FROM entity_tracks
        WHERE current_status = 'ACTIVE'
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
    .glass-panel { background: rgba(10, 15, 30, 0.88); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.2); }
    .pulse-target { animation: pulseRing 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite; }
    @keyframes pulseRing {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { transform: scale(1.05); box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .leaflet-popup-content-wrapper { background: rgba(15, 23, 42, 0.95) !important; color: #f8fafc !important; border: 1px solid #38bdf8; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .leaflet-popup-tip { background: #0f172a !important; }
  </style>
</head>
<body class="flex flex-col h-screen select-none">

  <!-- TOP HUD BAR -->
  <header class="h-14 bg-slate-950/95 border-b border-cyan-900/40 px-4 flex items-center justify-between z-50">
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2">
        <span class="mono font-bold text-lg tracking-wider text-cyan-400 bg-cyan-950/60 border border-cyan-500/40 px-2.5 py-0.5 rounded shadow-sm">[ RIAN ]</span>
        <span class="text-xs font-semibold text-slate-300 tracking-wider hidden sm:inline">AUTONOMOUS SPATIAL STATE ENGINE</span>
        <span class="text-[10px] mono text-cyan-500/70 border-l border-cyan-800/40 pl-2">v1.0.0 // TACTICAL C2</span>
      </div>
      <div id="live-indicator" class="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 text-[11px] mono">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
        <span id="conn-status">LIVE SSE STREAM</span>
      </div>
    </div>

    <!-- STATS COUNTERS -->
    <div class="flex items-center gap-5 text-xs mono">
      <div class="text-center">
        <div class="text-[10px] text-slate-400 uppercase">Active Targets</div>
        <div id="kpi-tracks" class="font-bold text-red-400 text-sm">0</div>
      </div>
      <div class="text-center">
        <div class="text-[10px] text-slate-400 uppercase">Sensor Nodes</div>
        <div id="kpi-sensors" class="font-bold text-emerald-400 text-sm">0</div>
      </div>
      <div class="text-center">
        <div class="text-[10px] text-slate-400 uppercase">Chokepoints</div>
        <div id="kpi-chokepoints" class="font-bold text-sky-400 text-sm">0</div>
      </div>
      <div class="text-center">
        <div class="text-[10px] text-slate-400 uppercase">Directives</div>
        <div id="kpi-directives" class="font-bold text-amber-400 text-sm">0</div>
      </div>
      <div class="text-right pl-3 border-l border-slate-800 hidden md:block">
        <div id="clock-utc" class="text-xs font-bold text-cyan-300">00:00:00 UTC</div>
        <div id="clock-local" class="text-[10px] text-slate-400">00:00:00 LOCAL</div>
      </div>
      <button onclick="promptApiUrl()" title="Configure Backend API URL" class="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-[10px] mono text-cyan-400 border border-cyan-800/50">⚙ API</button>
    </div>
  </header>

  <!-- MAIN VIEWPORT -->
  <div class="relative flex-1 flex overflow-hidden">

    <!-- LEFT SIDEBAR: FLEET & PERIMETER -->
    <aside id="left-sidebar" class="absolute top-3 left-3 bottom-16 w-80 glass-panel rounded-xl shadow-2xl z-40 flex flex-col transition-all duration-300">
      <div class="p-3 border-b border-cyan-900/40 flex items-center justify-between">
        <span class="mono text-xs font-bold text-cyan-400 tracking-wider">ESTATE SURVEILLANCE FLEET</span>
        <span id="fleet-count" class="text-[10px] mono bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800/40">6 NODES</span>
      </div>
      <div id="sensors-list" class="flex-1 overflow-y-auto p-2.5 space-y-2 text-xs">
        <!-- Rendered dynamically -->
        <div class="text-slate-500 text-center py-6">Connecting to sensor fabric...</div>
      </div>

      <div class="p-3 border-t border-cyan-900/40 bg-slate-950/60 rounded-b-xl">
        <div class="text-[11px] mono font-semibold text-slate-300 mb-1 flex items-center justify-between">
          <span>STRATEGIC CHOKEPOINTS</span>
          <span class="text-[10px] text-sky-400">PERIMETER</span>
        </div>
        <div id="chokepoints-mini" class="grid grid-cols-2 gap-1 text-[10px] mono text-slate-400">
          <!-- Filled dynamically -->
        </div>
      </div>
    </aside>

    <!-- RIGHT SIDEBAR: DIRECTIVES & THREAT TRACKER -->
    <aside id="right-sidebar" class="absolute top-3 right-3 bottom-16 w-88 glass-panel rounded-xl shadow-2xl z-40 flex flex-col transition-all duration-300">
      <div class="p-3 border-b border-cyan-900/40 flex items-center justify-between">
        <span class="mono text-xs font-bold text-red-400 tracking-wider">TACTICAL DIRECTIVES // THREATS</span>
        <span id="threat-indicator" class="text-[10px] mono bg-slate-900 text-slate-400 px-2 py-0.5 rounded border border-slate-700">STANDBY</span>
      </div>

      <!-- DIRECTIVES FEED -->
      <div id="directives-container" class="p-2.5 space-y-2 border-b border-cyan-900/30 max-h-56 overflow-y-auto">
        <div class="text-slate-500 text-xs text-center py-4">No active intercept directives</div>
      </div>

      <!-- ACTIVE TRACKS -->
      <div class="px-3 pt-2 pb-1 text-[11px] mono text-slate-400 font-semibold flex items-center justify-between">
        <span>KINEMATIC TRACKS (HMM)</span>
        <span id="active-tracks-count" class="text-red-400">0 ACTIVE</span>
      </div>
      <div id="tracks-list" class="flex-1 overflow-y-auto p-2.5 space-y-2 text-xs">
        <div class="text-slate-500 text-center py-6">Perimeter secure. Zero unassociated radar gates.</div>
      </div>
    </aside>

    <!-- MAP CONTAINER -->
    <div id="map"></div>

    <!-- BOTTOM FLOATING SIMULATION & CONTROL BAR -->
    <div class="absolute bottom-3 left-3 right-3 h-12 glass-panel rounded-xl shadow-2xl z-40 flex items-center justify-between px-4">
      <div class="flex items-center gap-2">
        <span class="text-[11px] mono text-slate-400 font-bold mr-1">SIMULATION ENGINE:</span>
        <button onclick="runScenario('vehicle_intrusion')" class="px-2.5 py-1 text-xs mono font-semibold bg-red-950/80 hover:bg-red-900 text-red-300 border border-red-700/50 rounded transition-colors">
          ► Vehicle Intrusion (North Gate)
        </button>
        <button onclick="runScenario('pedestrian_hike')" class="px-2.5 py-1 text-xs mono font-semibold bg-amber-950/80 hover:bg-amber-900 text-amber-300 border border-amber-700/50 rounded transition-colors">
          ► Pedestrian Hike
        </button>
        <button onclick="resetSimulation()" class="px-2.5 py-1 text-xs mono font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 rounded transition-colors">
          ↺ Reset Tracks
        </button>
      </div>

      <div class="flex items-center gap-3 text-xs mono">
        <!-- Manual Camera Trigger Form -->
        <div class="flex items-center gap-1.5 bg-slate-900/80 px-2 py-1 rounded border border-slate-700">
          <label class="text-[10px] text-slate-400">Inject:</label>
          <select id="sel-camera" class="bg-slate-950 text-cyan-300 text-xs rounded border border-slate-700 px-1 py-0.5 outline-none">
            <option value="cam-gate-north">cam-gate-north</option>
            <option value="cam-gate-south">cam-gate-south</option>
            <option value="cam-gate-east">cam-gate-east</option>
            <option value="cam-gate-west">cam-gate-west</option>
            <option value="cam-intersection">cam-intersection</option>
          </select>
          <select id="sel-class" class="bg-slate-950 text-amber-300 text-xs rounded border border-slate-700 px-1 py-0.5 outline-none">
            <option value="truck">Truck</option>
            <option value="person">Person</option>
            <option value="car">Car</option>
          </select>
          <button onclick="injectDetection()" class="px-2 py-0.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded text-[11px] transition-colors">
            Emit
          </button>
        </div>

        <span class="text-[10px] text-slate-400 hidden xl:inline">Click map to relocate Response Unit</span>
      </div>
    </div>

  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const API_BASE = (window.location.origin.includes('vercel.app') || window.location.protocol === 'file:')
      ? (localStorage.getItem('RIAN_BACKEND_URL') || 'http://localhost:8000')
      : '';

    function promptApiUrl() {
      const current = localStorage.getItem('RIAN_BACKEND_URL') || (window.location.origin.includes('vercel.app') ? 'http://localhost:8000' : window.location.origin);
      const next = prompt('Enter Rian Engine Backend URL (e.g. http://localhost:8000 or https://your-domain.com):', current);
      if (next !== null) {
        localStorage.setItem('RIAN_BACKEND_URL', next.replace(/\/$/, ''));
        window.location.reload();
      }
    }

    // Map initialization with dark tactical tiles
    const map = L.map('map', { zoomControl: false }).setView([36.745, -119.82], 13);
    L.control.zoom({ position: 'topright' }).addTo(map);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
      attribution: '&copy; CartoDB &copy; OpenStreetMap'
    }).addTo(map);

    const layers = {
      sensors: L.layerGroup().addTo(map),
      tracks: L.layerGroup().addTo(map),
      reach: L.layerGroup().addTo(map),
      choke: L.layerGroup().addTo(map),
      routes: L.layerGroup().addTo(map)
    };

    let currentState = null;

    // Clock
    setInterval(() => {
      const now = new Date();
      document.getElementById('clock-utc').textContent = now.toUTCString().slice(17, 25) + ' UTC';
      document.getElementById('clock-local').textContent = now.toTimeString().slice(0, 8) + ' LOCAL';
    }, 1000);

    // Map click sets responder unit location
    map.on('click', async (e) => {
      const lon = e.latlng.lng;
      const lat = e.latlng.lat;
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
        document.getElementById('threat-indicator').className = 'text-[10px] mono bg-red-950 text-red-300 px-2 py-0.5 rounded border border-red-500 animate-pulse';
        document.getElementById('threat-indicator').textContent = 'DIRECTIVE ACTIVE';
      } else {
        document.getElementById('threat-indicator').className = 'text-[10px] mono bg-slate-900 text-slate-400 px-2 py-0.5 rounded border border-slate-700';
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

      tracks.forEach(t => {
        const p = t.last_known_point;
        const marker = L.circleMarker([p[1], p[0]], {
          radius: 9,
          color: '#ef4444',
          fillColor: '#f87171',
          fillOpacity: 0.9,
          weight: 3,
          className: 'pulse-target'
        }).addTo(layers.tracks);
        marker.bindPopup(`<b>TRACK ${t.track_code}</b><br>Class: ${t.object_class}<br>Speed: ${t.estimated_velocity_ms ? (t.estimated_velocity_ms * 3.6).toFixed(1) + ' km/h' : '—'}`);

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

    async function loadState() {
      try {
        const res = await fetch(`${API_BASE}/api/v1/cop/state`);
        const state = await res.json();
        renderState(state);
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
        loadState();
      } finally {
        btn.textContent = originalText;
      }
    }

    async function resetSimulation() {
      await fetch(`${API_BASE}/api/v1/sim/reset`, { method: 'POST' });
      loadState();
    }

    async function injectDetection() {
      const camera_name = document.getElementById('sel-camera').value;
      const entity_class = document.getElementById('sel-class').value;
      await fetch(`${API_BASE}/api/v1/sim/inject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_name, entity_class })
      });
      loadState();
    }

    // Initial load and SSE stream
    loadState();
    const es = new EventSource(`${API_BASE}/api/v1/cop/stream`);
    es.addEventListener('cop_state', e => renderState(JSON.parse(e.data)));
    es.addEventListener('cop_update', () => loadState());
  </script>
</body>
</html>
"""
