"""
FastAPI Common Operating Picture (COP) dashboard and REST/SSE APIs (spec 04 §5).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.database import async_session_factory
from src.common.schemas import PointLonLat
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


def create_app() -> FastAPI:
    application = FastAPI(title="Rian COP", version="1.0.0")

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
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rian COP</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map { height: 100%; margin: 0; }
    #hud {
      position: absolute; top: 10px; left: 50px; z-index: 1000;
      background: rgba(15,23,42,0.85); color: #e2e8f0; padding: 10px 14px;
      font-family: system-ui, sans-serif; font-size: 13px; border-radius: 8px;
    }
  </style>
</head>
<body>
  <div id="hud">Rian COP — <span id="status">connecting</span></div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([36.745, -119.82], 14);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);
    const layers = { sensors: L.layerGroup().addTo(map), tracks: L.layerGroup().addTo(map),
      reach: L.layerGroup().addTo(map), choke: L.layerGroup().addTo(map),
      routes: L.layerGroup().addTo(map) };

    function renderState(state) {
      layers.sensors.clearLayers();
      layers.tracks.clearLayers();
      layers.reach.clearLayers();
      layers.choke.clearLayers();
      layers.routes.clearLayers();
      (state.sensors || []).forEach(s => {
        const color = s.status === 'active' ? '#22c55e' : '#ef4444';
        L.circleMarker([s.location[1], s.location[0]], { radius: 6, color }).addTo(layers.sensors);
        if (s.fov_wedge) {
          const ring = s.fov_wedge.coordinates[0].map(c => [c[1], c[0]]);
          L.polygon(ring, { color: '#38bdf8', fillOpacity: 0.15, weight: 1 }).addTo(layers.sensors);
        }
      });
      (state.tracks || []).forEach(t => {
        L.circleMarker([t.last_known_point[1], t.last_known_point[0]],
          { radius: 8, color: '#ef4444', fillColor: '#f87171', fillOpacity: 0.8 }).addTo(layers.tracks);
        if (t.history && t.history.coordinates) {
          const latlngs = t.history.coordinates.map(c => [c[1], c[0]]);
          L.polyline(latlngs, { color: '#f59e0b', dashArray: '4 6' }).addTo(layers.tracks);
        }
      });
      (state.reachability || []).forEach(f => {
        if (!f.geometry) return;
        const ring = (f.geometry.coordinates[0] || []).map(c => [c[1], c[0]]);
        if (ring.length) L.polygon(ring, { color: '#eab308', fillOpacity: 0.2 }).addTo(layers.reach);
      });
      (state.chokepoints || []).forEach(c => {
        L.circleMarker([c.location[1], c.location[0]],
          { radius: 7, color: '#3b82f6', fillColor: '#60a5fa', fillOpacity: 0.9 }).addTo(layers.choke);
      });
      (state.directives || []).forEach(d => {
        if (d.response_route && d.response_route.coordinates) {
          const latlngs = d.response_route.coordinates.map(c => [c[1], c[0]]);
          L.polyline(latlngs, { color: '#06b6d4', weight: 4 }).addTo(layers.routes);
        }
      });
      if (state.interceptor && state.interceptor.location) {
        const p = state.interceptor.location;
        L.circleMarker([p[1], p[0]], { radius: 9, color: '#06b6d4' }).addTo(layers.routes);
      }
      document.getElementById('status').textContent = 'live ' + (state.generated_at || '');
    }

    async function loadState() {
      const res = await fetch('/api/v1/cop/state');
      renderState(await res.json());
    }
    loadState();
    const es = new EventSource('/api/v1/cop/stream');
    es.addEventListener('cop_state', e => renderState(JSON.parse(e.data)));
    es.addEventListener('cop_update', e => { loadState(); });
  </script>
</body>
</html>
"""
