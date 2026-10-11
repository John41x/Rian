"""
Tactical LLM Copilot & SITREP Generator for the Rian Spatial State Engine.

Translates real-time spatial state (HMM kinematic tracks, Eikonal isochrones,
and Dijkstra intercept geometry) into military-grade situation reports (SITREPs)
and answers natural language operator queries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.common.config import settings
from src.dispatch.guards import (
    apply_output_evidence_guards,
    build_fenced_evidence,
    sanitize_cop_state_for_prompt,
    sanitize_operator_query,
)

logger = logging.getLogger("rian.llm_copilot")


def _normalize_api_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None

SITREP_JSON_SCHEMA_HINT = """
Return a single JSON object with exactly these keys:
timestamp_utc (ISO-8601 string),
defcon_level (integer 1-5),
threat_status (string),
headline (string),
primary_objective (string),
tactical_margin_minutes (number),
target_summary (string),
recommended_actions (array of strings),
engine_provider (string).
Only reference track codes and chokepoint names that appear in the supplied COP state.
"""


class TacticalSitrepReport(BaseModel):
    timestamp_utc: str
    defcon_level: int = Field(..., ge=1, le=5)
    threat_status: str
    headline: str
    primary_objective: str
    tactical_margin_minutes: float
    target_summary: str
    recommended_actions: List[str]
    fenced_evidence: Dict[str, Any]
    engine_provider: str


DEEPSEEK_SPATIAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_active_tracks_tool",
            "description": "Return verified active kinematic tracks with location and velocity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chokepoint_eta_tool",
            "description": "Return chokepoint arrival ETAs and tactical safety margins from directives.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_health_tool",
            "description": "Return sensor node online status, battery, and FOV metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def get_active_tracks_tool(cop_state: Dict[str, Any]) -> Dict[str, Any]:
    tracks = []
    for t in cop_state.get("tracks") or []:
        tracks.append(
            {
                "track_code": t.get("track_code"),
                "object_class": t.get("object_class"),
                "estimated_velocity_ms": t.get("estimated_velocity_ms"),
                "last_known_point": t.get("last_known_point"),
            }
        )
    return {"active_tracks": tracks, "count": len(tracks)}


def get_chokepoint_eta_tool(cop_state: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for d in cop_state.get("directives") or []:
        margin_s = float(d.get("tactical_margin_seconds") or 0.0)
        rows.append(
            {
                "track_code": d.get("track_code"),
                "chokepoint_name": d.get("chokepoint_name"),
                "target_eta": d.get("target_eta"),
                "interceptor_eta": d.get("interceptor_eta"),
                "tactical_margin_minutes": margin_s / 60.0,
            }
        )
    return {"chokepoint_etas": rows}


def get_camera_health_tool(cop_state: Dict[str, Any]) -> Dict[str, Any]:
    nodes = []
    for s in cop_state.get("sensors") or []:
        nodes.append(
            {
                "node_name": s.get("node_name") or s.get("name"),
                "status": s.get("status"),
                "battery_pct": s.get("battery_pct"),
                "fov_azimuth_deg": s.get("fov_azimuth_deg") or s.get("fov_angle"),
            }
        )
    active = sum(1 for n in nodes if n.get("status") == "active")
    return {"sensors": nodes, "active_count": active, "total": len(nodes)}


def execute_spatial_tool(name: str, cop_state: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_active_tracks_tool":
        return get_active_tracks_tool(cop_state)
    if name == "get_chokepoint_eta_tool":
        return get_chokepoint_eta_tool(cop_state)
    if name == "get_camera_health_tool":
        return get_camera_health_tool(cop_state)
    return {"error": f"unknown tool {name}"}


class TacticalCopilot:
    """Provides LLM-augmented tactical reasoning and natural language SITREPs."""

    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        model_name: Optional[str] = None,
        deepseek_base_url: Optional[str] = None,
    ) -> None:
        self.openai_key = _normalize_api_key(
            openai_key if openai_key is not None else settings.OPENAI_API_KEY
        )
        self.anthropic_key = _normalize_api_key(
            anthropic_key if anthropic_key is not None else settings.ANTHROPIC_API_KEY
        )
        self.deepseek_key = _normalize_api_key(
            deepseek_key if deepseek_key is not None else settings.DEEPSEEK_API_KEY
        )
        self.model = model_name or settings.LLM_MODEL
        self.deepseek_base_url = (deepseek_base_url or settings.DEEPSEEK_BASE_URL).rstrip("/")

    @property
    def provider(self) -> str:
        if self.deepseek_key and not str(self.deepseek_key).startswith("mock"):
            return "deepseek"
        if self.openai_key and not str(self.openai_key).startswith("mock"):
            return "openai"
        if self.anthropic_key and not str(self.anthropic_key).startswith("mock"):
            return "anthropic"
        return "local_tactical_reasoner"

    async def generate_sitrep(self, cop_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive tactical situation report from current COP telemetry."""
        safe_state = sanitize_cop_state_for_prompt(cop_state)

        if self.provider == "deepseek":
            try:
                report = await self._call_deepseek_sitrep(safe_state)
                return report.model_dump()
            except Exception as e:
                logger.warning("DeepSeek SITREP failed, falling back to local reasoner: %s", e)

        if self.provider == "openai":
            try:
                legacy = await self._call_openai_sitrep(safe_state)
                report = self._legacy_dict_to_report(legacy, cop_state, "OpenAI")
                return report.model_dump()
            except Exception as e:
                logger.warning("OpenAI API call failed, falling back to local reasoner: %s", e)

        if self.provider == "anthropic":
            try:
                legacy = await self._call_anthropic_sitrep(safe_state)
                report = self._legacy_dict_to_report(legacy, cop_state, "Anthropic")
                return report.model_dump()
            except Exception as e:
                logger.warning("Anthropic API call failed, falling back to local reasoner: %s", e)

        report = self._local_heuristic_sitrep(cop_state)
        return report.model_dump()

    async def answer_operator_query(self, query: str, cop_state: Dict[str, Any]) -> str:
        """Answer natural language queries regarding estate security and tactical directives."""
        safe_query = sanitize_operator_query(query)
        safe_state = sanitize_cop_state_for_prompt(cop_state)

        if self.provider == "deepseek":
            try:
                return await self._call_deepseek_query(safe_query, safe_state)
            except Exception as e:
                logger.warning("DeepSeek query failed, falling back to local reasoner: %s", e)

        if self.provider == "openai":
            try:
                return await self._call_openai_query(safe_query, safe_state)
            except Exception as e:
                logger.warning("OpenAI query failed, falling back to local reasoner: %s", e)

        if self.provider == "anthropic":
            try:
                return await self._call_anthropic_query(safe_query, safe_state)
            except Exception as e:
                logger.warning("Anthropic query failed, falling back to local reasoner: %s", e)

        return self._local_heuristic_query(safe_query, cop_state)

    async def call_deepseek_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.deepseek_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def _call_deepseek_sitrep(self, cop_state: Dict[str, Any]) -> TacticalSitrepReport:
        system = (
            "You are RIAN TACTICAL COMMAND. Authoritative, concise military C2 voice. "
            + SITREP_JSON_SCHEMA_HINT
        )
        user_content = json.dumps(cop_state, default=str)
        data = await self.call_deepseek_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"COP state JSON:\n{user_content}"},
            ],
            temperature=0.2,
        )
        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        parsed.setdefault("engine_provider", f"DeepSeek ({self.model})")
        guarded, _ = apply_output_evidence_guards(parsed, cop_state)
        guarded.setdefault("fenced_evidence", build_fenced_evidence(cop_state))
        return TacticalSitrepReport(**guarded)

    async def _call_deepseek_query(self, query: str, cop_state: Dict[str, Any]) -> str:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are RIAN TACTICAL COMMAND copilot. Answer using verified COP telemetry only. "
                    "Use tool calls when the operator asks about tracks, chokepoints, or cameras."
                ),
            },
            {
                "role": "user",
                "content": f"COP state:\n{json.dumps(cop_state, default=str)}\n\nQuery: {query}",
            },
        ]
        data = await self.call_deepseek_chat(
            messages=messages,
            temperature=0.3,
            tools=DEEPSEEK_SPATIAL_TOOLS,
        )
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            parts: List[str] = []
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                result = execute_spatial_tool(name, cop_state)
                parts.append(f"{name}: {json.dumps(result, default=str)}")
            return " | ".join(parts)
        content = message.get("content")
        if content:
            return str(content).strip()
        return self._local_heuristic_query(query, cop_state)

    def _local_heuristic_sitrep(self, cop_state: Dict[str, Any]) -> TacticalSitrepReport:
        tracks = cop_state.get("tracks", [])
        directives = cop_state.get("directives", [])
        sensors = cop_state.get("sensors", [])
        chokepoints = cop_state.get("chokepoints", [])

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        active_cams = len([s for s in sensors if s.get("status") == "active"])
        fenced = build_fenced_evidence(cop_state)
        provider = "local_tactical_reasoner"

        if not tracks:
            return TacticalSitrepReport(
                timestamp_utc=now_str,
                defcon_level=5,
                threat_status="GREEN // SECURE",
                headline="Perimeter Secure — No Active Incursions Detected",
                primary_objective="Maintain routine perimeter surveillance posture",
                tactical_margin_minutes=0.0,
                target_summary=(
                    f"All {active_cams}/{len(sensors)} sensor nodes operational. "
                    f"Zero kinematic tracks across {len(chokepoints)} chokepoints."
                ),
                recommended_actions=[
                    "Maintain routine sensor heartbeat polling.",
                    "Patrol units remain on standard perimeter watch.",
                ],
                fenced_evidence=fenced,
                engine_provider=provider,
            )

        defcon_level, threat_status = self._defcon_from_state(tracks, directives)
        t = tracks[0]
        speed_kmh = (t.get("estimated_velocity_ms") or 0.0) * 3.6
        obj_class = (t.get("object_class") or "Target").upper()
        t_code = t.get("track_code", "UNKNOWN")
        margin_min = 0.0

        if directives:
            d = directives[0]
            cp_name = d.get("chokepoint_name", "Perimeter Chokepoint")
            margin_min = float(d.get("tactical_margin_seconds") or 0.0) / 60.0
            headline = f"Interception Feasible at {cp_name} (+{margin_min:.1f} min advantage)"
            target_summary = (
                f"Track {t_code} ({obj_class}) at {speed_kmh:.1f} km/h. "
                f"Target ETA {d.get('target_eta', 'N/A')}; interceptor ETA {d.get('interceptor_eta', 'N/A')}."
            )
            primary_objective = f"Intercept at {cp_name}"
            actions = [
                f"Dispatch response unit to {cp_name} immediately.",
                "Acknowledge directive in COP dashboard.",
                "Monitor egress-axis cameras for visual handover.",
            ]
        else:
            headline = f"Target {t_code} Moving In Interior Acreage"
            target_summary = (
                f"Detected {obj_class} at {speed_kmh:.1f} km/h. "
                "Isochrone wavefront active; evaluating chokepoint margins."
            )
            primary_objective = "Reposition intercept assets toward central corridor"
            actions = [
                "Reposition response patrol toward central intersection.",
                "Verify camera snapshots for classification confidence.",
            ]

        report_dict = {
            "timestamp_utc": now_str,
            "defcon_level": defcon_level,
            "threat_status": threat_status,
            "headline": headline,
            "primary_objective": primary_objective,
            "tactical_margin_minutes": margin_min,
            "target_summary": target_summary,
            "recommended_actions": actions,
            "engine_provider": provider,
        }
        guarded, _ = apply_output_evidence_guards(report_dict, cop_state)
        guarded["fenced_evidence"] = fenced
        return TacticalSitrepReport(**guarded)

    @staticmethod
    def _defcon_from_state(
        tracks: List[Dict[str, Any]],
        directives: List[Dict[str, Any]],
    ) -> tuple[int, str]:
        if not tracks:
            return 5, "GREEN // SECURE"
        if directives:
            return 2, "RED // ACTIVE BREACH"
        return 3, "AMBER // UNVERIFIED TARGET"

    def _legacy_dict_to_report(
        self,
        legacy: Dict[str, Any],
        cop_state: Dict[str, Any],
        vendor: str,
    ) -> TacticalSitrepReport:
        tracks = cop_state.get("tracks") or []
        directives = cop_state.get("directives") or []
        defcon, threat = self._defcon_from_state(tracks, directives)
        margin = 0.0
        if directives:
            margin = float(directives[0].get("tactical_margin_seconds") or 0.0) / 60.0
        mapped = {
            "timestamp_utc": legacy.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "defcon_level": defcon,
            "threat_status": legacy.get("threat_level") or threat,
            "headline": legacy.get("headline", "Tactical Update"),
            "primary_objective": legacy.get("summary", "Maintain situational awareness")[:200],
            "tactical_margin_minutes": margin,
            "target_summary": legacy.get("summary", ""),
            "recommended_actions": legacy.get("action_items") or [],
            "engine_provider": legacy.get("provider", vendor),
        }
        guarded, _ = apply_output_evidence_guards(mapped, cop_state)
        guarded["fenced_evidence"] = build_fenced_evidence(cop_state)
        return TacticalSitrepReport(**guarded)

    def _local_heuristic_query(self, query: str, cop_state: Dict[str, Any]) -> str:
        q = query.lower()
        if any(k in q for k in ("track", "target", "threat", "intruder", "active tracks")):
            data = get_active_tracks_tool(cop_state)
            if data["count"] == 0:
                return "There are currently zero active tracks detected in the estate monitoring zone."
            t = data["active_tracks"][0]
            spd = (t.get("estimated_velocity_ms") or 0.0) * 3.6
            return (
                f"Active Track {t.get('track_code')} classified as {t.get('object_class')} "
                f"moving at {spd:.1f} km/h at position {t.get('last_known_point')}."
            )

        if any(k in q for k in ("chokepoint", "eta", "margin", "directive", "intercept")):
            data = get_chokepoint_eta_tool(cop_state)
            rows = data["chokepoint_etas"]
            if not rows:
                return "No actionable intercept directives currently issued. All perimeter gates secure."
            d = rows[0]
            return (
                f"Directive active for {d.get('track_code')} at {d.get('chokepoint_name')} "
                f"with +{d.get('tactical_margin_minutes', 0):.1f} min tactical safety margin."
            )

        if any(k in q for k in ("camera", "sensor", "fleet", "health")):
            data = get_camera_health_tool(cop_state)
            return (
                f"Surveillance fleet: {data['active_count']}/{data['total']} nodes active. "
                f"Telemetry: {json.dumps(data['sensors'][:5], default=str)}"
            )

        tracks = cop_state.get("tracks", [])
        directives = cop_state.get("directives", [])
        sensors = cop_state.get("sensors", [])
        return (
            f"Tactical Copilot standing by. Current state: {len(tracks)} active targets, "
            f"{len(directives)} directives issued, {len(sensors)} sensor nodes online."
        )

    async def _call_openai_sitrep(self, cop_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are RIAN TACTICAL COMMAND. Analyze COP state and return JSON with keys: "
            "timestamp, threat_level, headline, summary, action_items.\n"
            f"COP State: {cop_state}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": prompt}],
                    "temperature": 0.2,
                },
            )
            data = resp.json()
            parsed = json.loads(data["choices"][0]["message"]["content"])
            parsed["provider"] = f"OpenAI ({self.model})"
            return parsed

    async def _call_openai_query(self, query: str, cop_state: Dict[str, Any]) -> str:
        prompt = (
            "You are RIAN TACTICAL COMMAND copilot. Answer concisely from live COP state.\n"
            f"COP State: {cop_state}\nOperator Query: {query}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 250,
                },
            )
            data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()

    async def _call_anthropic_sitrep(self, cop_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are RIAN TACTICAL COMMAND. Output JSON with keys: timestamp, threat_level, "
            f"headline, summary, action_items for COP state: {cop_state}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            text_res = resp.json()["content"][0]["text"]
            parsed = json.loads(text_res)
            parsed["provider"] = "Anthropic (Claude 3.5 Sonnet)"
            return parsed

    async def _call_anthropic_query(self, query: str, cop_state: Dict[str, Any]) -> str:
        prompt = f"COP State: {cop_state}\nOperator Query: {query}\nAnswer as Rian Tactical Command briefly:"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 250,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return str(resp.json()["content"][0]["text"]).strip()
