"""
Modular domain profiles: terminology and default spatial zones per vertical (spec 13).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.common.config import settings

DomainProfileType = Literal["ranch_livestock", "industrial_terminal", "critical_infrastructure"]

# Demo ranch reference point inside North Pasture (used by tests and containment checks).
NORTH_PASTURE_TEST_LON = -119.82
NORTH_PASTURE_TEST_LAT = 36.754


def _ranch_polygon(
    west: float, south: float, east: float, north: float
) -> Dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def _ranch_livestock_zones() -> List[Dict[str, Any]]:
    return [
        {
            "name": "North Pasture",
            "zone_code": "PASTURE-NORTH",
            "zone_type": "pasture",
            "color": "#22c55e",
            "geometry": _ranch_polygon(-119.835, 36.750, -119.805, 36.758),
        },
        {
            "name": "Creek Draw & Wash",
            "zone_code": "ZONE-CREEK",
            "zone_type": "wash",
            "color": "#38bdf8",
            "geometry": _ranch_polygon(-119.835, 36.742, -119.805, 36.750),
        },
        {
            "name": "South Grazing Meadow",
            "zone_code": "PASTURE-SOUTH",
            "zone_type": "pasture",
            "color": "#eab308",
            "geometry": _ranch_polygon(-119.835, 36.735, -119.805, 36.742),
        },
        {
            "name": "Home Ranch Paddock",
            "zone_code": "PADDOCK-HOME",
            "zone_type": "paddock",
            "color": "#8b5cf6",
            "geometry": _ranch_polygon(-119.825, 36.743, -119.815, 36.747),
        },
    ]


def _industrial_terminal_zones() -> List[Dict[str, Any]]:
    return [
        {
            "name": "North Storage Yard",
            "zone_code": "YARD-NORTH",
            "zone_type": "storage_yard",
            "color": "#64748b",
            "geometry": _ranch_polygon(-119.835, 36.750, -119.805, 36.758),
        },
        {
            "name": "Container Bay A",
            "zone_code": "BAY-A",
            "zone_type": "container_bay",
            "color": "#0ea5e9",
            "geometry": _ranch_polygon(-119.835, 36.742, -119.805, 36.750),
        },
        {
            "name": "Container Bay B",
            "zone_code": "BAY-B",
            "zone_type": "container_bay",
            "color": "#6366f1",
            "geometry": _ranch_polygon(-119.835, 36.735, -119.805, 36.742),
        },
        {
            "name": "Gate Control Paddock",
            "zone_code": "GATE-HOME",
            "zone_type": "checkpoint",
            "color": "#f43f5e",
            "geometry": _ranch_polygon(-119.825, 36.743, -119.815, 36.747),
        },
    ]


def _critical_infrastructure_zones() -> List[Dict[str, Any]]:
    return [
        {
            "name": "North Inverter Block",
            "zone_code": "ARRAY-NORTH",
            "zone_type": "inverter_block",
            "color": "#84cc16",
            "geometry": _ranch_polygon(-119.835, 36.750, -119.805, 36.758),
        },
        {
            "name": "Substation Array",
            "zone_code": "SUBSTATION-MAIN",
            "zone_type": "substation",
            "color": "#a855f7",
            "geometry": _ranch_polygon(-119.835, 36.742, -119.805, 36.750),
        },
        {
            "name": "South Panel Row",
            "zone_code": "ARRAY-SOUTH",
            "zone_type": "inverter_block",
            "color": "#14b8a6",
            "geometry": _ranch_polygon(-119.835, 36.735, -119.805, 36.742),
        },
        {
            "name": "Perimeter Operations Hub",
            "zone_code": "HUB-PERIM",
            "zone_type": "perimeter",
            "color": "#f59e0b",
            "geometry": _ranch_polygon(-119.825, 36.743, -119.815, 36.747),
        },
    ]


_PROFILES: Dict[str, Dict[str, Any]] = {
    "ranch_livestock": {
        "profile_type": "ranch_livestock",
        "app_title": "Rian Ranch Operations",
        "entity_label": "Livestock / Wildlife",
        "zone_label": "Pastures & Paddocks",
        "node_label": "Gates & Water Troughs",
        "task_label": "Herd Roundups & Checks",
        "status_nominal": "NOMINAL // ALL HERDS IN ZONE",
        "status_warning": "ADVISORY // LIVESTOCK NEAR HIGHWAY",
        "status_alert": "ALERT // PREDATOR / OPEN GATE",
        "persona_system_prompt": (
            "You are the Rian Ranch tactical copilot. Prioritize herd safety, open-gate risk, "
            "predator advisories, and pasture containment. Use agricultural terminology and "
            "concise SITREP formatting."
        ),
        "default_zones": _ranch_livestock_zones(),
    },
    "industrial_terminal": {
        "profile_type": "industrial_terminal",
        "app_title": "Rian Terminal Security",
        "entity_label": "Perimeter Intruder",
        "zone_label": "Storage Yards & Bays",
        "node_label": "Security Checkpoints",
        "task_label": "Intercept Directives",
        "status_nominal": "SECURE // NO TAMPERING",
        "status_warning": "ADVISORY // OFF-HOURS YARD MOTION",
        "status_alert": "ALERT // ACTIVE CARGO THEFT",
        "persona_system_prompt": (
            "You are the Rian container terminal defense copilot. Focus on perimeter breaches, "
            "cargo theft, yard motion after hours, and intercept routing to security checkpoints."
        ),
        "default_zones": _industrial_terminal_zones(),
    },
    "critical_infrastructure": {
        "profile_type": "critical_infrastructure",
        "app_title": "Rian Solar Security",
        "entity_label": "Wire Thief / Intruder / Vehicle",
        "zone_label": "Inverter Blocks & Substation Arrays",
        "node_label": "Perimeter Gates & Breach Points",
        "task_label": "Intercept & Deterrence Orders",
        "status_nominal": "ARRAY SECURE // NO WIRE BREACH",
        "status_warning": "ADVISORY // PERIMETER GAP DETECTED",
        "status_alert": "ALERT // CRITICAL INFRASTRUCTURE BREACH",
        "persona_system_prompt": (
            "You are the Rian solar-farm security copilot. Emphasize copper-wire theft, "
            "inverter tampering, perimeter gaps, and rapid deterrence orders."
        ),
        "default_zones": _critical_infrastructure_zones(),
    },
}


class DomainProfile(BaseModel):
    profile_type: str
    app_title: str
    entity_label: str
    zone_label: str
    node_label: str
    task_label: str
    status_nominal: str
    status_warning: str
    status_alert: str
    persona_system_prompt: str
    default_zones: List[Dict[str, Any]] = Field(default_factory=list)


def get_domain_profile(profile_name: Optional[str] = None) -> DomainProfile:
    """Resolve the active operation profile from settings or an explicit override."""
    key = (profile_name or settings.OPERATION_PROFILE or "ranch_livestock").strip()
    if key not in _PROFILES:
        key = "ranch_livestock"
    return DomainProfile(**_PROFILES[key])
