"""Real-time SLA monitoring and latency percentile aggregation (spec 12)."""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_COMPONENT_BUDGETS_MS: Dict[str, float] = {
    "ingestion_decode": 50.0,
    "yolo_reid_inference": 60.0,
    "isochrone_wavefront": 100.0,
    "intercept_routing": 50.0,
    "hotpath_end_to_end": 500.0,
}

TIER3_ALERT_BUDGET_P95_S = 5.0
TIER2_UPTIME_TARGET_PCT = 99.0
LOW_BATTERY_THRESHOLD_PCT = 20
MAX_LATENCY_SAMPLES = 1000
MAX_ALERT_SAMPLES = 1000


class ComponentLatencyMetric(BaseModel):
    component: str
    target_budget_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    status: str  # HEALTHY, DEGRADED, BREACHED
    samples_count: int


class SlaDashboardSummary(BaseModel):
    timestamp_utc: str
    tier1_hotpath_sla: Dict[str, ComponentLatencyMetric]
    tier2_fleet_uptime_pct: float
    tier2_low_battery_nodes: List[str]
    tier3_alert_delivery_p95_s: float
    overall_health: str  # OPTIMAL, WARNING, CRITICAL


class RecordLatencyRequest(BaseModel):
    component: str
    latency_ms: float = Field(..., ge=0)


class RecordAlertDeliveryRequest(BaseModel):
    duration_s: float = Field(..., ge=0)


def compute_percentiles(
    samples: Sequence[float],
) -> tuple[float, float, float, float]:
    """Return (mean, p50, p95, p99) in milliseconds; zeros when empty."""
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(samples, dtype=np.float64)
    mean = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50, method="linear"))
    p95 = float(np.percentile(arr, 95, method="linear"))
    p99 = float(np.percentile(arr, 99, method="linear"))
    return mean, p50, p95, p99


def latency_status(p95_ms: float, p99_ms: float, budget_ms: float) -> str:
    if p99_ms > budget_ms:
        return "BREACHED"
    if p95_ms > budget_ms * 0.9:
        return "DEGRADED"
    return "HEALTHY"


class SlaService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: Dict[str, Deque[float]] = {
            name: deque(maxlen=MAX_LATENCY_SAMPLES) for name in DEFAULT_COMPONENT_BUDGETS_MS
        }
        self._alert_delivery_s: Deque[float] = deque(maxlen=MAX_ALERT_SAMPLES)

    def record_latency(self, component: str, latency_ms: float) -> None:
        with self._lock:
            if component not in self._latencies:
                self._latencies[component] = deque(maxlen=MAX_LATENCY_SAMPLES)
            self._latencies[component].append(float(latency_ms))

    def record_alert_delivery(self, duration_s: float) -> None:
        with self._lock:
            self._alert_delivery_s.append(float(duration_s))

    def _snapshot_latencies(self) -> Dict[str, List[float]]:
        with self._lock:
            return {k: list(v) for k, v in self._latencies.items()}

    def _snapshot_alerts(self) -> List[float]:
        with self._lock:
            return list(self._alert_delivery_s)

    def get_sla_summary(
        self, cop_state: Optional[Dict[str, Any]] = None
    ) -> SlaDashboardSummary:
        latency_snap = self._snapshot_latencies()
        tier1: Dict[str, ComponentLatencyMetric] = {}
        breached = False
        degraded = False

        for component, budget in DEFAULT_COMPONENT_BUDGETS_MS.items():
            samples = latency_snap.get(component, [])
            _, p50, p95, p99 = compute_percentiles(samples)
            status = latency_status(p95, p99, budget)
            if status == "BREACHED":
                breached = True
            elif status == "DEGRADED":
                degraded = True
            tier1[component] = ComponentLatencyMetric(
                component=component,
                target_budget_ms=budget,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                status=status,
                samples_count=len(samples),
            )

        uptime_pct, low_battery = self._tier2_fleet_metrics(cop_state)
        alert_samples = self._snapshot_alerts()
        if alert_samples:
            alert_p95 = float(
                np.percentile(np.asarray(alert_samples, dtype=np.float64), 95)
            )
        else:
            alert_p95 = 0.0

        overall = self._overall_health(
            breached=breached,
            degraded=degraded,
            uptime_pct=uptime_pct,
            low_battery_nodes=low_battery,
            alert_p95_s=alert_p95,
        )

        return SlaDashboardSummary(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            tier1_hotpath_sla=tier1,
            tier2_fleet_uptime_pct=uptime_pct,
            tier2_low_battery_nodes=low_battery,
            tier3_alert_delivery_p95_s=alert_p95,
            overall_health=overall,
        )

    def _tier2_fleet_metrics(
        self, cop_state: Optional[Dict[str, Any]]
    ) -> tuple[float, List[str]]:
        sensors: List[Dict[str, Any]] = []
        if cop_state and "sensors" in cop_state:
            sensors = list(cop_state.get("sensors") or [])
        elif cop_state is None:
            sensors = _fetch_sensors_sync()

        if not sensors:
            return 100.0, []

        active = sum(1 for s in sensors if s.get("status") == "active")
        uptime_pct = 100.0 * active / len(sensors)
        low_battery: List[str] = []
        for s in sensors:
            level = s.get("battery_level")
            if level is not None and float(level) < LOW_BATTERY_THRESHOLD_PCT:
                name = s.get("node_name") or str(s.get("id", "unknown"))
                low_battery.append(str(name))
        return uptime_pct, low_battery

    @staticmethod
    def _overall_health(
        *,
        breached: bool,
        degraded: bool,
        uptime_pct: float,
        low_battery_nodes: List[str],
        alert_p95_s: float,
    ) -> str:
        if breached or uptime_pct < TIER2_UPTIME_TARGET_PCT or alert_p95_s > TIER3_ALERT_BUDGET_P95_S:
            return "CRITICAL"
        if degraded or low_battery_nodes:
            return "WARNING"
        return "OPTIMAL"


def _fetch_sensors_sync() -> List[Dict[str, Any]]:
    try:
        from src.common.config import settings
        import psycopg

        sql = """
            SELECT node_name, is_active, battery_level
            FROM sensor_nodes
            ORDER BY node_name
        """
        with psycopg.connect(settings.sync_database_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        sensors: List[Dict[str, Any]] = []
        for node_name, is_active, battery_level in rows:
            sensors.append(
                {
                    "node_name": node_name,
                    "status": "active" if is_active else "offline",
                    "battery_level": battery_level,
                }
            )
        return sensors
    except Exception as exc:
        logger.debug("SLA fleet DB lookup skipped: %s", exc)
        return []


_sla_service: Optional[SlaService] = None
_sla_singleton_lock = threading.Lock()


def get_sla_service() -> SlaService:
    global _sla_service
    with _sla_singleton_lock:
        if _sla_service is None:
            _sla_service = SlaService()
        return _sla_service


def reset_sla_service() -> None:
    global _sla_service
    with _sla_singleton_lock:
        _sla_service = None


sla_router = APIRouter(prefix="/api/v1/sla", tags=["sla"])


@sla_router.get("/summary", response_model=SlaDashboardSummary)
async def get_sla_summary_endpoint() -> SlaDashboardSummary:
    from src.web.app import build_cop_state

    state = await build_cop_state()
    cop_dict = state.model_dump(mode="json")
    return get_sla_service().get_sla_summary(cop_state=cop_dict)


@sla_router.post("/record")
async def record_latency_endpoint(body: RecordLatencyRequest) -> Dict[str, str]:
    get_sla_service().record_latency(body.component, body.latency_ms)
    return {"status": "recorded", "component": body.component}


@sla_router.post("/record-alert")
async def record_alert_endpoint(body: RecordAlertDeliveryRequest) -> Dict[str, str]:
    get_sla_service().record_alert_delivery(body.duration_s)
    return {"status": "recorded"}


@sla_router.post("/run-benchmark", response_model=SlaDashboardSummary)
async def run_benchmark_endpoint(runs: int = 25) -> SlaDashboardSummary:
    import asyncio
    from tools.benchmark import run_all_benchmarks_efficient

    safe_runs = max(5, min(100, runs))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: run_all_benchmarks_efficient(safe_runs, push_to_sla=True))

    from src.web.app import build_cop_state
    state = await build_cop_state()
    return get_sla_service().get_sla_summary(cop_state=state.model_dump(mode="json"))
