# Spec 12: Real-Time Benchmarking & SLA Monitoring Dashboard

## 1. Module Overview & Purpose
Provides an executive and technical SLA (Service Level Agreement) monitoring and benchmarking dashboard embedded in the Admin portal. Tracks real-time latency percentiles, uptime availability, and delivery success rates across all Rian subsystems.

### The 3 Core SLA Tiers:
1. **Tier 1: Hot-Path Processing Speed (Target: $< 500\text{ms}$):**
   - Ingestion & MIME Decode Latency (Target: $< 50\text{ms}$)
   - YOLOv11 Classification & OSNet Re-ID Inference (Target: $< 60\text{ms}$)
   - Eikonal Isochrone & Dijkstra Intercept Solver (Target: $< 150\text{ms}$)
2. **Tier 2: Sensor Fleet Availability (Target: $\ge 99.0\%$ Uptime):**
   - Active reporting ratio of cameras over rolling 24 hours.
   - Low-battery alerts for solar-powered edge batteries ($< 20\%$).
3. **Tier 3: End-to-End Alert Delivery SLA (Target: $< 5.0\text{ seconds}$):**
   - Elapsed duration from camera PIR motion trigger to Twilio SMS and CoT radio dispatch.

---

## 2. Dependencies & Reference Architecture
- **Reference Repos:** `adminsyspro/sla-monitor` (SLA calculation, uptime metrics, incident registry).
- **Target Paths:** `src/web/sla_service.py`, `tools/benchmark.py`, `tests/test_web/test_sla.py`.

---

## 3. SLA Data Contracts & Services (`src/web/sla_service.py`)

```python
class ComponentLatencyMetric(BaseModel):
    component: str
    target_budget_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    status: str  # "HEALTHY" (p99 < budget), "DEGRADED", "BREACHED"

class SlaDashboardSummary(BaseModel):
    timestamp_utc: str
    tier1_hotpath_sla: Dict[str, ComponentLatencyMetric]
    tier2_fleet_uptime_pct: float
    tier2_low_battery_nodes: List[str]
    tier3_alert_delivery_p95_s: float
    overall_health: str  # "OPTIMAL", "WARNING", "CRITICAL"
```

---

## 4. Automated Benchmark Harness CLI (`tools/benchmark.py`)
Provides a standalone CLI command:
```bash
python -m tools.benchmark --runs 100
```
Measures:
- Ingestion MIME decoding of 180KB JPEG payloads.
- YOLOv11 + OSNet Re-ID inference across 100 sample frames.
- Eikonal wavefront propagation across the $10\text{km} \times 10\text{km}$ synthetic terrain grid.
- Dijkstra shortest path solver across `property_network_edges`.
- Outputs a formatted ASCII and Markdown table comparing measured metrics to official SLA budgets.

---

## 5. Acceptance Criteria
- [ ] `SlaService` accurately records execution latencies and computes rolling p50/p95/p99 percentiles.
- [ ] SLA status transitions to `BREACHED` if p99 exceeds the component's defined budget.
- [ ] `tools/benchmark.py` executes 100 iterations and outputs a valid benchmark report.
- [ ] SLA and benchmark tabs integrated into the web dashboard interface.
- [ ] 100% test pass rate in `tests/test_web/test_sla.py`.
