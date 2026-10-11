"""SLA service, API, and benchmark harness tests (spec 12)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.web.app import create_app
from src.web.sla_service import (
    SlaService,
    compute_percentiles,
    get_sla_service,
    latency_status,
    reset_sla_service,
)
from tools.benchmark import (
    benchmark_ingestion_mime_decode,
    benchmark_intercept_routing,
    benchmark_isochrone_wavefront,
    benchmark_yolo_reid_inference,
    format_results_table,
    run_all_benchmarks_efficient,
    run_micro_benchmark,
)


@pytest.fixture(autouse=True)
def _fresh_sla_service():
    reset_sla_service()
    yield
    reset_sla_service()


def test_compute_percentiles_and_status_healthy_vs_breached():
    healthy_samples = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0]
    _, _, p95_h, p99_h = compute_percentiles(healthy_samples)
    assert latency_status(p95_h, p99_h, 50.0) == "HEALTHY"

    breached_samples = [40.0] * 90 + [80.0] * 10
    _, _, p95_b, p99_b = compute_percentiles(breached_samples)
    assert p99_b > 50.0
    assert latency_status(p95_b, p99_b, 50.0) == "BREACHED"

    degraded_samples = [47.0] * 100
    _, _, p95_d, p99_d = compute_percentiles(degraded_samples)
    assert latency_status(p95_d, p99_d, 50.0) == "DEGRADED"


def test_sla_service_records_and_summarizes_status():
    svc = SlaService()
    for value in [10.0, 12.0, 14.0, 16.0, 18.0]:
        svc.record_latency("ingestion_decode", value)
    summary = svc.get_sla_summary(
        cop_state={
            "sensors": [
                {"node_name": "cam-a", "status": "active", "battery_level": 90},
                {"node_name": "cam-b", "status": "active", "battery_level": 15},
            ]
        }
    )
    metric = summary.tier1_hotpath_sla["ingestion_decode"]
    assert metric.status == "HEALTHY"
    assert metric.samples_count == 5
    assert summary.tier2_fleet_uptime_pct == 100.0
    assert summary.tier2_low_battery_nodes == ["cam-b"]
    assert summary.overall_health == "WARNING"

    for value in [120.0] * 20:
        svc.record_latency("ingestion_decode", value)
    summary_breach = svc.get_sla_summary(cop_state={"sensors": []})
    assert summary_breach.tier1_hotpath_sla["ingestion_decode"].status == "BREACHED"
    assert summary_breach.overall_health == "CRITICAL"


@pytest.mark.asyncio
async def test_get_sla_summary_endpoint():
    app = create_app()
    svc = get_sla_service()
    svc.record_latency("intercept_routing", 5.0)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/sla/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "tier1_hotpath_sla" in body
    assert body["tier1_hotpath_sla"]["intercept_routing"]["samples_count"] == 1
    assert body["overall_health"] in {"OPTIMAL", "WARNING", "CRITICAL"}


@pytest.mark.asyncio
async def test_post_sla_record_endpoint():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        post = await client.post(
            "/api/v1/sla/record",
            json={"component": "yolo_reid_inference", "latency_ms": 22.5},
        )
        assert post.status_code == 200
        summary = await client.get("/api/v1/sla/summary")
    assert summary.status_code == 200
    metric = summary.json()["tier1_hotpath_sla"]["yolo_reid_inference"]
    assert metric["samples_count"] == 1
    assert metric["p50_ms"] == 22.5


def test_benchmark_micro_functions_return_positive_latency():
    assert benchmark_ingestion_mime_decode() > 0.0
    assert benchmark_yolo_reid_inference() > 0.0
    assert benchmark_isochrone_wavefront() > 0.0
    assert benchmark_intercept_routing() > 0.0


def test_run_micro_benchmark_and_table():
    result = run_micro_benchmark("ingestion_mime", runs=3, push_to_sla=True)
    assert result.runs == 3
    assert result.budget_ms == 50.0
    assert result.mean_ms > 0.0
    table = format_results_table([result])
    assert "ingestion_decode" in table
    assert "HEALTHY" in table or "DEGRADED" in table or "BREACHED" in table


def test_run_all_benchmarks_efficient_pushes_sla_samples():
    results = run_all_benchmarks_efficient(2, push_to_sla=True)
    assert len(results) >= 4
    svc = get_sla_service()
    summary = svc.get_sla_summary(cop_state={"sensors": []})
    assert summary.tier1_hotpath_sla["ingestion_decode"].samples_count == 2
    assert summary.tier1_hotpath_sla["hotpath_end_to_end"].samples_count == 2
