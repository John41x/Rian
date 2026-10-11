"""
Micro-benchmark harness for Rian hot-path SLA budgets (spec 12).

Usage:
    python -m tools.benchmark --runs 100
    python -m tools.benchmark --runs 10 --save-markdown
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.ingestion.inference import InferenceConfig, YOLOInferenceProcessor
from src.ingestion.mime_utils import extract_jpeg_attachments, parse_email_message
from src.spatial.routing import _default_mock_graph, solve_shortest_path_networkx
from src.tracking.isochrone import build_local_friction_window, solve_travel_time_grid
from src.tracking.constants import EntityClass
from src.web.sla_service import (
    DEFAULT_COMPONENT_BUDGETS_MS,
    compute_percentiles,
    get_sla_service,
    reset_sla_service,
)
from PIL import Image

from tools.mock_camera import generate_synthetic_jpeg

_CACHED_180KB_EMAIL: bytes | None = None

@dataclass
class BenchmarkResult:
    name: str
    sla_component: str
    budget_ms: float
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @property
    def status(self) -> str:
        from src.web.sla_service import latency_status

        return latency_status(self.p95_ms, self.p99_ms, self.budget_ms)


def _jpeg_payload_min_size(min_bytes: int) -> bytes:
    """JPEG with high-frequency noise so compressed size reaches the target."""
    width, height = 640, 480
    rng = np.random.default_rng(42)
    while True:
        noise = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        img = Image.fromarray(noise, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        payload = buf.getvalue()
        if len(payload) >= min_bytes:
            return payload
        width = int(width * 1.25)
        height = int(height * 1.25)


def _build_180kb_jpeg_email() -> bytes:
    """Build a MIME message with a ~180KB JPEG attachment (cached)."""
    global _CACHED_180KB_EMAIL
    if _CACHED_180KB_EMAIL is not None:
        return _CACHED_180KB_EMAIL
    payload = _jpeg_payload_min_size(180 * 1024)
    msg = EmailMessage()
    msg["Subject"] = "Motion detection alert"
    msg["From"] = "cam-gate-north@ranch.domain"
    msg["To"] = "ingest@rian.internal"
    msg.set_content("Motion event")
    msg.add_attachment(payload, maintype="image", subtype="jpeg", filename="snapshot.jpg")
    _CACHED_180KB_EMAIL = msg.as_bytes()
    return _CACHED_180KB_EMAIL


def benchmark_ingestion_mime_decode() -> float:
    raw = _build_180kb_jpeg_email()
    start = time.perf_counter()
    message = parse_email_message(raw)
    attachments = extract_jpeg_attachments(message)
    if not attachments:
        raise RuntimeError("benchmark email missing JPEG attachment")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms


def benchmark_yolo_reid_inference() -> float:
    processor = YOLOInferenceProcessor(
        config=InferenceConfig(force_mock=True),
    )
    image_bytes = generate_synthetic_jpeg(width=640, height=480)
    start = time.perf_counter()
    processor.predict(image_bytes)
    return (time.perf_counter() - start) * 1000.0


def benchmark_isochrone_wavefront() -> float:
    side = 41
    friction = build_local_friction_window(side, EntityClass.PERSON, slope_deg=0.0)
    start = time.perf_counter()
    solve_travel_time_grid(friction, cell_size_m=10.0)
    return (time.perf_counter() - start) * 1000.0


def benchmark_intercept_routing() -> float:
    graph = _default_mock_graph()
    start = time.perf_counter()
    solve_shortest_path_networkx(graph, 1, 3)
    return (time.perf_counter() - start) * 1000.0


BENCHMARK_FUNCS: Dict[str, Tuple[str, Callable[[], float]]] = {
    "ingestion_mime": ("ingestion_decode", benchmark_ingestion_mime_decode),
    "yolo_reid": ("yolo_reid_inference", benchmark_yolo_reid_inference),
    "isochrone_wavefront": ("isochrone_wavefront", benchmark_isochrone_wavefront),
    "intercept_routing": ("intercept_routing", benchmark_intercept_routing),
}


def run_micro_benchmark(
    bench_key: str,
    runs: int,
    *,
    push_to_sla: bool = True,
) -> BenchmarkResult:
    if bench_key not in BENCHMARK_FUNCS:
        raise KeyError(f"Unknown benchmark: {bench_key}")
    sla_component, func = BENCHMARK_FUNCS[bench_key]
    budget = DEFAULT_COMPONENT_BUDGETS_MS[sla_component]
    samples: List[float] = []
    for _ in range(runs):
        samples.append(func())
    mean_ms, p50, p95, p99 = compute_percentiles(samples)
    if push_to_sla:
        svc = get_sla_service()
        for sample in samples:
            svc.record_latency(sla_component, sample)
    return BenchmarkResult(
        name=bench_key,
        sla_component=sla_component,
        budget_ms=budget,
        runs=runs,
        mean_ms=mean_ms,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
    )


def run_all_benchmarks_efficient(runs: int, *, push_to_sla: bool = True) -> List[BenchmarkResult]:
    """Run each benchmark once per iteration and aggregate stats."""
    results: List[BenchmarkResult] = []
    svc = get_sla_service() if push_to_sla else None
    collected: Dict[str, List[float]] = {k: [] for k in BENCHMARK_FUNCS}
    hotpath: List[float] = []

    for _ in range(runs):
        iteration: Dict[str, float] = {}
        for key, (sla_component, func) in BENCHMARK_FUNCS.items():
            elapsed = func()
            collected[key].append(elapsed)
            iteration[sla_component] = elapsed
            if svc is not None:
                svc.record_latency(sla_component, elapsed)
        e2e = sum(iteration.values())
        hotpath.append(e2e)
        if svc is not None:
            svc.record_latency("hotpath_end_to_end", e2e)

    for key, (sla_component, _) in BENCHMARK_FUNCS.items():
        samples = collected[key]
        mean_ms, p50, p95, p99 = compute_percentiles(samples)
        budget = DEFAULT_COMPONENT_BUDGETS_MS[sla_component]
        results.append(
            BenchmarkResult(
                name=key,
                sla_component=sla_component,
                budget_ms=budget,
                runs=runs,
                mean_ms=mean_ms,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
            )
        )
    if svc is not None and hotpath:
        mean_ms, p50, p95, p99 = compute_percentiles(hotpath)
        results.append(
            BenchmarkResult(
                name="hotpath_end_to_end",
                sla_component="hotpath_end_to_end",
                budget_ms=DEFAULT_COMPONENT_BUDGETS_MS["hotpath_end_to_end"],
                runs=runs,
                mean_ms=mean_ms,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
            )
        )
    return results


def format_results_table(results: Sequence[BenchmarkResult]) -> str:
    header = (
        f"{'Component':<24} {'Budget':>8} {'Mean':>10} {'P50':>10} "
        f"{'P95':>10} {'P99':>10} {'Status':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.sla_component:<24} {r.budget_ms:>7.1f}ms {r.mean_ms:>9.2f}ms "
            f"{r.p50_ms:>9.2f}ms {r.p95_ms:>9.2f}ms {r.p99_ms:>9.2f}ms {r.status:>10}"
        )
    return "\n".join(lines)


def format_results_markdown(results: Sequence[BenchmarkResult]) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# Rian SLA Benchmark Report",
        f"",
        f"Generated: `{ts}` UTC",
        f"",
        "| Component | Budget (ms) | Mean | P50 | P95 | P99 | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.sla_component} | {r.budget_ms:.1f} | {r.mean_ms:.2f} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} | {r.status} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rian hot-path SLA micro-benchmarks")
    parser.add_argument("--runs", type=int, default=100, help="Iterations per benchmark")
    parser.add_argument(
        "--save-markdown",
        action="store_true",
        help="Write benchmark report to benchmarks/latest.md",
    )
    parser.add_argument(
        "--no-sla-push",
        action="store_true",
        help="Do not publish samples to in-process SlaService",
    )
    args = parser.parse_args(argv)

    reset_sla_service()
    results = run_all_benchmarks_efficient(
        max(1, args.runs),
        push_to_sla=not args.no_sla_push,
    )
    table = format_results_table(results)
    print(table)
    if args.save_markdown:
        out_dir = Path("benchmarks")
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "latest.md"
        md_path.write_text(format_results_markdown(results), encoding="utf-8")
        print(f"\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
