"""
CLI entrypoint for the Rian simulation & validation harness.
"""

from __future__ import annotations

import argparse
import logging
import sys

from tools.mock_camera import MockCameraEmitter
from tools.scenarios import SCENARIOS
from tools.synthetic_gis import SENSOR_NODES, seed_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tools.simulator")


def _camera_email(camera_name: str) -> str:
    for cam in SENSOR_NODES:
        if cam.node_name == camera_name:
            return cam.sender_email
    raise SystemExit(f"Unknown camera: {camera_name}")


def cmd_init_db(_: argparse.Namespace) -> None:
    summary = seed_all()
    logger.info("Database seeded: %s", summary)


def cmd_emit_detection(args: argparse.Namespace) -> None:
    emitter = MockCameraEmitter.from_settings()
    email = _camera_email(args.camera)
    emitter.emit(email, args.entity_class)
    logger.info("Emitted %s detection from %s", args.entity_class, args.camera)


def cmd_run_scenario(args: argparse.Namespace) -> None:
    name = args.name
    if name not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS)}")
    result = SCENARIOS[name](emit_smtp=not args.no_smtp)
    logger.info("Scenario %s passed: %s", name, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rian simulation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Seed synthetic GIS, network, chokepoints, cameras")

    emit_p = sub.add_parser(
        "emit-detection",
        help="Emit a single mock Reolink snapshot email",
    )
    emit_p.add_argument("--camera", required=True, help="sensor node_name (e.g. cam-gate-north)")
    emit_p.add_argument(
        "--entity-class",
        "--target",
        dest="entity_class",
        default="person",
        help="Detected entity class label",
    )

    # Alias documented in spec 05
    emit_legacy = sub.add_parser("emit", help="Alias for emit-detection")
    emit_legacy.add_argument("--camera", required=True)
    emit_legacy.add_argument("--target", dest="entity_class", default="person")

    run_p = sub.add_parser("run-scenario", help="Execute an automated validation scenario")
    run_p.add_argument("--name", required=True, choices=sorted(SCENARIOS))
    run_p.add_argument(
        "--no-smtp",
        action="store_true",
        help="Skip live SMTP emission (assertions only)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init-db":
        cmd_init_db(args)
    elif args.command in ("emit-detection", "emit"):
        cmd_emit_detection(args)
    elif args.command == "run-scenario":
        cmd_run_scenario(args)
    else:
        parser.error(f"Unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
