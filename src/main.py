"""
Rian CLI: Autonomous Spatial State Engine Entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from src.web.app import app

logger = logging.getLogger("rian.main")


def cmd_run(args: argparse.Namespace) -> None:
    """Run full autonomous stack: Ingestion, Tracking, Interception, and COP Dashboard."""
    import uvicorn

    os.environ["RIAN_START_ENGINE"] = "1"
    host = args.host or "0.0.0.0"
    port = args.port or 8000

    logging.info("Starting Rian Autonomous Engine & Web Dashboard on http://%s:%d", host, port)
    uvicorn.run("src.web.app:app", host=host, port=port, log_level="info")


def cmd_cop(args: argparse.Namespace) -> None:
    """Run COP Web Dashboard only (without background ingestion workers)."""
    import uvicorn

    os.environ["RIAN_START_ENGINE"] = "0"
    host = args.host or "0.0.0.0"
    port = args.port or 8000

    logging.info("Starting Rian COP Dashboard on http://%s:%d", host, port)
    uvicorn.run("src.web.app:app", host=host, port=port, log_level="info")


def cmd_worker(args: argparse.Namespace) -> None:
    """Run headless autonomous engine (SMTP listener, CV, Tracking, Interception)."""
    from src.engine import get_engine

    async def _run():
        engine = get_engine()
        await engine.start()

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

        logging.info("Rian Headless Worker running. Press Ctrl+C to terminate.")
        await stop_event.wait()
        await engine.stop()

    asyncio.run(_run())


def cmd_init_db(_: argparse.Namespace) -> None:
    """Initialize and seed database with GIS terrain and network edges."""
    from tools.synthetic_gis import seed_all

    res = seed_all()
    logging.info("Database initialized successfully: %s", res)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rian",
        description="Rian Autonomous Spatial State Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. run (all-in-one)
    run_p = sub.add_parser("run", help="Run full autonomous engine and COP dashboard")
    run_p.add_argument("--host", default="0.0.0.0", help="Web host interface")
    run_p.add_argument("--port", type=int, default=8000, help="Web port (default: 8000)")

    # 2. cop (dashboard only)
    cop_p = sub.add_parser("cop", help="Run COP web dashboard only")
    cop_p.add_argument("--host", default="0.0.0.0", help="Web host interface")
    cop_p.add_argument("--port", type=int, default=8000, help="Web port (default: 8000)")

    # 3. worker (headless daemon)
    sub.add_parser("worker", help="Run headless autonomous processing engine")

    # 4. init-db
    sub.add_parser("init-db", help="Seed PostGIS database with synthetic GIS & network")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "cop":
        cmd_cop(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "init-db":
        cmd_init_db(args)
    else:
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
