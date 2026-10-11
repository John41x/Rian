"""Fixtures for kinematic tracking tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import psycopg

from src.common.config import settings
from src.common.schemas import EntityClass


def _db_available() -> bool:
    try:
        with psycopg.connect(settings.sync_database_url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_available() -> bool:
    return _db_available()


@pytest.fixture
async def clean_test_tracks(db_available):
    """Remove test track rows before/after association integration tests."""
    if not db_available:
        pytest.skip("PostgreSQL (rian_db) not available on port 5434")
    prefix = "TRK-TEST-"
    with psycopg.connect(settings.sync_database_url) as conn:
        conn.execute(
            "DELETE FROM entity_tracks WHERE track_code LIKE %s",
            (f"{prefix}%",),
        )
        conn.commit()
    yield prefix
    with psycopg.connect(settings.sync_database_url) as conn:
        conn.execute(
            "DELETE FROM entity_tracks WHERE track_code LIKE %s",
            (f"{prefix}%",),
        )
        conn.commit()


@pytest.fixture
def t0() -> datetime:
    return datetime(2026, 10, 10, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
async def reset_async_engine():
    """Avoid asyncpg connections bound to a stale event loop between tests."""
    yield
    from src.common.database import async_engine

    await async_engine.dispose()


@pytest.fixture
def ranch_corridor_points() -> tuple[tuple[float, float], tuple[float, float]]:
    """Two points ~800 m apart along a plausible walking corridor."""
    start = (-119.8200, 36.7450)
    end = (-119.8110, 36.7452)
    return start, end
