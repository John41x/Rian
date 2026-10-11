"""Unit and integration tests for auth helpers (spec 10)."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import jwt
import psycopg
import pytest
from sqlalchemy import text

from src.common.auth import (
    UserCreateRequest,
    UserRole,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.common.config import settings


def test_hash_password_and_verify_success():
    hashed = hash_password("rian_test_secret")
    assert hashed != "rian_test_secret"
    assert verify_password("rian_test_secret", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_create_and_decode_roundtrip():
    token = create_access_token("operator@rian.internal")
    payload = decode_access_token(token)
    assert payload["sub"] == "operator@rian.internal"
    assert payload["type"] == "access"


def test_jwt_rejects_expired_token():
    from fastapi import HTTPException

    token = create_access_token(
        "expired@rian.internal",
        expires_delta=timedelta(seconds=-60),
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401


def test_jwt_rejects_invalid_signature():
    token = create_access_token("user@rian.internal")
    with pytest.raises(jwt.PyJWTError):
        jwt.decode(token, "not-the-real-secret", algorithms=["HS256"])


def _db_available() -> bool:
    try:
        with psycopg.connect(settings.sync_database_url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def db_available():
    if not _db_available():
        pytest.skip("PostgreSQL not available")
    return True


@pytest.fixture
async def auth_session(db_available):
    from src.common.auth import ensure_users_table
    from src.common.database import async_session_factory

    async with async_session_factory() as session:
        await ensure_users_table(session)
        yield session


@pytest.mark.asyncio
async def test_create_and_authenticate_user(auth_session):
    from src.common.auth import authenticate_user, create_user, delete_user

    email = f"auth-test-{uuid4().hex}@rian.internal"
    created = await create_user(
        auth_session,
        UserCreateRequest(
            email=email,
            password="test_pass_123",
            full_name="Test Operator",
            role=UserRole.OPERATOR,
        ),
    )
    assert created.email == email
    assert created.role == UserRole.OPERATOR

    bad = await authenticate_user(auth_session, email, "wrong")
    assert bad is None

    ok = await authenticate_user(auth_session, email, "test_pass_123")
    assert ok is not None
    assert ok.id == created.id

    await delete_user(auth_session, created.id)
