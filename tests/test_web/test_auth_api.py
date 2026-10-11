"""HTTP API tests for authentication and admin user routes."""

from __future__ import annotations

from uuid import uuid4

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

from src.common.config import settings
from src.web.app import create_app


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
async def client(db_available):
    from src.common.auth import bootstrap_initial_admin, ensure_users_table
    from src.common.database import async_session_factory

    async with async_session_factory() as session:
        await ensure_users_table(session)
        await bootstrap_initial_admin(session)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def admin_token(client):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": settings.FIRST_SUPERUSER, "password": settings.FIRST_SUPERUSER_PASSWORD},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.mark.asyncio
async def test_login_returns_valid_jwt(client):
    resp = await client.post(
        "/api/v1/auth/login",
        json={
            "email": settings.FIRST_SUPERUSER,
            "password": settings.FIRST_SUPERUSER_PASSWORD,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert "access_token" in body
    assert body["user"]["role"] == "ADMIN"
    assert body["user"]["email"] == settings.FIRST_SUPERUSER


@pytest.mark.asyncio
async def test_me_requires_authentication(client):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_current_user(client, admin_token):
    resp = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == settings.FIRST_SUPERUSER


@pytest.mark.asyncio
async def test_admin_users_forbidden_for_unauthenticated(client):
    resp = await client.get("/api/v1/admin/users")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_users_forbidden_for_operator_role(client, admin_token):
    email = f"operator-{uuid4().hex}@rian.internal"
    create_resp = await client.post(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": email, "password": "op_secret_1", "role": "OPERATOR"},
    )
    assert create_resp.status_code == 201

    login_resp = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "op_secret_1"},
    )
    op_token = login_resp.json()["access_token"]

    list_resp = await client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {op_token}"},
    )
    assert list_resp.status_code == 403

    await client.delete(
        f"/api/v1/admin/users/{create_resp.json()['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )


@pytest.mark.asyncio
async def test_admin_create_and_list_users(client, admin_token):
    email = f"new-user-{uuid4().hex}@rian.internal"
    create_resp = await client.post(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "email": email,
            "password": "new_user_pass",
            "full_name": "Listed User",
            "role": "RESPONDER",
        },
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["email"] == email
    assert created["role"] == "RESPONDER"

    list_resp = await client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_resp.status_code == 200
    emails = {u["email"] for u in list_resp.json()}
    assert email in emails

    delete_resp = await client.delete(
        f"/api/v1/admin/users/{created['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_resp.status_code == 204
