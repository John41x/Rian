"""JWT authentication, password hashing, and user persistence (spec 10)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, List, Optional
from uuid import UUID

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.config import settings
from src.common.database import get_db_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    RESPONDER = "RESPONDER"


class User(BaseModel):
    id: UUID
    email: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


class LoginRequest(BaseModel):
    email: str
    password: str


class UserCreateRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.OPERATOR


_USERS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(128),
    role VARCHAR(32) NOT NULL DEFAULT 'OPERATOR',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
)
"""

_USERS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def create_access_token(
    subject: str,
    *,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(hours=settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS)
    )
    payload: dict[str, Any] = {"sub": subject, "exp": expire, "type": "access"}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256"],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _row_to_user(row: Any) -> User:
    return User(
        id=row.id,
        email=row.email,
        full_name=row.full_name,
        role=UserRole(row.role),
        is_active=row.is_active,
        created_at=row.created_at,
    )


async def ensure_users_table(session: AsyncSession) -> None:
    await session.execute(text(_USERS_TABLE_DDL))
    await session.execute(text(_USERS_INDEX_DDL))
    await session.commit()


async def bootstrap_initial_admin(session: AsyncSession) -> None:
    result = await session.execute(text("SELECT COUNT(*) AS cnt FROM users"))
    count = int(result.scalar_one())
    if count > 0:
        return
    hashed = hash_password(settings.FIRST_SUPERUSER_PASSWORD)
    await session.execute(
        text(
            """
            INSERT INTO users (email, hashed_password, full_name, role, is_active)
            VALUES (:email, :hashed_password, :full_name, :role, TRUE)
            """
        ),
        {
            "email": settings.FIRST_SUPERUSER,
            "hashed_password": hashed,
            "full_name": "Rian Administrator",
            "role": UserRole.ADMIN.value,
        },
    )
    await session.commit()


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    result = await session.execute(
        text(
            """
            SELECT id, email, full_name, role, is_active, created_at
            FROM users
            WHERE email = :email
            """
        ),
        {"email": email.lower()},
    )
    row = result.mappings().first()
    if row is None:
        return None
    return _row_to_user(row)


async def get_user_by_id(session: AsyncSession, user_id: UUID) -> Optional[User]:
    result = await session.execute(
        text(
            """
            SELECT id, email, full_name, role, is_active, created_at
            FROM users
            WHERE id = :user_id
            """
        ),
        {"user_id": user_id},
    )
    row = result.mappings().first()
    if row is None:
        return None
    return _row_to_user(row)


async def authenticate_user(
    session: AsyncSession, email: str, password: str
) -> Optional[User]:
    result = await session.execute(
        text(
            """
            SELECT id, email, full_name, role, is_active, created_at, hashed_password
            FROM users
            WHERE email = :email
            """
        ),
        {"email": email.lower()},
    )
    row = result.mappings().first()
    if row is None:
        return None
    if not row.is_active:
        return None
    if not verify_password(password, row.hashed_password):
        return None
    await session.execute(
        text("UPDATE users SET last_login = NOW() WHERE id = :id"),
        {"id": row.id},
    )
    await session.commit()
    return _row_to_user(row)


async def create_user(session: AsyncSession, user_create: UserCreateRequest) -> User:
    existing = await get_user_by_email(session, user_create.email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )
    hashed = hash_password(user_create.password)
    result = await session.execute(
        text(
            """
            INSERT INTO users (email, hashed_password, full_name, role, is_active)
            VALUES (:email, :hashed_password, :full_name, :role, TRUE)
            RETURNING id, email, full_name, role, is_active, created_at
            """
        ),
        {
            "email": user_create.email.lower(),
            "hashed_password": hashed,
            "full_name": user_create.full_name,
            "role": user_create.role.value,
        },
    )
    row = result.mappings().one()
    await session.commit()
    return _row_to_user(row)


async def list_users(session: AsyncSession) -> List[User]:
    result = await session.execute(
        text(
            """
            SELECT id, email, full_name, role, is_active, created_at
            FROM users
            ORDER BY created_at ASC
            """
        )
    )
    return [_row_to_user(row) for row in result.mappings().all()]


async def delete_user(session: AsyncSession, user_id: UUID) -> bool:
    result = await session.execute(
        text("DELETE FROM users WHERE id = :user_id RETURNING id"),
        {"user_id": user_id},
    )
    deleted = result.scalar_one_or_none()
    await session.commit()
    return deleted is not None


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    payload = decode_access_token(token)
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await get_user_by_email(session, str(subject))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive or unknown user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(*roles: UserRole):
    allowed = set(roles)

    async def _require_role(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return _require_role


async def parse_login_credentials(request: Request) -> LoginRequest:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        return LoginRequest.model_validate(body)
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="username and password are required",
        )
    return LoginRequest(email=str(username), password=str(password))
