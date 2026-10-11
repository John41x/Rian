"""Authentication and admin user management routes (spec 10)."""

from __future__ import annotations

from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.auth import (
    LoginRequest,
    TokenResponse,
    User,
    UserCreateRequest,
    UserRole,
    authenticate_user,
    create_access_token,
    create_user,
    delete_user,
    get_current_user,
    list_users,
    parse_login_credentials,
    require_role,
)
from src.common.database import get_db_session

router = APIRouter(tags=["auth"])


@router.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(
    login_data: Annotated[LoginRequest, Depends(parse_login_credentials)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenResponse:
    user = await authenticate_user(session, login_data.email, login_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user.email)
    return TokenResponse(access_token=token, user=user)


@router.get("/api/v1/auth/me", response_model=User)
async def read_current_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user


@router.get("/api/v1/admin/users", response_model=List[User])
async def admin_list_users(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    _: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> List[User]:
    return await list_users(session)


@router.post(
    "/api/v1/admin/users",
    response_model=User,
    status_code=status.HTTP_201_CREATED,
)
async def admin_create_user(
    body: UserCreateRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    _: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> User:
    return await create_user(session, body)


@router.delete("/api/v1/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def admin_delete_user(
    user_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    _: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> None:
    removed = await delete_user(session, user_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
