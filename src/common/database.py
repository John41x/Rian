from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.common.config import settings

# Async engine for services
async_engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    future=True,
    poolclass=NullPool,
)

async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Synchronous engine for GIS / GeoPandas / migration utilities
sync_engine = create_engine(
    settings.sync_database_url,
    echo=False,
    pool_size=5
)

sync_session_factory = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
