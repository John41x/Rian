"""Database access for sensor resolution and detection event persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.schemas import BoundingBox, DetectionItem, PointLonLat

logger = logging.getLogger(__name__)


@dataclass
class SensorRecord:
    id: UUID
    node_name: str
    location: PointLonLat
    azimuth_deg: float


async def resolve_sensor_by_email(
    session: AsyncSession, sender_email: str
) -> Optional[SensorRecord]:
    stmt = text(
        """
        SELECT id, node_name, azimuth_deg,
               ST_X(location::geometry) AS lon,
               ST_Y(location::geometry) AS lat
        FROM sensor_nodes
        WHERE sender_email = :sender_email AND is_active = TRUE
        LIMIT 1
        """
    )
    result = await session.execute(stmt, {"sender_email": sender_email})
    row = result.mappings().first()
    if row is None:
        logger.warning("Unmapped sensor sender_email=%s", sender_email)
        return None
    return SensorRecord(
        id=row["id"],
        node_name=row["node_name"],
        location=(float(row["lon"]), float(row["lat"])),
        azimuth_deg=float(row["azimuth_deg"]),
    )


async def insert_detection_events(
    session: AsyncSession,
    *,
    sensor_id: UUID,
    event_timestamp: datetime,
    detections: List[DetectionItem],
    raw_image_path: Optional[str],
    processed_latency_ms: int,
) -> List[UUID]:
    """Insert one row per detection; returns inserted event IDs."""
    event_ids: List[UUID] = []
    insert_stmt = text(
        """
        INSERT INTO detection_events (
            id, sensor_id, event_timestamp, object_class, confidence,
            bounding_box, raw_image_path, processed_latency_ms
        )
        VALUES (
            :id, :sensor_id, :event_timestamp, :object_class, :confidence,
            CAST(:bounding_box AS jsonb), :raw_image_path, :processed_latency_ms
        )
        RETURNING id
        """
    )
    for item in detections:
        event_id = uuid4()
        bbox_json = json.dumps(_bbox_to_list(item.bounding_box))
        await session.execute(
            insert_stmt,
            {
                "id": event_id,
                "sensor_id": sensor_id,
                "event_timestamp": event_timestamp,
                "object_class": item.object_class.value,
                "confidence": item.confidence,
                "bounding_box": bbox_json,
                "raw_image_path": raw_image_path,
                "processed_latency_ms": processed_latency_ms,
            },
        )
        event_ids.append(event_id)
    await session.commit()
    return event_ids


def _bbox_to_list(bbox: BoundingBox) -> List[float]:
    return [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]
