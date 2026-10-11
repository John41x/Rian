from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID

class EntityClass(str, Enum):
    PERSON = "person"
    CAR = "car"
    TRUCK = "truck"
    COW = "cow"
    HORSE = "horse"
    DOG = "dog"

class TrackStatus(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    RESOLVED = "RESOLVED"
    LOST = "LOST"

# Coordinates: (longitude, latitude) in EPSG:4326
PointLonLat = Tuple[float, float]

class BoundingBox(BaseModel):
    ymin: float = Field(..., ge=0.0, le=1.0)
    xmin: float = Field(..., ge=0.0, le=1.0)
    ymax: float = Field(..., ge=0.0, le=1.0)
    xmax: float = Field(..., ge=0.0, le=1.0)

class DetectionEventInbound(BaseModel):
    sensor_sender_email: str
    received_at: datetime
    image_bytes: bytes
    image_filename: Optional[str] = "snapshot.jpg"

class DetectionItem(BaseModel):
    object_class: EntityClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    reid_embedding: Optional[List[float]] = None

class DetectionEventProcessed(BaseModel):
    event_id: UUID
    sensor_id: UUID
    node_name: str
    sensor_location: PointLonLat
    sensor_azimuth_deg: float
    event_timestamp: datetime
    detections: List[DetectionItem]
    raw_image_path: Optional[str] = None
    processed_latency_ms: int

class TrackState(BaseModel):
    track_id: UUID
    track_code: str
    object_class: EntityClass
    status: TrackStatus
    first_detected_at: datetime
    last_detected_at: datetime
    last_known_point: PointLonLat
    estimated_heading_deg: Optional[float] = None
    estimated_velocity_ms: Optional[float] = None
    confidence_score: float = 1.0
    reachability_geojson: Optional[Dict[str, Any]] = None
    last_reid_embedding: Optional[List[float]] = None

class TacticalDirective(BaseModel):
    directive_id: UUID
    track_id: UUID
    track_code: str
    target_class: EntityClass
    target_chokepoint_id: UUID
    chokepoint_name: str
    chokepoint_location: PointLonLat
    target_arrival_time: datetime
    interceptor_arrival_time: datetime
    safety_margin_seconds: float
    action_message: str

# Class-based Maximum Physical Velocity Constraints (m/s)
CLASS_MAX_VELOCITY: Dict[EntityClass, float] = {
    EntityClass.PERSON: 4.0,   # ~14.4 km/h (sprinting hiker)
    EntityClass.CAR: 35.0,     # ~126 km/h
    EntityClass.TRUCK: 25.0,   # ~90 km/h
    EntityClass.COW: 6.0,      # ~21.6 km/h (running cattle)
    EntityClass.HORSE: 12.0,   # ~43.2 km/h (gallop)
    EntityClass.DOG: 10.0,     # ~36.0 km/h
}

# Impassable slope angle in degrees
CLASS_MAX_SLOPE: Dict[EntityClass, float] = {
    EntityClass.PERSON: 45.0,
    EntityClass.CAR: 18.0,
    EntityClass.TRUCK: 15.0,
    EntityClass.COW: 30.0,
    EntityClass.HORSE: 35.0,
    EntityClass.DOG: 40.0,
}
