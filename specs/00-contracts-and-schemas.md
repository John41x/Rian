# Spec 00: Core Data Contracts & Schemas

## 1. Overview
This document defines the canonical database schemas (PostgreSQL / PostGIS / pgRouting), Pydantic v2 data transfer models, and GeoJSON interchange structures for the Rian platform. **All agents must strictly adhere to these definitions.** Any proposed changes to this document must be reviewed by the Orchestrator.

---

## 2. PostgreSQL + PostGIS DDL Specification

```sql
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- 1. Static Sensor Nodes (Camera & Hardware Metadata)
CREATE TABLE IF NOT EXISTS sensor_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_name VARCHAR(64) NOT NULL,
    hardware_mac VARCHAR(32) UNIQUE,
    sender_email VARCHAR(128) UNIQUE NOT NULL,
    location GEOMETRY(Point, 4326) NOT NULL,
    azimuth_deg DOUBLE PRECISION NOT NULL,       -- Camera boresight direction (0-360)
    fov_angle DOUBLE PRECISION DEFAULT 85.0,     -- Field of view in degrees
    detection_radius_m DOUBLE PRECISION DEFAULT 25.0, -- PIR trigger radius in meters
    battery_level INTEGER,                       -- 0-100%
    solar_charging BOOLEAN DEFAULT FALSE,
    last_heartbeat TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sensor_nodes_location ON sensor_nodes USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_sensor_nodes_sender_email ON sensor_nodes(sender_email);

-- 2. Raw Telemetry Event Log
CREATE TABLE IF NOT EXISTS detection_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sensor_id UUID NOT NULL REFERENCES sensor_nodes(id) ON DELETE CASCADE,
    event_timestamp TIMESTAMPTZ NOT NULL,
    object_class VARCHAR(32) NOT NULL,           -- 'person', 'car', 'truck', 'cow', etc.
    confidence DOUBLE PRECISION NOT NULL,        -- Model confidence score 0.0 - 1.0
    bounding_box JSONB NOT NULL,                 -- [ymin, xmin, ymax, xmax] in normalized or pixel coordinates
    raw_image_path TEXT,                         -- Local path or S3/MinIO object URL
    processed_latency_ms INTEGER,                -- Processing time from arrival to DB
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_detection_events_sensor ON detection_events(sensor_id);
CREATE INDEX IF NOT EXISTS idx_detection_events_time ON detection_events(event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_detection_events_class ON detection_events(object_class);

-- 3. Dynamic Tracking State (Probabilistic Spatial Tracks)
CREATE TYPE track_status_enum AS ENUM ('ACTIVE', 'STALE', 'RESOLVED', 'LOST');

CREATE TABLE IF NOT EXISTS entity_tracks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    track_code VARCHAR(32) UNIQUE NOT NULL,      -- Human readable identifier (e.g. TRK-20261010-001)
    object_class VARCHAR(32) NOT NULL,
    current_status track_status_enum DEFAULT 'ACTIVE',
    first_detected_at TIMESTAMPTZ NOT NULL,
    last_detected_at TIMESTAMPTZ NOT NULL,
    last_sensor_id UUID REFERENCES sensor_nodes(id),
    last_known_point GEOMETRY(Point, 4326) NOT NULL,
    estimated_heading_deg DOUBLE PRECISION,      -- Degrees (0-360)
    estimated_velocity_ms DOUBLE PRECISION,      -- Meters per second
    history_linestring GEOMETRY(LineString, 4326),
    reachability_polygon GEOMETRY(Polygon, 4326),-- Latest dynamic isochrone
    confidence_score DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_entity_tracks_spatial ON entity_tracks USING GIST(last_known_point);
CREATE INDEX IF NOT EXISTS idx_entity_tracks_polygon ON entity_tracks USING GIST(reachability_polygon);
CREATE INDEX IF NOT EXISTS idx_entity_tracks_status ON entity_tracks(current_status);

-- 4. Property Topological Network (Roads, Trails, Washes)
CREATE TABLE IF NOT EXISTS property_network_edges (
    id SERIAL PRIMARY KEY,
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,
    cost DOUBLE PRECISION NOT NULL,              -- Traversal time in seconds (forward)
    reverse_cost DOUBLE PRECISION NOT NULL,      -- Traversal time in seconds (reverse, -1 if one-way)
    surface_type VARCHAR(32) DEFAULT 'gravel',   -- 'paved', 'gravel', 'two_track', 'wash'
    max_speed_kmh DOUBLE PRECISION DEFAULT 40.0,
    geom GEOMETRY(LineString, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_network_edges_geom ON property_network_edges USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_network_edges_source ON property_network_edges(source);
CREATE INDEX IF NOT EXISTS idx_network_edges_target ON property_network_edges(target);

-- 5. Interception Chokepoint Nodes (Gates, Crossings, Grids)
CREATE TABLE IF NOT EXISTS strategic_chokepoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(64) NOT NULL,
    chokepoint_type VARCHAR(32) DEFAULT 'gate',  -- 'gate', 'cattle_guard', 'intersection', 'culvert'
    location GEOMETRY(Point, 4326) NOT NULL,
    network_node_id INTEGER,                     -- References topology vertex in property_network_edges
    is_accessible_to_response BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chokepoints_location ON strategic_chokepoints USING GIST(location);

-- 6. Interception Action Directives
CREATE TABLE IF NOT EXISTS tactical_directives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    track_id UUID NOT NULL REFERENCES entity_tracks(id) ON DELETE CASCADE,
    chokepoint_id UUID NOT NULL REFERENCES strategic_chokepoints(id),
    target_eta TIMESTAMPTZ NOT NULL,
    interceptor_eta TIMESTAMPTZ NOT NULL,
    tactical_margin_seconds DOUBLE PRECISION NOT NULL,
    notification_dispatched BOOLEAN DEFAULT FALSE,
    dispatch_payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tactical_directives_track ON tactical_directives(track_id);
```

---

## 3. Class Velocity Bounds & Movement Physics

For kinematic gating envelopes:
$$d_{ij} \le v_{max}(class) \times \Delta t$$

| Entity Class | $v_{max}$ (m/s) | $v_{max}$ (km/h) | Default Heading Weight $\kappa$ | Impassable Slope Threshold |
| :--- | :--- | :--- | :--- | :--- |
| `person` | 4.0 | 14.4 | 1.5 | $45^\circ$ (Tobler Hiking decay) |
| `car` | 35.0 | 126.0 | 4.0 | $18^\circ$ (Infinite friction off road) |
| `truck` | 25.0 | 90.0 | 4.0 | $15^\circ$ (Infinite friction off road) |
| `cow` | 6.0 | 21.6 | 1.0 | $30^\circ$ (Avoids steep rocky grades) |
| `horse` | 12.0 | 43.2 | 1.2 | $35^\circ$ |
| `dog` | 10.0 | 36.0 | 1.0 | $40^\circ$ |

---

## 4. Pydantic v2 Contract Models (`src/common/schemas.py`)

```python
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
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

# Coordinates: [longitude, latitude] in EPSG:4326
PointLonLat = Tuple[float, float]

class BoundingBox(BaseModel):
    ymin: float
    xmin: float
    ymax: float
    xmax: float

class DetectionEventInbound(BaseModel):
    sensor_sender_email: str
    received_at: datetime
    image_bytes: bytes
    image_filename: Optional[str] = "snapshot.jpg"

class DetectionItem(BaseModel):
    object_class: EntityClass
    confidence: float
    bounding_box: BoundingBox

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
```

---

## 5. GeoJSON Format Specification

All spatial exports (APIs and frontend feeds) must adhere to standard GeoJSON RFC 7946.

### Reachability Isochrone Polygon
```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [-119.8234, 36.7451],
        [-119.8211, 36.7460],
        [-119.8180, 36.7445],
        [-119.8195, 36.7420],
        [-119.8234, 36.7451]
      ]
    ]
  },
  "properties": {
    "track_id": "8f8b3c92-38d1-4e78-bc5a-e7cf7b3400a1",
    "track_code": "TRK-20261010-001",
    "target_class": "person",
    "elapsed_seconds": 360,
    "confidence": 0.88,
    "generated_at": "2026-10-10T21:30:00Z"
  }
}
```
