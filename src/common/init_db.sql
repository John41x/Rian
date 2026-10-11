-- Enable Required PostGIS and Routing Extensions
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
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'track_status_enum') THEN
        CREATE TYPE track_status_enum AS ENUM ('ACTIVE', 'STALE', 'RESOLVED', 'LOST');
    END IF;
END$$;

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
    last_reid_embedding DOUBLE PRECISION[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'entity_tracks' AND column_name = 'last_reid_embedding'
    ) THEN
        ALTER TABLE entity_tracks ADD COLUMN last_reid_embedding DOUBLE PRECISION[];
    END IF;
END$$;
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

-- 7. Raster Surfaces (Elevation DEM)
CREATE TABLE IF NOT EXISTS elevation_rasters (
    rid SERIAL PRIMARY KEY,
    rast RASTER,
    filename VARCHAR(128)
);
CREATE INDEX IF NOT EXISTS idx_elevation_rasters_rast ON elevation_rasters USING GIST(ST_ConvexHull(rast));

-- 8. Spatial operation zones (pastures, yards, arrays — spec 13)
CREATE TABLE IF NOT EXISTS spatial_zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(64) NOT NULL,
    zone_code VARCHAR(32) UNIQUE NOT NULL,
    zone_type VARCHAR(32) NOT NULL,
    boundary GEOMETRY(Polygon, 4326) NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_spatial_zones_boundary ON spatial_zones USING GIST(boundary);
