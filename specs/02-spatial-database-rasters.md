# Spec 02: Spatial Database & Raster Surfaces

## 1. Module Overview
The **Spatial Database & Rasters Module** manages the spatial data fabric of the Rian platform. It provides high-performance elevation, slope, and land-cover rasters, alongside topological road/trail networks powered by PostgreSQL 16, PostGIS 3.4, and pgRouting.

Because entities navigate along paths of least mechanical and energetic resistance, continuous kinematic tracking across unmonitored blind zones relies on terrain-derived cost surfaces. This module is responsible for storing, transforming, and querying these surfaces in real time.

---

## 2. Dependencies & Prerequisites
- **Contracts:** `specs/00-contracts-and-schemas.md`
- **Libraries:**
  - `psycopg` / `asyncpg` / `SQLAlchemy`
  - `GDAL` (v3.8+)
  - `Rasterio` (Raster manipulation and windowed reading)
  - `Shapely` (Vector geometric operations)
  - `GeoPandas` (Spatial data ingestion and transformation)
  - `pgrouting` (Graph network routing)

---

## 3. Data Architecture & Tables

### 3.1 Digital Elevation Models (DEM)
- **Source:** USGS 3D Elevation Program (3DEP) 10-meter (1/3 arc-second) or 1-meter LiDAR GeoTIFFs.
- **Storage Strategy:** Dual-tier:
  - Raw GeoTIFF files archived in `data/rasters/dem/` with Cloud-Optimized GeoTIFF (COG) tiling.
  - Tiled raster tables in PostGIS for spatial SQL analysis:
```sql
CREATE TABLE IF NOT EXISTS elevation_rasters (
    rid SERIAL PRIMARY KEY,
    rast RASTER,
    filename VARCHAR(128)
);
CREATE INDEX IF NOT EXISTS idx_elevation_rasters_rast ON elevation_rasters USING GIST(ST_ConvexHull(rast));
```

### 3.2 Derived Slope & Aspect Rasters
- **Slope Angle ($\theta$):** Calculated in degrees from the DEM using GDAL or PostGIS `ST_Slope(rast, 1, '32BF', 'DEGREES')`.
- Slope informs mechanical impedance:
  - Foot travel: parameterized via Tobler's Hiking function ($e^{-3.5 \cdot |\tan \theta + 0.05|}$).
  - Wheeled vehicles: slopes exceeding $15^\circ$ (truck) or $18^\circ$ (car) receive infinite impedance ($\infty$).

### 3.3 Land-Cover Classification & Friction Weights
Land-cover data is derived from the National Land Cover Database (NLCD) or ranch boundary shapefiles:

| Surface Classification | PostGIS Code / Tag | Base Friction Multiplier ($C_{land}$) | Passable by Vehicle? | Passable by Livestock? | Passable by Foot? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Paved / Graded Road | `road_graded` | 1.0 (baseline) | Yes ($60\text{ km/h}$) | Yes ($6\text{ km/h}$) | Yes ($5\text{ km/h}$) |
| Two-Track Ranch Trail | `two_track` | 1.8 | Yes ($25\text{ km/h}$) | Yes ($6\text{ km/h}$) | Yes ($4.5\text{ km/h}$) |
| Open Grassland / Pasture | `pasture` | 2.5 | No (unless 4WD tractor) | Yes | Yes |
| Dense Scrub & Chaparral | `scrub` | 4.5 | No | Slow ($1\text{ km/h}$) | Slow ($2\text{ km/h}$) |
| Heavy Timber & Boulders | `timber_rock` | 9.0 | No | Impassable | Very Slow ($1\text{ km/h}$) |
| Deep Water / High Fence | `barrier` | $\infty$ (No-go) | No | No | No |

### 3.4 Road & Trail Network Graph (pgRouting)
The topological network represents all traversable corridors for both intruders and security interceptors:
- Ingested from OpenStreetMap (OSM) via `osm2pgrouting` or custom property vector files.
- Topology verification:
```sql
SELECT pgr_createTopology('property_network_edges', 0.0001, 'geom', 'id', 'source', 'target');
SELECT pgr_analyzeGraph('property_network_edges', 0.0001, 'geom', 'id', 'source', 'target');
```
- Cost calculation:
  - Forward cost: $\text{cost} = \text{ST\_Length}(\text{ST\_Transform}(\text{geom}, 3857)) / (max\_speed\_kmh \times 0.27778)$ (seconds)
  - Reverse cost: equal to forward cost for bi-directional roads; $-1.0$ for one-way edges.

---

## 4. Spatial Service APIs (`src/spatial/service.py`)

### 4.1 Windowed Friction Surface Extraction
```python
def get_friction_surface_window(
    center_lon: float,
    center_lat: float,
    radius_meters: float,
    entity_class: EntityClass,
    resolution_m: float = 10.0
) -> Tuple[np.ndarray, Affine, int]:
    """
    Extracts a 2D numpy array representing traversal friction in seconds/meter
    for the specified entity class around the coordinates.
    """
```

### 4.2 Route Solvers (`src/spatial/routing.py`)
```python
async def find_shortest_route(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    travel_mode: str = "vehicle"
) -> RouteResult:
    """
    Snaps coordinates to nearest network vertices and executes pgr_dijkstra.
    Returns travel time in seconds, distance in meters, and GeoJSON LineString.
    """
```

---

## 5. Seeders & Property Provisioning
To facilitate immediate multi-agent development and tests without waiting for gigabytes of federal GIS downloads:
- `tools/seed_synthetic_gis.py` generates:
  - A synthetic $10\text{ km} \times 10\text{ km}$ DEM raster with hills, valleys, and creek washes.
  - A synthetic ranch road graph with gates, intersections, and chokepoints.
  - Populates `sensor_nodes`, `property_network_edges`, and `strategic_chokepoints`.

---

## 6. Acceptance Criteria
- [ ] PostGIS and pgRouting extensions enabled in target database.
- [ ] Elevation and slope extraction returns correct physical units ($m$ and degrees).
- [ ] Friction surface generation enforces infinite impedance for vehicles off-road.
- [ ] `pgr_dijkstra` routes along road networks and correctly honors edge costs.
- [ ] Synthetic GIS seeder script runs in under 10 seconds.
