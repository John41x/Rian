# Spec 18: Interactive Pasture & Virtual Fence Polygon Editor

## 1. Module Overview & Purpose
Provides an intuitive, point-and-click drawing interface on the satellite map allowing ranchers, farmers, and security operators to create, edit, and monitor custom **Pasture Boundaries**, **Paddocks**, and **Virtual Geofences** without GIS software or manual coordinate entry.

When livestock or unauthorized vehicles cross these user-drawn boundaries:
1. The engine calculates spatial containment (`Point in Polygon`) in real time.
2. Updates live headcounts for each pasture.
3. Automatically triggers a **Geofence Breach Advisory** if an animal strays outside its designated home paddock or approaches a property boundary.
4. In the `industrial_terminal` profile, this exact same module serves as the **Container Bay & Restricted Security Zone Editor**.

---

## 2. Dependencies & Reference Architecture
- **Reference Repos:** `openpasture/openpasture` (virtual fence polygons), `ToastSeagers/Cattleitics` (interactive paddock boundary manager), `Leaflet-Geoman` / Leaflet Polygon Draw primitives.
- **Contracts:** `specs/00-contracts-and-schemas.md`, `specs/13-modular-domain-profiles.md`.
- **Target Paths:** `src/spatial/zones.py`, `src/web/zone_routes.py`, `src/web/app.py`.
- **Test Path:** `tests/test_spatial/test_zone_editor.py`.

---

## 3. Data Contracts & Pydantic Models

```python
class ZoneCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    zone_code: str = Field(..., min_length=2, max_length=32)
    zone_type: str = "pasture"  # "pasture", "paddock", "wash", "container_bay", "restricted_zone"
    color: str = "#22c55e"
    coordinates: List[List[float]]  # Array of [lon, lat] points forming closed polygon
    properties: Dict[str, Any] = Field(default_factory=dict)

class ZoneUpdateRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    coordinates: Optional[List[List[float]]] = None
    properties: Optional[Dict[str, Any]] = None

class GeofenceBreachEvent(BaseModel):
    event_id: UUID
    track_id: UUID
    track_code: str
    entity_class: str
    assigned_zone_name: str
    current_location: List[float]
    distance_to_boundary_m: float
    breach_type: str  # "STRAY_OUT_OF_BOUNDS", "RESTRICTED_ZONE_ENTRY"
    timestamp: datetime
```

---

## 4. API Endpoints

- `GET /api/v1/zones`: Returns all active spatial zones with live headcounts.
- `POST /api/v1/zones`: Creates and persists a new spatial zone in PostgreSQL `spatial_zones` table (with in-memory fallback).
- `PUT /api/v1/zones/{zone_id}`: Updates zone boundary, name, or color.
- `DELETE /api/v1/zones/{zone_id}`: Deletes or archives a spatial zone.
- `GET /api/v1/zones/breaches`: Evaluates active tracks against boundaries and returns any active geofence breaches.

---

## 5. Frontend Interactive Map Drawing UX
- Added to the map toolbar: `[ ✏️ Draw Pasture ]` button.
- Clicking enters **Polygon Draw Mode**:
  - Cursor changes to crosshair.
  - Clicking on the satellite map places vertices with live connecting dashed lines.
  - Double-clicking or clicking the starting vertex closes the polygon.
  - Opens the **"Save Pasture / Zone"** card:
    - Pre-populates calculated acreage / area in hectares.
    - Input: Pasture Name (`e.g. Alfalfa South 40`), Type (`Pasture`, `Nursery Paddock`, `Restricted`), and Color picker.
    - Saves immediately via `POST /api/v1/zones` and renders the persistent tinted polygon with live headcount!

---

## 6. Acceptance Criteria
- [ ] Users can create custom closed polygons via API and persist them in PostGIS.
- [ ] `find_zone_for_point` accurately identifies containment within user-drawn polygons.
- [ ] Deleting or updating a zone refreshes map and recalculates headcounts.
- [ ] Geofence breach detector identifies tracks moving outside designated pastures.
- [ ] 100% test pass rate in `tests/test_spatial/test_zone_editor.py`.
