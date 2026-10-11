# Spec 13: Modular Domain Profiles & Spatial Zones

## 1. Module Overview & Purpose
To enable Rian to operate seamlessly across multiple commercial verticals without code rewrites, this module introduces **Domain Profiles** and **Spatial Zones** (`spatial_zones`).

The system serves as a proving ground for **Ranch & Livestock Operations** (tracking cattle, horses, pasture gates, water troughs, and predators) while maintaining a strict 1-to-1 modular mapping to high-stakes **Industrial & Container Shipping Terminal Security** (tracking cargo bays, turnstiles, wire thieves, and forklifts).

---

## 2. Zero-Overhead Domain Abstraction Pattern

The active profile is controlled by `OPERATION_PROFILE` in `src/common/config.py`:
- `ranch_livestock` (Default: Farmers & Ranchers tracking cattle/pastures)
- `industrial_terminal` (Shipping container terminals & logistics yards)
- `critical_infrastructure` (Utility-scale solar farms & copper theft prevention)

### 2.1 Terminology Mapping Dictionary (`src/common/domains.py`)

| Concept | `ranch_livestock` | `industrial_terminal` | `critical_infrastructure` |
| :--- | :--- | :--- | :--- |
| **System Name** | Rian Ranch Operations | Rian Terminal Defense | Rian Solar Security |
| **Monitored Entity** | Livestock / Stray Cattle / Predator | Intruder / Container Theft / Vehicle | Wire Thief / Intruder / Vehicle |
| **Spatial Zone** | Pasture / Paddock / Grazing Draw | Container Bay / Storage Yard | Inverter Block / Substation Array |
| **Transit / Access Node** | Pasture Gate / Cattle Guard / Trough | Security Gate / Turnstile / Checkpoint | Perimeter Gate / Breach Point |
| **Operational Directive**| Herd Inspection & Roundup | Security Intercept Directive | Intercept & Deterrence Order |
| **Status Nominal** | PASTURE NOMINAL // ALL HERDS IN ZONE | PERIMETER SECURE // NO TAMPERING | ARRAY SECURE // NO WIRE BREACH |
| **Status Warning** | ADVISORY // LIVESTOCK NEAR HIGHWAY | ADVISORY // OFF-HOURS YARD MOTION | ADVISORY // PERIMETER GAP DETECTED |
| **Status Alert** | ALERT // PREDATOR / OPEN GATE | ALERT // ACTIVE CARGO THEFT | ALERT // CRITICAL INFRASTRUCTURE BREACH |

---

## 3. Database Schema: `spatial_zones` Table

```sql
CREATE TABLE IF NOT EXISTS spatial_zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(64) NOT NULL,
    zone_code VARCHAR(32) UNIQUE NOT NULL,
    zone_type VARCHAR(32) NOT NULL,  -- 'pasture', 'paddock', 'container_bay', 'storage_yard'
    boundary GEOMETRY(Polygon, 4326) NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb, -- e.g. {"headcount": 12, "target_capacity": 50, "water_source": true}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_spatial_zones_boundary ON spatial_zones USING GIST(boundary);
```

### 3.1 Demo Ranch Zones (Compact Scale)
Four primary named pasture zones centered on the demo ranch:
1. **North Pasture** (`PASTURE-NORTH`): Lush open grazing, bordered by North County Road.
2. **Creek Draw & Wash** (`ZONE-CREEK`): Low-lying shaded wash where livestock congregate.
3. **South Grazing Meadow** (`PASTURE-SOUTH`): Open meadow containing South Trough and Gate.
4. **Home Ranch Paddock** (`PADDOCK-HOME`): High-security paddock near ranch house.

---

## 4. Implementation Requirements (`src/common/domains.py` & `src/spatial/zones.py`)

1. **`get_domain_profile(profile_name: Optional[str] = None) -> DomainProfile`**:
   - Returns typed terminology mappings, badge colors, icon sets, and prompt personas.
2. **`ZoneManager`**:
   - `get_zones() -> List[Dict[str, Any]]`: Returns all spatial zones with GeoJSON geometry and occupancy counts.
   - `find_zone_for_point(lon: float, lat: float) -> Optional[str]`: Evaluates which pasture/zone contains the coordinates using spatial SQL or Shapely.

---

## 5. Acceptance Criteria
- [ ] `OPERATION_PROFILE=ranch_livestock` translates all UI labels, SITREPs, and metrics to agricultural terminology.
- [ ] `OPERATION_PROFILE=industrial_terminal` immediately switches all terminology to shipping container security with zero code changes.
- [ ] `spatial_zones` accurately detects which pasture an entity resides in based on coordinates.
- [ ] 100% test pass rate in `tests/test_spatial/test_zones.py`.
