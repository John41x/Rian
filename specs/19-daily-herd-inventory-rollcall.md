# Spec 19: Automated Daily Herd Inventory & Roll-Call Export

## 1. Module Overview & Purpose
Ranchers and livestock managers spend hours every day driving thousands of acres to manually verify headcounts across dispersed pastures. In industrial container shipping yards, managers conduct daily audits to identify misplaced cargo containers and off-hours vehicle movements.

This module introduces the **Automated Daily Inventory Roll-Call & Audit Subsystem**:
1. **Automated Pasture Census**: Cross-references all active and historical tracks against `spatial_zones` to generate an authoritative herd inventory.
2. **Missing / Unaccounted Alerting**: Automatically flags any animal or tagged asset that has not crossed a camera node within a configurable threshold (e.g., 24 hours), alerting the rancher to possible injury, theft, or boundary breach.
3. **One-Click CSV / Spreadsheet Export**: Formats the roll-call into downloadable, print-friendly CSV and JSON reports compatible with Excel and farm management software.
4. **Domain Profile Adaptability**:
   - In `ranch_livestock`: Generates the **Daily Morning Herd Roll-Call** (Tag, Breed/Class, Current Pasture, Last Camera, Visual Health Status, Missing Hours).
   - In `industrial_terminal`: Generates the **Terminal Cargo & Equipment Audit** (Unit ID, Bay Location, Dwell Time, Movement Status).

---

## 2. Dependencies & Reference Architecture
- **Reference Repos:** `maurice2428/PenziFarm` (inventory audit & CSV export), `ToastSeagers/Cattleitics` (herd roll-call & pasture moves).
- **Contracts:** `specs/00-contracts-and-schemas.md`, `specs/13-modular-domain-profiles.md`, `specs/16-vlm-forensic-snapshot-inspector.md`.
- **Target Paths:** `src/spatial/rollcall_service.py`, `src/web/rollcall_routes.py`, `src/web/app.py`.
- **Test Path:** `tests/test_spatial/test_rollcall.py`.

---

## 3. Data Contracts & Pydantic Models

```python
class RollCallItem(BaseModel):
    track_id: UUID
    track_code: str
    entity_class: str
    current_zone: str              # e.g. "North Pasture" or "Unassigned"
    last_known_point: List[float]
    last_seen_timestamp: datetime
    last_camera_name: str
    hours_since_last_seen: float
    health_status: str             # "VERIFIED_ACTIVE", "OVERDUE_CHECK", "MISSING_CRITICAL"
    vlm_summary: Optional[str] = None

class HerdRollCallSummary(BaseModel):
    generated_at_utc: datetime
    profile_type: str
    total_tracked_count: int
    accounted_count: int
    missing_count: int
    zone_distribution: Dict[str, int]
    items: List[RollCallItem]
```

---

## 4. API Endpoints

- `GET /api/v1/reports/herd-rollcall`: Returns structured `HerdRollCallSummary` JSON.
- `GET /api/v1/reports/herd-rollcall/csv`: Streams RFC 4180 formatted CSV download (`Content-Disposition: attachment; filename=herd_rollcall_YYYYMMDD.csv`).

---

## 5. Acceptance Criteria
- [ ] Correctly aggregates all known tracks, matches them to current pasture zones, and calculates elapsed hours.
- [ ] Flags entities with $>24\text{ hours}$ unobserved as `MISSING_CRITICAL`.
- [ ] CSV endpoint returns clean comma-separated values compatible with spreadsheet software.
- [ ] Top header in COP features `[ 📋 Roll-Call ]` button opening the interactive census table.
- [ ] 100% test pass rate in `tests/test_spatial/test_rollcall.py`.
