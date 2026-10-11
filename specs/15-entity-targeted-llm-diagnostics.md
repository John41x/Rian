# Spec 15: Entity-Targeted LLM Diagnostics & Field Prompts

## 1. Module Overview & Purpose
Provides targeted, single-entity operational insights when an operator or ranch hand clicks on an active entity dot on the map.

Instead of generating a generic whole-property SITREP, clicking on a specific cow, predator, vehicle, or intruder opens an **Entity Diagnostic Card** and triggers **DeepSeek-V3 / DeepSeek-R1** to synthesize a high-value, actionable diagnostic:
- **For Livestock (Ranch Profile)**: Alerts on herd separation, fence boundary proximity, water trough dehydration risks, and unlatched perimeter gates.
- **For Cargo / Security (Industrial Profile)**: Alerts on off-hours container yard prowling, restricted perimeter breaches, and vehicle loitering near cargo bays.

---

## 2. API Endpoint Specification

### `POST /api/v1/cop/entity/{track_id}/insight`
- **Request**: Path parameter `track_id: UUID`.
- **Response (`EntityDiagnosticResponse`)**:
  ```json
  {
    "track_id": "3ad7325a-2fe3-40b3-88bd-8aa084f1d15d",
    "track_code": "HERD-COW-04",
    "entity_class": "cow",
    "risk_level": "ADVISORY // BOUNDARY PROXIMITY",
    "current_zone": "North Pasture",
    "headline": "Cattle Separation & Boundary Drift",
    "insight": "Hereford Cow #4 has drifted 280m south of main herd along Creek Draw. Currently 90m from the County Road fence trending southwest at 1.4 m/s. At current heading, fence line contact in ~5 minutes. Water trough #2 is accessible 150m west.",
    "recommended_actions": [
      "Task pasture ATV to check North Gate latch.",
      "Herd cow back toward central grazing meadow."
    ],
    "telemetry": {
      "speed_kmh": 5.0,
      "heading_deg": 215.0,
      "elapsed_seconds": 45,
      "uncertainty_radius_m": 24.5
    },
    "provider": "DeepSeek-V3 (Ranch Operations Persona)"
  }
  ```

---

## 3. Implementation Requirements (`src/dispatch/llm_copilot.py`)

1. **`generate_entity_insight(track_id: UUID, cop_state: Dict[str, Any]) -> EntityDiagnosticResponse`**:
   - Isolates the specific track's telemetry, historical points, current spatial zone, and proximity to nearest gates, fences, or water troughs.
   - Formulates a focused system prompt tailored to the active `OPERATION_PROFILE`:
     - `ranch_livestock`: Experienced Ranch Manager diagnosing animal behavior and pasture welfare.
     - `industrial_terminal`: Port Facility Security Officer diagnosing unauthorized movement and cargo risk.
   - DeepSeek call with JSON response schema enforcement and local heuristic fallback.
2. **Frontend Integration (`src/web/app.py`)**:
   - Clicking an entity dot centers the map, opens the **Entity Inspector Drawer**, and loads the AI insight with a live loading spinner.

---

## 4. Acceptance Criteria
- [ ] Clicking any entity dot fires `/api/v1/cop/entity/{track_id}/insight`.
- [ ] Returns actionable, high-value guidance specific to that animal or intruder.
- [ ] Prompts dynamically adapt between `ranch_livestock` and `industrial_terminal`.
- [ ] Local reasoner fallback functions seamlessly if DeepSeek API is offline.
- [ ] 100% test pass rate in `tests/test_dispatch/test_entity_insights.py`.
