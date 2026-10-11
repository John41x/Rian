# Spec 17: Predator Threat Detection & Automated Wildlife Deterrents

## 1. Module Overview & Purpose
Coyotes, wolves, mountain lions, and feral dog packs pose a severe economic threat to cow-calf ranching operations, frequently killing newborn calves and sheep.

This module introduces a real-time **Predator Threat Detection & Deterrent Subsystem**:
1. **Predator Classification & Gating**: Distinguishes domestic working ranch dogs from wild predators (coyotes, feral dog packs, wolves, bears) based on detection confidence, pack aggregation, proximity to perimeter fences, and absence of human handlers.
2. **Herd-to-Predator Spatial Proximity Engine**: Computes continuous real-time distances between every active livestock track (`cow`, `calf`, `horse`, `sheep`) and detected predators.
3. **Escalation Levels**:
   - `PROXIMITY_ADVISORY` (Predator within 1,000m of herd)
   - `HERD_THREAT_WARNING` (Predator within 500m of herd or in nursery paddock)
   - `CRITICAL_INCURSION` (Predator within 200m closing with positive velocity)
4. **Automated Edge Deterrent Triggers**:
   - Sends commands to edge cameras to activate high-decibel acoustic sirens and flashing LED spotlights (`trigger_camera_deterrent`).
   - Dispatches emergency high-priority Twilio SMS / WhatsApp notifications to the rancher.
5. **COP Visual Threat Vectors**:
   - Renders a glowing crimson vector arrow on the satellite map from the predator toward the nearest vulnerable livestock group, displaying live distance in meters ($\text{e.g. } \text{"Coyote } \to \text{ Herd-04 (280m)"}$).

---

## 2. Dependencies & Reference Architecture
- **Contracts:** `specs/00-contracts-and-schemas.md`, `specs/13-modular-domain-profiles.md`, `specs/15-entity-targeted-llm-diagnostics.md`, `specs/16-vlm-forensic-snapshot-inspector.md`.
- **Target Paths:** `src/tracking/predator_service.py`, `src/web/predator_routes.py`, `src/web/app.py`.
- **Test Path:** `tests/test_tracking/test_predator_service.py`.

---

## 3. Data Contracts & Pydantic Models

```python
class PredatorThreatLevel(str, Enum):
    NOMINAL = "NOMINAL"
    PROXIMITY_ADVISORY = "PROXIMITY_ADVISORY"
    HERD_THREAT_WARNING = "HERD_THREAT_WARNING"
    CRITICAL_INCURSION = "CRITICAL_INCURSION"

class PredatorThreatEvent(BaseModel):
    threat_id: UUID
    predator_track_id: UUID
    predator_track_code: str
    predator_class: str             # "coyote", "dog", "wolf", "bear"
    nearest_livestock_id: UUID
    nearest_livestock_code: str
    distance_meters: float
    threat_level: PredatorThreatLevel
    headline: str
    deterrent_activated: bool = False
    deterrent_type: Optional[str] = None  # "acoustic_siren", "strobe_light", "manual"
    timestamp: datetime

class PredatorProximitySummary(BaseModel):
    active_threats_count: int
    highest_threat_level: PredatorThreatLevel
    threats: List[PredatorThreatEvent]
    nursery_zone_breaches: int
```

---

## 4. API Endpoints

- `GET /api/v1/predators/threats`: Returns all active predator threats with distance to nearest herds and threat levels.
- `POST /api/v1/predators/{threat_id}/trigger-deterrent`: Sends acoustic/strobe trigger command to the nearest camera node.
- `POST /api/v1/predators/simulate-incursion`: Simulation endpoint that seeds a coyote/wolf track approaching a nursery calf herd.

---

## 5. Acceptance Criteria
- [ ] Correctly computes Euclidean and terrain distance between predator and livestock tracks.
- [ ] Escalates threat levels: $>1000\text{m}$ = Nominal, $500\text{m}–1000\text{m}$ = Advisory, $200\text{m}–500\text{m}$ = Warning, $<200\text{m}$ = Critical.
- [ ] Deterrent activation records timestamp and broadcasts SSE notification.
- [ ] Visual threat vector renders in COP between predator dot and target herd.
- [ ] 100% test pass rate in `tests/test_tracking/test_predator_service.py`.
