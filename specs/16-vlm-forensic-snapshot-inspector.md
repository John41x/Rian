# Spec 16: Multimodal VLM Forensic Snapshot Inspector & Media Gallery

## 1. Module Overview & Purpose
While fast edge object detection (YOLOv11) classifies entities into categories (`cow`, `truck`, `person`, `dog`) in under 25ms, human operators and AI supervisors need to inspect the **actual visual evidence**. 

This module introduces an asynchronous **Multimodal Vision-Language (VLM) Forensic Inspector** and **Camera Snapshot Gallery**:
1. **Visual Evidence Serving**: Persists and serves raw edge snapshot JPEGs with normalized bounding box metadata.
2. **Deep Forensic Vision Analysis**: Dispatches candidate frames to a multimodal vision model (DeepSeek-VLM / OpenAI / local computer vision analyzer) to generate rich, descriptive visual forensic reports (e.g. animal markings/coat color, signs of injury or limping, vehicle make/cargo/tools in bed, open gate latches, or nearby predator shadows).
3. **Interactive Media Inspector in COP**: Operators can click any camera node or tracked entity to view the high-resolution photo with an interactive HTML5 `<canvas>` bounding box overlay and an instant "🔍 AI Vision Analysis" breakdown.

---

## 2. Dependencies & Reference Architecture
- **Reference Repos:** `ALOOHA-hub/Violence-Action-Detection` (asynchronous decoupled forensic VLM reasoning), `sahilyadav-01/secureview-camera-monitoring` (media playback & snapshot inspection).
- **Contracts:** `specs/00-contracts-and-schemas.md`, `specs/13-modular-domain-profiles.md`, `specs/15-entity-targeted-llm-diagnostics.md`.
- **Target Paths:** `src/ingestion/vlm_inspector.py`, `src/web/snapshot_routes.py`, `src/web/app.py`.
- **Test Path:** `tests/test_ingestion/test_vlm_inspector.py`.

---

## 3. Data Contracts & Pydantic Models

```python
class BoundingBoxOverlay(BaseModel):
    label: str
    confidence: float
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    color: str = "#22c55e"

class SnapshotRecord(BaseModel):
    snapshot_id: UUID
    sensor_id: UUID
    camera_name: str
    timestamp: datetime
    image_url: str
    entity_class: str
    confidence: float
    bounding_boxes: List[BoundingBoxOverlay]
    vlm_analyzed: bool = False

class VlmInspectionReport(BaseModel):
    snapshot_id: UUID
    entity_class: str
    visual_description: str        # e.g. "Mature black Angus steer with white blaze on forehead"
    environmental_context: str     # e.g. "Midday sunlight, dry pasture grass, fence line visible 30m background"
    risk_assessment: str          # "LOW // HEALTHY GRAZING", "MODERATE // STRAGGLE", "HIGH // THREAT"
    findings: List[str]            # Key visual observations
    recommended_action: str        # Actionable field instruction
    provider: str                  # "DeepSeek-VLM", "OpenAI-Vision", or "Rian Local Forensic Analyzer"
```

---

## 4. API Endpoints

- `GET /api/v1/cameras/{camera_id}/snapshots`: Returns the latest 10 snapshots for the camera with detection boxes.
- `GET /api/v1/snapshots/{snapshot_id}/image`: Streams raw JPEG image bytes (`image/jpeg`).
- `POST /api/v1/snapshots/{snapshot_id}/vlm-analyze`: Executes deep VLM forensic inspection on the snapshot image and returns `VlmInspectionReport`.
- `GET /api/v1/cop/entity/{track_id}/snapshot`: Returns the latest visual snapshot associated with the active track.

---

## 5. Frontend UI Integration (`_COP_HTML`)
- In `#entity-modal` and camera cards:
  - Responsive image viewer with an HTML5 `<canvas>` drawing colored bounding boxes around detected entities.
  - "🔍 Deep AI Vision Analysis" button triggering the VLM analysis with real-time loading feedback.
  - Collapsible metadata badge showing camera name, shutter time, and detection confidence.

---

## 6. Acceptance Criteria
- [ ] Snapshot images can be stored and retrieved over HTTP with proper `image/jpeg` headers.
- [ ] Normalized bounding boxes accurately scale and draw over images regardless of viewport aspect ratio.
- [ ] `VlmForensicInspector` produces structured forensic reports analyzing animal condition, vehicle cargo, and environment.
- [ ] Local forensic analyzer functions seamlessly if external VLM API keys are omitted.
- [ ] 100% test pass rate in `tests/test_ingestion/test_vlm_inspector.py`.
