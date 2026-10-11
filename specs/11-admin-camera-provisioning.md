# Spec 11: Admin Portal & Interactive Camera Provisioning Wizard

## 1. Module Overview & Purpose
Provides an intuitive web administration interface for property managers to commission, calibrate, test, and manage cellular camera nodes (`sensor_nodes`) and user accounts without writing SQL scripts or touching backend configuration.

### Core Capabilities:
1. **Interactive Map Placement Wizard:** Click anywhere on the satellite basemap to place a new camera pin with precise WGS84 GPS coordinates (`lon`, `lat`).
2. **Compass Azimuth & FOV Dial:** Visual orientation control allowing operators to aim the camera's boresight bearing ($0^\circ$ to $360^\circ$) down approach corridors and set the optical Field of View (default $85^\circ$, radius 25m).
3. **Telemetry Connectivity Probe:** A "Test Connection" button that transmits a synthetic SMTP trigger to port 2525 and verifies that the node receives, processes, and logs the camera in under 3 seconds.
4. **User & Node Management:** Full CRUD management for sensor nodes and user access control.

---

## 2. Dependencies & Reference Architecture
- **Reference Repos:** `sahilyadav-01/secureview-camera-monitoring` (SOC camera setup), `OneByJorah/SentryView`, `FastAPI`.
- **Target Paths:** `src/web/camera_routes.py`, `src/web/admin_routes.py`, `src/web/app.py`.
- **Test Path:** `tests/test_web/test_camera_provisioning.py`.

---

## 3. API Endpoints

- `GET /api/v1/cameras`: Returns all registered sensor nodes with live status, battery levels, and FOV coordinates.
- `POST /api/v1/cameras`: Registers a new camera node:
  ```json
  {
    "node_name": "cam-ridge-west",
    "sender_email": "cam-ridge-west@ranch.domain",
    "lon": -119.825,
    "lat": 36.751,
    "azimuth_deg": 135.0,
    "fov_angle": 85.0,
    "detection_radius_m": 25.0
  }
  ```
- `PATCH /api/v1/cameras/{camera_id}`: Updates coordinates, azimuth, or active status.
- `DELETE /api/v1/cameras/{camera_id}`: Decommissions a sensor node.
- `POST /api/v1/cameras/{camera_id}/test-probe`: Simulates an incoming SMTP snapshot for this camera to verify end-to-end detection and alert delivery.

---

## 4. Frontend Admin & Provisioning Interface
Integrated into the Web Dashboard (`src/web/app.py` & `public/index.html`):
- Modal/Tab: `[ 🛡 Admin & Fleet ]` accessible by clicking the Admin icon in the top header.
- Sub-tabs:
  1. **Sensor Fleet & Add Camera Wizard**: Form with live coordinate crosshair and compass dial.
  2. **User Accounts**: List users, create operator accounts, reset credentials.
  3. **Live System Health**: Immediate connectivity checks for PostgreSQL, SMTP daemon, YOLO engine, and DeepSeek copilot.

---

## 5. Acceptance Criteria
- [ ] New cameras can be created and persisted in `sensor_nodes` table via API.
- [ ] Created cameras immediately render with correct green/red icon and directional cyan FOV wedge on the map.
- [ ] Test probe endpoint triggers an instantaneous detection event that appears in COP.
- [ ] 100% test pass rate in `tests/test_web/test_camera_provisioning.py`.
