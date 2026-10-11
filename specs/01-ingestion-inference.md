# Spec 01: Ingestion Gateway & Computer Vision Inference

## 1. Module Overview
The **Ingestion & Inference Module** provides zero-lock-in telemetry ingestion for off-grid cellular IoT camera nodes (specifically Reolink Go Plus and NDAA-compliant successors) and processes snapshot frames through high-efficiency YOLOv11 object detection models.

Because standalone cellular battery cameras operate behind carrier CGNAT without public inbound ports or FTPS client binaries, data egress is achieved via outbound SMTP with MIME snapshot attachments. This module hosts both a native non-blocking `aiosmtpd` daemon and an HTTP webhook parser for managed email services (SendGrid/AWS SES).

---

## 2. Dependencies & Prerequisites
- **Contracts:** `specs/00-contracts-and-schemas.md`
- **Libraries:**
  - Python 3.12+
  - `aiosmtpd` (Async SMTP server daemon)
  - `onnxruntime` or `ultralytics` (YOLOv11 ONNX / PyTorch execution)
  - `Pillow` / `numpy` (Image decoding and manipulation)
  - `SQLAlchemy` (asyncpg) / `asyncpg` (PostgreSQL client)
  - `fastapi` & `uvicorn` (Webhook fallback endpoint)

---

## 3. Pipeline Architecture

```
[Camera Client: Reolink Go Plus]
              │ Outbound SMTP (TLS 465/587)
              ▼
    aiosmtpd CameraSMTPHandler (:2525)
              │ (Non-blocking ACK '250 Message accepted')
              ▼
       MIME Message Parser
              │
              ├── [Invalid/No Image] ──> Discard & Log Warning
              │
              └── [Valid JPEG Bytes]
                      │
                      ▼
            Async Ingestion Queue (asyncio.Queue)
                      │
                      ▼
             Worker: Inference Pipeline
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    Save Image File       ONNX YOLOv11 Model
    (/data/snapshots/)    (conf >= 0.45, Target Classes)
                                 │
                                 ▼
                     Lookup Sensor Metadata
                     (sensor_nodes table via sender_email)
                                 │
                                 ▼
                    Persist `detection_events`
                                 │
                                 ▼
                     Emit DetectionEventProcessed
                     (Forward to Spec 03 Kinematic Engine)
```

---

## 4. Ingestion Daemon Specification

### 4.1 Native SMTP Daemon (`src/ingestion/smtp_server.py`)
- **Port:** Configurable via `SMTP_PORT` (default: `2525`, mapped via Docker/firewall from `25` or `587`).
- **Handler Class:** `CameraSMTPHandler`
- **Protocol Contract:**
  1. Camera connects via SMTP and issues `EHLO`, `MAIL FROM`, `RCPT TO`, `DATA`.
  2. Handler receives envelope, parses MIME multipart payload using `email.message_from_bytes(..., policy=policy.default)`.
  3. Extracts JPEG attachment bytes (`image/jpeg` or disposition `attachment`).
  4. Returns `250 Message accepted for delivery` to camera client immediately. Camera must not be kept waiting to prevent LTE connection timeouts.
  5. Puts extracted payload onto internal `asyncio.Queue`.

### 4.2 Webhook Ingestion Fallback (`src/ingestion/webhook_server.py`)
- **Endpoint:** `POST /api/v1/telemetry/smtp-webhook`
- Accepts multipart form data or JSON payloads formatted by SendGrid Inbound Parse / AWS SES.
- Normalizes parsed email into `DetectionEventInbound` and enqueues to the same worker.

---

## 5. Computer Vision & Inference Pipeline (`src/ingestion/inference.py`)

### 5.1 Model Specifications
- **Model Architecture:** YOLOv11n (Nano) or YOLOv11s.
- **Runtime:** ONNX Runtime (`yolo11n.onnx`) with CUDA or Apple MPS acceleration (falls back to optimized CPU).
- **Target Filter Classes:**
  - `person` (Intruders, hikers, trespassers)
  - `car`, `truck` (Vehicles, ATVs, unauthorized trucks)
  - `cow`, `horse`, `dog` (Livestock and ranch working animals)
- **Confidence Floor:** Configurable, default `CONFIDENCE_THRESHOLD = 0.45`.
- **Bounding Box Normalization:** Returns coordinates normalized to `[ymin, xmin, ymax, xmax]` within $[0.0, 1.0]$.

### 5.2 Resolution Handling & Cooldown Coalescing
- Cameras stream snapshots at sub-stream resolution (Fluent: $640\times360$ or $896\times512$).
- Downscaled images directly match YOLOv11 native tensor size ($640\times640$) without manual interpolation overhead.
- Rapid burst events from the same sender within `< 3.0` seconds are tagged as a cohesive detection burst for optical flow / velocity initialization.

---

## 6. Sensor Identity & Storage

### 6.1 Database Resolution
When an email arrives from `camera-gate-north@reolink.domain`:
```sql
SELECT id, node_name, location, azimuth_deg, fov_angle, is_active
FROM sensor_nodes
WHERE sender_email = :sender_email AND is_active = TRUE;
```
If the sender email is unrecognized, an unmapped sensor record is logged, and the event is tagged with `sensor_id = NULL` for manual operator claim.

### 6.2 Raw Image Archival
- Snapshot storage path: `STORAGE_ROOT/snapshots/{YYYY}/{MM}/{DD}/{sensor_id}_{timestamp}_{uuid}.jpg`.
- Local disk in development/simulation; S3/MinIO compatible object store in production.

---

## 7. Error Handling & Edge Cases
1. **Corrupted / Truncated MIME:** If camera cuts LTE data transfer midway, discard gracefully, log error, and do not crash the daemon.
2. **Non-Detection Events:** If YOLO yields 0 bounding boxes with confidence $\ge 0.45$ (e.g., wind-blown brush, heat mirage), log a false-trigger event with `object_class = 'unclassified'` or discard according to `RECORD_EMPTY_TRIGGERS` config.
3. **High Burst Storms:** The `asyncio.Queue` has a max size of 500 items. If backpressured, warn and prioritize newer frames.

---

## 8. Unit & Integration Testing Plan
1. **Mock SMTP Transmission:** Use Python `smtplib` to send standard Reolink snapshot emails to `localhost:2525`.
2. **Inference Verification:** Verify that test images with known targets (`person.jpg`, `truck.jpg`, `cow.jpg`) correctly classify and output bounding boxes $\ge 0.45$.
3. **End-to-End Latency Benchmark:** Time from socket write to database row insertion must be `< 5.0` seconds (target $< 1.2$ seconds).

---

## 9. Acceptance Criteria
- [ ] `aiosmtpd` daemon runs and accepts SMTP connections on configured port.
- [ ] Correctly extracts JPEG attachments from Reolink multi-part MIME formats.
- [ ] YOLOv11 ONNX pipeline classifies target classes with bounding boxes.
- [ ] Successful insertion into `detection_events` table in PostgreSQL.
- [ ] Dispatches typed `DetectionEventProcessed` message to downstream tracking engine.
