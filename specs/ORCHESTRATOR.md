# Rian Master Orchestrator Specification

## 1. System Vision & Purpose
**Rian** is an Autonomous Spatial State Engine built to track terrestrial entities (intruders, vehicles, livestock, predators) across vast off-grid infrastructure (ranches, solar farms, timberlands) using sparse, low-cost cellular IoT nodes (Reolink Go Plus and NDAA-compliant equivalents).

Because nodes are separated by 500m to 3,000m (>95% blind coverage) and operate via bursty egress (SMTP snapshot MIME over LTE), Rian transforms discrete asynchronous snapshots into continuous kinematic tracks through:
1. Class-conditioned kinematic spatiotemporal gating ($v_{max}$).
2. Hidden Markov Model (HMM) track association.
3. Terrain-constrained friction rasters (Tobler hiking function + DEM slope + land-cover impedance).
4. Anisotropic cost-distance isochrone wavefront expansion (Eikonal equation).
5. Dynamic tactical interception routing over topological road networks (pgRouting + Dijkstra).

---

## 2. Orchestration & Asynchronous Subagent Model

This repository is organized to allow multiple specialized subagents to work in parallel on decoupled modules without stepping on each other.

### Subagent Rules of Engagement
1. **Contract Supremacy:** No agent may alter database DDL or inter-module Pydantic schemas in `specs/00-contracts-and-schemas.md` without explicit orchestrator approval.
2. **Interface Isolation:** Each module communicates via strictly typed data contracts (SQL schemas, GeoJSON, and Pydantic message envelopes).
3. **Synthetic Self-Sufficiency:** Every module implementation must run and pass tests against the synthetic test harness (`specs/05-simulation-harness.md`) without requiring physical hardware or internet access.
4. **Specification Status Tracking:** When an agent picks up a spec, it updates the module registry in this document from `READY` to `IN_PROGRESS`, and to `VERIFIED` once unit/integration tests pass.

---

## 3. Module Specification Registry

| Spec ID | Module Name | Target Path | Dependencies | Status | Assigned Agent / Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **00** | Contracts & Schemas | `specs/00-contracts-and-schemas.md` | None | `READY` | Orchestrator Core |
| **01** | Ingestion & CV Inference | `specs/01-ingestion-inference.md` | Spec 00 | `READY` | Ingestion Agent |
| **02** | Spatial Database & Rasters | `specs/02-spatial-database-rasters.md` | Spec 00 | `READY` | Geospatial DB Agent |
| **03** | Kinematic Tracking Engine | `specs/03-kinematic-tracking-engine.md` | Spec 00, 02 | `READY` | Algorithm & Math Agent |
| **04** | Interception & Dispatch | `specs/04-tactical-interception-dispatch.md` | Spec 00, 02, 03 | `READY` | Tactical Dispatch Agent |
| **05** | Simulation & Validation Harness | `specs/05-simulation-harness.md` | Spec 00 | `READY` | QA & Simulation Agent |

### Status Definitions
- `DRAFT`: Under specification; not ready for agent implementation.
- `READY`: Fully specified with input/output contracts, tests, and acceptance criteria. Ready to be assigned.
- `IN_PROGRESS`: Currently being implemented by a subagent.
- `TESTING`: Implementation complete; running integration & simulation suite.
- `VERIFIED`: Passed all acceptance criteria and merged into master state.
- `DEPRECATED`: Replaced or removed by orchestrator.

---

## 4. End-to-End System Architecture

```
                                  [ PHYSICAL WORLD ]
             PIR Motion Trigger / LoRaWAN Seismic Tripwire
                                          │
                                          ▼
                      [ Edge Node: Reolink Go Plus / LTE ]
                             Outbound SMTP / Port 587
                                          │
                                          ▼
             ═══════════════════════════════════════════════════════════
                             [ INGESTION SUBSYSTEM ] (Spec 01)
                     Headless aiosmtpd Listener (:2525)
                                          │
                                 MIME Attachment Parser
                                          │
                                          ▼ (RGB Image Buffer)
                     YOLOv11 ONNX Inference Pipeline (conf >= 0.45)
                                          │
                                          ▼ (DetectionEventPayload)
             ═══════════════════════════════════════════════════════════
                         [ DATABASE & SPATIAL FABRIC ] (Spec 00 & 02)
                   PostgreSQL 16 + PostGIS + pgRouting
                      ├── sensor_nodes
                      ├── detection_events
                      ├── property_network_edges (OSM/Ranch Trails)
                      └── raster_surfaces (USGS 3DEP DEM, Slope, NLCD)
                                          │
                                          ▼
             ═══════════════════════════════════════════════════════════
                         [ KINEMATIC STATE ENGINE ] (Spec 03)
                    Kinematic Gating Envelope (d_ij <= v_max * delta_t)
                                          │
                       HMM Association (P_visual * P_spatial)
                                          │
                       Terrain Friction Cost Surface (Tobler Hiking)
                                          │
                   Dynamic Isochrone Wavefront Solver (Eikonal)
                                          │
                                          ▼ (GeoJSON Reachability Polygon)
             ═══════════════════════════════════════════════════════════
                      [ TACTICAL DISPATCH & DISPLAY ] (Spec 04)
                    Isochrone-Chokepoint Intersection (t_arrival)
                                          │
                     Dijkstra Interceptor Router (t_transit)
                                          │
                       Condition: (t_arrival - t_transit) >= 3 min
                                          │
                       ├── Twilio SMS / WhatsApp Directives
                       └── Streamlit / Mapbox Live Common Operating Picture (COP)
```

---

## 5. Protocol for Adding and Modifying Specs

When a developer or agent wants to modify or introduce a new specification:
1. **RFC in Orchestrator:** Propose the addition in `specs/ORCHESTRATOR.md` under Module Registry with `DRAFT` status.
2. **Contract Compatibility Check:** Verify that new tables, fields, or APIs do not break backwards compatibility with existing consumers in `specs/00-contracts-and-schemas.md`.
3. **Write Spec Document:** Create `specs/NN-<feature-name>.md` following the mandatory template:
   - Module Overview & Purpose
   - Input Interfaces & Dependencies
   - Internal Logic & Algorithms
   - Output Contracts & Side Effects
   - Error Handling & Edge Cases
   - Test Plan & Simulation Harness Integration
   - Acceptance Criteria
4. **Promote to READY:** Once the spec meets all template criteria, change status to `READY`.
5. **Agent Assignment:** Dispatch a subagent with bounded scope targeting only that spec.

---

## 6. Directory Layout Standard

```
.
├── Rian Plan.txt                  # Original foundational vision & research paper
├── specs/                         # Modular engineering specifications
│   ├── ORCHESTRATOR.md            # Master registry & orchestration instructions
│   ├── 00-contracts-and-schemas.md# Canonical DDL, Pydantic models, GeoJSON formats
│   ├── 01-ingestion-inference.md  # SMTP listener & YOLOv11 ONNX pipeline
│   ├── 02-spatial-database-rasters.md # PostGIS, DEM/slope, routing topology
│   ├── 03-kinematic-tracking-engine.md# Association, friction, isochrones
│   ├── 04-tactical-interception-dispatch.md # Chokepoints, Dijkstra, Twilio, UI
│   └── 05-simulation-harness.md   # Mock camera SMTP client & synthetic GIS
├── src/                           # Codebase (to be built by subagents)
│   ├── common/                    # Shared Pydantic schemas, DB session, logger
│   ├── ingestion/                 # SMTP daemon & YOLO inference
│   ├── spatial/                   # PostGIS raster & vector services
│   ├── tracking/                  # Kinematic state & isochrone engine
│   ├── dispatch/                  # Interception solver & alert service
│   └── web/                       # Operator map dashboard
├── tests/                         # Unit & integration test suites
│   ├── test_ingestion/
│   ├── test_tracking/
│   └── test_dispatch/
└── tools/                         # Simulation CLI, test seeders, synthetic DEMs
```
