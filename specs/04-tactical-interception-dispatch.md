# Spec 04: Tactical Interception, Alerts & Common Operating Picture (COP)

## 1. Module Overview
The **Tactical Interception & Dispatch Module** closes the operational loop of Rian. Rather than merely alerting operators that an intrusion has occurred, the engine calculates predictive interception geometry and dispatches tactical directives to interceptors (ranch managers, security patrol vehicles, or law enforcement) before targets reach perimeter exits or critical assets.

It also provides an interactive Common Operating Picture (COP) dashboard displaying live sensor health, active entity tracks, dynamic reachability polygons, and automated response routing.

---

## 2. Dependencies & Prerequisites
- **Contracts:** `specs/00-contracts-and-schemas.md`
- **Spatial Network:** `specs/02-spatial-database-rasters.md`
- **Kinematic State:** `specs/03-kinematic-tracking-engine.md`
- **Libraries:**
  - `pgrouting` & `SQLAlchemy`
  - `twilio` (SMS & WhatsApp API)
  - `fastapi` & `jinja2` or `streamlit` / `Mapbox GL JS`
  - `pydantic` v2

---

## 3. Tactical Interception Geometry

```
                                [ Target Track ]
                        Expanding Isochrone Wavefront R(t)
                                        │
                         Wavefront Intersects Chokepoints
                                        │
                    Target Arrival Time: t_target(C_m)
                                        │
    [ Response Vehicle Location ]       │
                 │                      │
       Dijkstra Road Routing            │
                 ▼                      ▼
    Interceptor Arrival: t_resp(C_m)   t_target(C_m)
                 │                      │
                 └──────────┬───────────┘
                            ▼
              Calculate Tactical Margin:
              delta_t = t_target(C_m) - t_resp(C_m)
                            │
               delta_t >= t_margin (180s - 300s)?
               ├── YES ──> Create Tactical Directive & Dispatch SMS/WhatsApp
               └── NO  ──> Mark Chokepoint Unreachable; Evaluate Next Chokepoint
```

### 3.1 Mathematical Interception Logic
1. Let $\mathcal{C} = \{C_1, C_2, \dots, C_n\}$ be the set of indexed strategic chokepoints (gates, cattle guards, canyon forks) with coordinates $(x_m, y_m)$.
2. For an active track $T_j$, the Eikonal arrival time surface yields target arrival time $t_{target}(C_m)$.
3. For response unit at coordinates $X_r(t)$, transit time $t_{resp}(C_m)$ is solved via Dijkstra's shortest path algorithm on `property_network_edges`:
   $$t_{resp}(C_m) = \text{DijkstraTransitTime}(X_r(t), C_m)$$
4. A chokepoint $C_m$ qualifies as an **Actionable Intercept** if:
   $$t_{target}(C_m) - t_{resp}(C_m) \ge t_{margin}$$
   Where $t_{margin} = 180\text{ to }300\text{ seconds}$ (3 to 5 minutes safety buffer).
5. The chokepoint minimizing $t_{target}(C_m)$ among all actionable chokepoints is selected as the primary intercept objective.

---

## 4. Alert & Directive Dispatch (`src/dispatch/alerter.py`)

### 4.1 Twilio Notification Service
- Supports SMS and WhatsApp channels.
- Configurable environment keys: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `DISPATCH_PHONE_NUMBERS`.
- **Message Template:**
  ```text
  [RIAN TACTICAL ALERT]
  Target: {object_class.upper()}
  Track: {track_code}
  Directive: INTERCEPT AT {chokepoint_name}
  Target ETA: {target_eta_formatted} ({target_minutes_away} min)
  Your ETA: {resp_eta_formatted} ({resp_minutes_away} min)
  Tactical Margin: {margin_minutes} min advantage
  Map: https://rian.domain/cop?track={track_id}
  ```
- **Debounce Policy:** Never dispatch duplicate notifications for the same `(track_id, chokepoint_id)` within 300 seconds unless ETA changes by $> 3\text{ minutes}$.

---

## 5. Common Operating Picture (COP) Dashboard (`src/web/`)

### 5.1 Technology Options
- **Lightweight Prototype:** Streamlit with `folium` or `pydeck`.
- **Production Dashboard:** FastAPI backend serving HTML5/Tailwind + Mapbox GL JS map layer with Server-Sent Events (SSE) or WebSockets.

### 5.2 Visual Layer Architecture
1. **Satellite / Topo Base Map:** Mapbox Outdoors or ESRI World Imagery.
2. **Sensor Nodes Layer:**
   - Icons: Camera marker colored by status (green = active, red = offline).
   - FOV Wedges: Transparent polygon representing boresight azimuth $\theta$ and $85^\circ$ FOV arc.
3. **Active Entity Tracks Layer:**
   - Pulsing red circle at `last_known_point`.
   - Historical path trail as dotted amber LineString.
4. **Dynamic Isochrone Layer:**
   - Semi-transparent gradient polygon (red $\to$ amber $\to$ yellow) representing the dynamic reachability boundary.
5. **Strategic Chokepoints & Intercept Vectors:**
   - Blue shield icons for perimeter gates.
   - Pulsing target highlight on designated active intercept chokepoint.
   - Cyan route polyline showing optimal response vehicle path to chokepoint.

### 5.3 REST & Streaming APIs (`src/web/api.py`)
- `GET /api/v1/cop/state`: Returns snapshot of sensors, tracks, and directives.
- `GET /api/v1/cop/stream`: Server-Sent Events (SSE) streaming real-time detections and track updates.
- `POST /api/v1/interceptor/telemetry`: Updates response vehicle GPS coordinates.
- `POST /api/v1/directives/{id}/status`: Updates directive status (`ACKNOWLEDGED`, `ON_SCENE`, `RESOLVED`).

---

## 6. Acceptance Criteria
- [ ] Interception solver computes arrival time differentials across all perimeter chokepoints in $< 500\text{ms}$.
- [ ] Correctly identifies candidate chokepoints meeting the $t_{margin} \ge 180\text{s}$ criterion.
- [ ] Twilio service dispatches structured SMS/WhatsApp alerts with de-duplication.
- [ ] Dashboard renders live map with camera wedges, tracks, isochrones, and response routing.
- [ ] Response vehicle telemetry updates dynamically recalculate transit times in real time.
