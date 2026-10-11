# Spec 03: Kinematic Tracking & Dynamic Isochrone Engine

## 1. Module Overview
The **Kinematic Tracking Engine** is the algorithmic core of Rian. It is responsible for converting discrete, asynchronous, spatially sparse detections (separated by 500m to 3,000m) into continuous probabilistic spatial state estimates across unmonitored terrain.

The engine solves three fundamental mathematical problems:
1. **Track Association under Sensor Sparsity:** Deciding whether an incoming detection belongs to an existing track or represents a new entity via kinematic gating and Hidden Markov Models (HMM).
2. **Terrain Cost-Distance Modeling:** Calculating anisotropic spatial travel impedance using Tobler's Hiking Function and vehicle slope barriers.
3. **Dynamic Isochrone Wavefront Propagation:** Solving the Eikonal equation on friction surfaces to project reachability polygons across blind zones over time.

---

## 2. Dependencies & Prerequisites
- **Contracts:** `specs/00-contracts-and-schemas.md`
- **Spatial Fabric:** `specs/02-spatial-database-rasters.md`
- **Libraries:**
  - `numpy`, `scipy` (Numerical computing and spatial interpolations)
  - `scikit-fmm` or custom priority-queue Eikonal solver (Fast Marching Method)
  - `shapely` (Polygonization, contouring, and vector buffering)
  - `geopy` (Accurate Vincenty/Geodesic distance calculations)
  - `SQLAlchemy` (Async session management for `entity_tracks`)

---

## 3. Mathematical Formulations

### 3.1 Kinematic Spatiotemporal Gating Envelope
When detection $z_k = (x_k, y_k, t_k, c_k, \theta_k)$ arrives:
1. For every active track $T_j$ with last state $(x_j, y_j, t_j, c_j)$:
   $$\Delta t = t_k - t_j$$
   $$d_{jk} = \text{GeodesicDistance}((x_j, y_j), (x_k, y_k))$$
2. Evaluate velocity reachability constraint:
   $$d_{jk} \le v_{max}(c_k) \times \Delta t$$
   If this inequality fails, $P(\text{associate} | T_j) = 0.0$.
3. Candidate tracks passing the envelope proceed to HMM evaluation.

### 3.2 HMM Track Association
$$P(z_k | T_j) = P_{visual}(c_k | T_j) \times P_{spatial}(P_k | T_j, \Delta t, C_{surface})$$
- **$P_{visual}$:** Visual & classification consistency matrix:
  - $P(c_k == c_j) = 1.0$
  - Class transitions (e.g., `car` to `truck`) are assigned low cross-probabilities ($0.15$), while disjoint transitions (e.g., `cow` to `car`) are $0.0$.
- **$P_{spatial}$:** Minimum cost traversal likelihood derived from the terrain friction raster:
  $$P_{spatial} = \exp\left(-\frac{\Delta t_{observed} - \Delta t_{least\_cost}}{\sigma_t}\right)$$
- If $\max_j P(z_k | T_j) \ge \tau_{associate}$ (default $0.5$), the detection updates track $T_j$. Otherwise, a new track $T_{new}$ is created.

### 3.3 Terrain Friction Surface Formulation
The spatial impedance $S(x, y)$ (seconds per meter) combines slope impedance and surface cover:
$$S(x, y) = S_{slope}(x, y) \times C_{land}(x, y)$$

#### Foot / Pedestrian Movement (Tobler's Hiking Function)
$$v_{kmh} = 6 \cdot \exp\left(-3.5 \cdot |\tan \theta + 0.05|\right)$$
$$S_{slope}(x, y) = \frac{1}{v_{kmh} / 3.6} \quad (\text{s/m})$$
Where $\theta$ is the slope angle in radians derived from USGS 3DEP rasters.

#### Wheeled Vehicle Movement
- On road corridors: $S(x, y) = 1 / (v_{road} / 3.6)$
- Slopes $> 15^\circ$ ($> 27\%$ grade) or off-road rocky terrain: $S(x, y) = \infty$ (impassable).

---

## 4. Anisotropic Isochrone Propagation (Eikonal Wavefront)

### 4.1 Continuous Reachability Formulation
The arrival time $T(x, y)$ of the target wavefront satisfies the generalized Eikonal boundary value problem:
$$|\nabla T(x, y)| = S(x, y) = \frac{1}{v(x, y)}$$
Boundary condition: $T(x_0, y_0) = 0$ at last known location.

### 4.2 Fast Marching Solver (`src/tracking/isochrone.py`)
1. Extract local sub-raster centered at $(x_0, y_0)$ with radius $R_{max} = v_{max} \times (t_{now} - t_0)$.
2. Discretize into cost grid with cell size $\Delta x = 10\text{m}$.
3. Solve arrival time grid $T(x, y)$ using Fast Marching Method (`skfmm.travel_time`).
4. Apply directional heading bias:
   $$\text{Weight}(\theta) = \frac{\exp(\kappa \cos(\theta - \theta_0))}{2\pi I_0(\kappa)}$$
   Where $\theta_0$ is the camera boresight/vector heading, and $\kappa$ is the directional concentration parameter.
5. Extract contour level where $T(x, y) \le (t_{now} - t_0)$.
6. Convert contour to GeoJSON Polygon / MultiPolygon, transform to EPSG:4326, and update `entity_tracks.reachability_polygon`.

---

## 5. Periodic Track State Daemon (`src/tracking/state_worker.py`)
- Runs every 15 seconds.
- Queries all active tracks (`current_status = 'ACTIVE'`).
- Computes elapsed time $\Delta t = t_{now} - t_{last\_detected}$.
- If $\Delta t > 30\text{ minutes}$ and no chokepoint reached, status transitions to `STALE`.
- If $\Delta t \le 30\text{ minutes}$, recalculates and saves updated reachability polygon.
- Dispatches updated track state to `Spec 04` Interception solver.

---

## 6. Acceptance Criteria
- [ ] Gating envelope strictly rejects physically impossible detections ($d / \Delta t > v_{max}$).
- [ ] HMM correctly associates consecutive detections along a corridor to the same track.
- [ ] Isochrone correctly channels along roads and avoids steep cliffs for vehicles.
- [ ] Isochrone generation takes $< 1.5$ seconds per active track on standard CPU.
- [ ] Reachability polygon output complies with GeoJSON RFC 7946.
