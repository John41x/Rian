# Spec 08: PostGIS HMM Road Snapping & Map Matching

## 1. Module Overview & Purpose
In rural ranches and solar fields, vehicular targets and intruders navigate along topological road networks (two-track trails, graded utility corridors, perimeter fences). Standalone GPS coordinates from camera detections suffer from orientation offset, PIR trigger delays (0.8s–1.2s), and spatial noise.

This module adapts the Hidden Markov Model (HMM) map-matching approach from **`amillb/pgMapMatch`** directly into PostGIS. Given an incoming detection or sequence of sightings:
1. Candidate edges in `property_network_edges` within distance buffer $r_{cand} \le 150\text{m}$ are identified using spatial indices (`ST_DWithin`).
2. Emission probability $P(z_t \mid e_i)$ is evaluated based on perpendicular projection distance:
   $$P(z_t \mid e_i) = \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left( -\frac{\text{dist}(z_t, e_i)^2}{2\sigma_z^2} \right)$$
3. Spatial transition probability $P(e_j \mid e_i)$ evaluates topological distance along network edges via `pgr_dijkstra` against Euclidean distance:
   $$P(e_j \mid e_i) = \frac{1}{\beta} \exp\left( -\frac{|d_{\text{network}}(e_i, e_j) - d_{\text{euclidean}}(z_{t-1}, z_t)|}{\beta} \right)$$
4. The most probable sequence of edges is identified, snapping the track's path precisely to physical roadways and computing realistic arrival times.

---

## 2. Dependencies & Reference Architecture
- **Reference Repo:** `external_references/pgMapMatch` (`mapmatcher.py`, SQL queries).
- **Target Module Path:** `src/spatial/map_match.py`.
- **Test Path:** `tests/test_spatial/test_map_match.py`.

---

## 3. Implementation Requirements (`src/spatial/map_match.py`)

1. **`RoadMapMatcher`**:
   - `find_candidate_edges(lon: float, lat: float, radius_m: float = 150.0) -> List[CandidateEdge]`: Queries `property_network_edges` via spatial SQL to find candidate road segments with projected point coordinates.
   - `calculate_emission_probability(distance_m: float, sigma_z: float = 20.0) -> float`: Gaussian distance likelihood.
   - `calculate_transition_probability(network_dist_m: float, euclidean_dist_m: float, beta: float = 30.0) -> float`: Exponential route discrepancy likelihood.
   - `snap_point_to_network(lon: float, lat: float) -> Tuple[float, float, int]`: Snaps raw sighting to the highest-likelihood edge, returning `(snapped_lon, snapped_lat, edge_id)`.
2. **Integration into `SpatialService`**:
   - Add `map_match_track(history_coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]` to smooth jittery camera trails along true topological road polylines.

---

## 4. Acceptance Criteria
- [ ] Detections within 100m of a road corridor correctly snap to the nearest edge centerline.
- [ ] Off-road sightings beyond 150m return raw coordinates without forcing false edge snapping.
- [ ] Route snapping produces continuous, gap-free LineString geometries along `property_network_edges`.
- [ ] 100% test pass rate in `tests/test_spatial/test_map_match.py`.
