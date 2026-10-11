# Spec 14: Dynamic Kinematic Visualization & Uncertainty Halos

## 1. Module Overview & Purpose
Replaces abstract military polyline grids and rigid polygon wavefronts with an intuitive, physically grounded visual language:
1. **Blinking Moving Entity Dot**: An animated pulsating marker placed at the entity's real-time estimated coordinate, moving across the map in real time along physical terrain trails based on kinematic dead reckoning:
   $$X(t) = X_{\text{last}} + \vec{v}_{\text{est}} \cdot (t - t_{\text{last}})$$
2. **Context-Aware Dynamic Uncertainty Halo**: A concentric, hollow animated ring around each dot representing the positional probability boundary.
   - **Shrinks tight** (e.g. 15m–25m) when fresh camera sightings or high-confidence Re-ID matches confirm exact position.
   - **Expands smoothly** as time elapses without new sensor fixes ($\text{radius} \propto \sqrt{\Delta t}$), visually communicating to the rancher or operator that search area has broadened.
3. **Multi-Entity Tracking**: Every tracked animal, vehicle, or predator receives an independent moving dot and confidence halo.
4. **Compact Demonstration Canvas**: Rescales the synthetic demonstration environment from 10km down to a compact **800m to 1,200m estate** so entity movement is visibly dynamic and immediate without extreme zooming.

---

## 2. Mathematical Motion & Uncertainty Formulation

### 2.1 Kinematic Dead Reckoning
For an active track with last confirmed point $(x_0, y_0)$ at timestamp $t_0$, velocity magnitude $v$ (m/s), and heading $\theta$ (degrees clockwise from North):
$$\Delta t = t - t_0$$
$$\Delta x = v \cdot \Delta t \cdot \sin(\theta)$$
$$\Delta y = v \cdot \Delta t \cdot \cos(\theta)$$
$$\text{lon}(t) = \text{lon}_0 + \frac{\Delta x}{111320 \cdot \cos(\text{lat}_0)}, \quad \text{lat}(t) = \text{lat}_0 + \frac{\Delta y}{111320}$$

### 2.2 Dynamic Uncertainty Halo Radius
$$\text{Radius}(t) = R_{\text{base}} + \alpha \cdot \sqrt{\max(0, \Delta t)} \cdot (1.0 - c)$$
Where:
- $R_{\text{base}} = 15.0\text{ meters}$ (optical resolution limit)
- $\alpha = 3.5$ (expansion velocity coefficient)
- $c \in [0.1, 1.0]$ is the `confidence_score` (from YOLO detection confidence and Re-ID similarity)
- When a new camera sighting occurs, $\Delta t \to 0$ and $c \to 1.0$, snapping the halo tight.

---

## 3. Implementation Requirements (`src/web/app.py` & `public/index.html`)

1. **CSS & Canvas Keyframe Animations**:
   - `.blinking-entity-dot`: Core dot with radial pulse animation.
   - `.uncertainty-halo`: Concentric SVG/Leaflet ring with dashed border and breathing glow.
2. **Client-Side Animation Loop (`requestAnimationFrame` / 1000ms tick)**:
   - Interpolates coordinates between state updates so dots visibly glide across pasture trails.
3. **Interactive Hover Tooltip**:
   - Shows entity class, tag code, live speed (km/h & m/s), heading bearing, current pasture zone, and uncertainty radius ($\pm X\text{ meters}$).

---

## 4. Acceptance Criteria
- [ ] Active tracks render as pulsating blinking dots moving across the screen along their heading.
- [ ] Uncertainty halo expands dynamically over elapsed time and contracts when fresh detections arrive.
- [ ] Multiple entities render simultaneously with distinct identifiers and halos.
- [ ] Compact canvas (800m–1200m) shows clearly visible motion across pastures in under 10 seconds.
- [ ] 100% test pass rate in Selenium responsive layout suite.
