# Spec 07: Cross-Camera Re-ID & Appearance Embeddings

## 1. Module Overview & Purpose
In vast off-grid infrastructure, camera nodes are separated by 500m to 3,000m. Single-frame object detectors (YOLOv11) only output bounding boxes and coarse class labels (`person`, `car`, `truck`). When two white trucks or two workers transit the property, spatial gating alone cannot disambiguate identity across blind zones.

This module introduces a lightweight **deep feature re-identification (Re-ID)** pipeline adapted from **BoxMOT (`mikel-brostrom/yolo_tracking`)** using an **OSNet $\times 0.25$** feature backbone. For every detection bounding box, a compact 512-dimensional $L_2$-normalized appearance vector $\mathbf{e}_k$ is extracted and stored. During HMM association, visual similarity is computed using cosine distance:

$$S_{\text{appearance}}(D_1, D_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|}$$

---

## 2. Dependencies & Reference Architecture
- **Reference Repo:** `external_references/yolo_tracking` (`boxmot/reid/backbones/families/osnet/` and `boxmot/configs/reid/osnet-x0-25-msmt17.yaml`).
- **Contracts:** `specs/00-contracts-and-schemas.md` (`DetectionItem`, `DetectionEventProcessed`, `TrackState`).
- **Target Module Path:** `src/tracking/reid.py` + integration into `src/ingestion/inference.py` and `src/tracking/association.py`.
- **Test Path:** `tests/test_tracking/test_reid.py`.

---

## 3. Mathematical & Algorithmic Integration

### 3.1 OSNet Lightweight Architecture
- Input: Cropped bounding box tensor resized to $256 \times 128$ (RGB).
- Normalization: ImageNet mean $([0.485, 0.456, 0.406])$ and standard deviation $([0.229, 0.224, 0.225])$.
- Output: 512-dimensional feature vector $\mathbf{e}_k \in \mathbb{R}^{512}$, $L_2$-normalized such that $\|\mathbf{e}_k\|_2 = 1.0$.

### 3.2 Enhanced HMM Emission Probability
In `src/tracking/association.py`, replace discrete class matching with hybrid class + appearance scoring:
$$P_{\text{visual}}(z_k \mid T_j) = w_{\text{class}} \cdot \mathbf{1}_{c_k = c_j} + w_{\text{reid}} \cdot \max(0.0, S_{\text{appearance}}(\mathbf{e}_k, \mathbf{e}_{T_j}))$$

Where:
- $w_{\text{class}} = 0.3$ (coarse type filter)
- $w_{\text{reid}} = 0.7$ (fine visual appearance similarity)
- When Re-ID embeddings are unavailable or offline, gracefully falls back to coarse class-similarity table.

---

## 4. Implementation Requirements (`src/tracking/reid.py`)

1. **`OSNetFeatureExtractor`**:
   - Implements lightweight OSNet architecture (Omni-Scale Network with multi-scale depthwise separable convolutions) or pre-trained ONNX/PyTorch feature extractor.
   - Extracts 512-d normalized embedding vector from PIL Image or NumPy RGB crop.
   - Runs in under 15ms per crop on CPU.
2. **Schema Update**:
   - Optional `reid_embedding: Optional[List[float]]` on `DetectionItem`.
   - `last_reid_embedding: Optional[List[float]]` stored on `ActiveTrackRow` / `entity_tracks` table.
3. **Cosine Metric**:
   - `cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float`.

---

## 5. Acceptance Criteria
- [ ] `OSNetFeatureExtractor` extracts exactly 512-d $L_2$-normalized float vector for any RGB image crop.
- [ ] Crops of the same vehicle/person yield $S_{\text{appearance}} \ge 0.75$.
- [ ] Crops of distinctly different entities yield $S_{\text{appearance}} < 0.40$.
- [ ] `compute_association_score` incorporates appearance similarity when embeddings are present.
- [ ] Deterministic fallback preserves 100% backward compatibility when Re-ID is disabled.
