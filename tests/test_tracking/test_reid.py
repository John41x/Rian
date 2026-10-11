"""Tests for Spec 07: Cross-Camera Re-ID & Appearance Embeddings."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.common.schemas import EntityClass
from src.tracking.association import (
    ActiveTrackRow,
    DetectionAssociationInput,
    compute_association_score,
)
from src.tracking.reid import OSNetFeatureExtractor, cosine_similarity


def _make_test_image(color: tuple[int, int, int], pattern: str = "flat") -> Image.Image:
    arr = np.zeros((200, 100, 3), dtype=np.uint8)
    arr[:, :] = color
    if pattern == "stripe":
        arr[50:100, :, :] = (255 - color[0], 255 - color[1], 255 - color[2])
    return Image.fromarray(arr)


def test_extractor_output_dimension_and_normalization():
    extractor = OSNetFeatureExtractor()
    img = _make_test_image((120, 40, 200))
    emb = extractor.extract(img)

    assert isinstance(emb, list)
    assert len(emb) == 512
    norm = float(np.linalg.norm(np.array(emb)))
    assert abs(norm - 1.0) < 1e-4


def test_cosine_similarity_identical_and_different():
    extractor = OSNetFeatureExtractor()
    img_red_1 = _make_test_image((220, 20, 20))
    img_red_2 = _make_test_image((225, 25, 18))  # very similar red
    img_blue = _make_test_image((20, 40, 220))   # distinctly different blue

    emb_red_1 = extractor.extract(img_red_1)
    emb_red_2 = extractor.extract(img_red_2)
    emb_blue = extractor.extract(img_blue)

    sim_identical = cosine_similarity(emb_red_1, emb_red_1)
    assert abs(sim_identical - 1.0) < 1e-5

    sim_similar = cosine_similarity(emb_red_1, emb_red_2)
    sim_different = cosine_similarity(emb_red_1, emb_blue)

    assert sim_similar > 0.85
    assert sim_similar > sim_different


def test_compute_association_score_with_reid_discrimination():
    from datetime import datetime, timezone

    extractor = OSNetFeatureExtractor()
    red_truck = _make_test_image((200, 30, 30))
    blue_truck = _make_test_image((30, 30, 200))

    emb_red = extractor.extract(red_truck)
    emb_blue = extractor.extract(blue_truck)

    t0 = datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 10, 20, 2, tzinfo=timezone.utc)

    # Active track was a red truck
    track = ActiveTrackRow(
        track_id="00000000-0000-0000-0000-000000000001",
        track_code="TRK-RED-TRUCK",
        object_class=EntityClass.TRUCK,
        last_detected_at=t0,
        last_lon=-119.82,
        last_lat=36.74,
        estimated_heading_deg=0.0,
        estimated_velocity_ms=10.0,
        confidence_score=0.9,
        last_reid_embedding=emb_red,
    )

    # Detection 1: Sighting of red truck
    det_same = DetectionAssociationInput(
        detection_lon=-119.818,
        detection_lat=36.742,
        detection_time=t1,
        object_class=EntityClass.TRUCK,
        reid_embedding=emb_red,
    )

    # Detection 2: Sighting of blue truck at same spot
    det_diff = DetectionAssociationInput(
        detection_lon=-119.818,
        detection_lat=36.742,
        detection_time=t1,
        object_class=EntityClass.TRUCK,
        reid_embedding=emb_blue,
    )

    friction = np.ones((61, 61), dtype=np.float64) * 0.1
    score_same, gated_same = compute_association_score(det_same, track, friction, -119.819, 36.741)
    score_diff, gated_diff = compute_association_score(det_diff, track, friction, -119.819, 36.741)

    assert gated_same and gated_diff
    assert score_same > score_diff
    assert score_same >= 0.70
