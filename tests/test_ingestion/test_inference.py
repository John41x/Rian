"""YOLO inference processor tests (mock mode)."""

import io

import pytest
from PIL import Image

from src.common.schemas import EntityClass
from src.ingestion.inference import (
    InferenceConfig,
    YOLOInferenceProcessor,
    _normalize_xyxy,
)


def test_mock_inference_output_format(synthetic_jpeg_bytes):
    processor = YOLOInferenceProcessor(
        InferenceConfig(force_mock=True, confidence_threshold=0.45)
    )
    detections = processor.predict(synthetic_jpeg_bytes)

    assert len(detections) == 1
    item = detections[0]
    assert item.object_class == EntityClass.PERSON
    assert item.confidence >= 0.45
    bbox = item.bounding_box
    assert 0.0 <= bbox.ymin <= 1.0
    assert 0.0 <= bbox.xmin <= 1.0
    assert 0.0 <= bbox.ymax <= 1.0
    assert 0.0 <= bbox.xmax <= 1.0
    assert bbox.ymax >= bbox.ymin
    assert bbox.xmax >= bbox.xmin


def test_normalize_xyxy_to_ymin_xmin_ymax_xmax():
    bbox = _normalize_xyxy([10, 20, 30, 40], width=100, height=200)
    assert bbox.ymin == pytest.approx(0.1)
    assert bbox.xmin == pytest.approx(0.1)
    assert bbox.ymax == pytest.approx(0.2)
    assert bbox.xmax == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_predict_async_matches_sync(synthetic_jpeg_bytes):
    processor = YOLOInferenceProcessor(InferenceConfig(force_mock=True))
    sync = processor.predict(synthetic_jpeg_bytes)
    async_result = await processor.predict_async(synthetic_jpeg_bytes)
    assert sync == async_result
