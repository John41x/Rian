"""YOLOv11 object detection for camera snapshot frames."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from PIL import Image

from src.common.config import settings
from src.common.schemas import BoundingBox, DetectionItem, EntityClass

logger = logging.getLogger(__name__)

TARGET_CLASS_NAMES: Set[str] = {c.value for c in EntityClass}
DEFAULT_MODEL_NAME = "yolo11n.pt"


@dataclass
class InferenceConfig:
    confidence_threshold: float = settings.CONFIDENCE_THRESHOLD
    model_name: str = DEFAULT_MODEL_NAME
    force_mock: bool = False


class YOLOInferenceProcessor:
    """Lazy-loaded Ultralytics YOLO with mock fallback for tests and offline use."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        self._model = None
        self._mock_mode = self.config.force_mock or _env_mock_enabled()

    def _ensure_model(self) -> bool:
        if self._mock_mode:
            return False
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_name)
            return True
        except Exception:
            logger.warning(
                "YOLO model unavailable (%s); using mock inference",
                self.config.model_name,
                exc_info=True,
            )
            self._mock_mode = True
            return False

    def predict(self, image_bytes: bytes) -> List[DetectionItem]:
        if self._ensure_model():
            return self._predict_ultralytics(image_bytes)
        return self._predict_mock(image_bytes)

    async def predict_async(self, image_bytes: bytes) -> List[DetectionItem]:
        return self.predict(image_bytes)

    def _predict_ultralytics(self, image_bytes: bytes) -> List[DetectionItem]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        results = self._model.predict(
            source=image,
            conf=self.config.confidence_threshold,
            verbose=False,
        )
        detections: List[DetectionItem] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        names = results[0].names or {}
        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = str(names.get(cls_id, "")).lower()
            confidence = float(box.conf.item())
            if class_name not in TARGET_CLASS_NAMES:
                continue
            if confidence < self.config.confidence_threshold:
                continue
            xyxy = box.xyxy[0].tolist()
            bbox = _normalize_xyxy(xyxy, width, height)
            detections.append(
                DetectionItem(
                    object_class=EntityClass(class_name),
                    confidence=confidence,
                    bounding_box=bbox,
                )
            )
        return detections

    def _predict_mock(self, image_bytes: bytes) -> List[DetectionItem]:
        """Deterministic mock output for unit tests when weights are unavailable."""
        _ = image_bytes
        return [
            DetectionItem(
                object_class=EntityClass.PERSON,
                confidence=0.92,
                bounding_box=BoundingBox(
                    ymin=0.1, xmin=0.2, ymax=0.6, xmax=0.5
                ),
            )
        ]


def _normalize_xyxy(
    xyxy: Sequence[float], width: int, height: int
) -> BoundingBox:
    x1, y1, x2, y2 = xyxy
    w = max(width, 1)
    h = max(height, 1)
    ymin = max(0.0, min(1.0, y1 / h))
    xmin = max(0.0, min(1.0, x1 / w))
    ymax = max(0.0, min(1.0, y2 / h))
    xmax = max(0.0, min(1.0, x2 / w))
    return BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)


def _env_mock_enabled() -> bool:
    return os.environ.get("RIAN_INFERENCE_MOCK", "").lower() in {
        "1",
        "true",
        "yes",
    }
