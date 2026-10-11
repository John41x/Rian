"""Ingestion gateway, SMTP daemon, and computer-vision inference pipeline."""

from src.ingestion.service import IngestionService
from src.ingestion.vlm_inspector import (
    BoundingBoxOverlay,
    SnapshotRecord,
    VlmForensicInspector,
    VlmInspectionReport,
    get_vlm_inspector,
)

__all__ = [
    "IngestionService",
    "BoundingBoxOverlay",
    "SnapshotRecord",
    "VlmForensicInspector",
    "VlmInspectionReport",
    "get_vlm_inspector",
]
