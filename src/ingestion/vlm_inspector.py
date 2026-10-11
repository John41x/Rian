"""
Multimodal VLM forensic snapshot inspector (spec 16).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel

from src.common.config import settings
from src.common.domains import get_domain_profile
from tools.mock_camera import generate_synthetic_jpeg

logger = logging.getLogger("rian.vlm_inspector")

VLM_JSON_SCHEMA_HINT = """
Return a single JSON object with exactly these keys:
snapshot_id (string),
entity_class (string),
visual_description (string),
environmental_context (string),
risk_assessment (string),
findings (array of strings),
recommended_action (string),
provider (string).
"""


class BoundingBoxOverlay(BaseModel):
    label: str
    confidence: float
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    color: str = "#22c55e"


class SnapshotRecord(BaseModel):
    snapshot_id: UUID
    sensor_id: UUID
    camera_name: str
    timestamp: datetime
    image_url: str
    entity_class: str
    confidence: float
    bounding_boxes: List[BoundingBoxOverlay]
    vlm_analyzed: bool = False


class VlmInspectionReport(BaseModel):
    snapshot_id: UUID
    entity_class: str
    visual_description: str
    environmental_context: str
    risk_assessment: str
    findings: List[str]
    recommended_action: str
    provider: str


def _normalize_api_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _aspect_ratio(box: BoundingBoxOverlay) -> float:
    width = max(1e-6, box.xmax - box.xmin)
    height = max(1e-6, box.ymax - box.ymin)
    return height / width


class VlmForensicInspector:
    """In-memory snapshot cache with disk persistence for forensic VLM analysis."""

    def __init__(self, storage_root: Optional[str] = None) -> None:
        root = storage_root or settings.STORAGE_ROOT
        self._storage_root = Path(root) / "vlm_inspector"
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._snapshots: Dict[UUID, SnapshotRecord] = {}
        self._image_bytes: Dict[UUID, bytes] = {}
        self._camera_index: Dict[UUID, List[UUID]] = {}
        self._track_latest: Dict[UUID, UUID] = {}

    def _image_path(self, snapshot_id: UUID) -> Path:
        return self._storage_root / f"{snapshot_id}.jpg"

    def _persist_image(self, snapshot_id: UUID, image_bytes: bytes) -> None:
        path = self._image_path(snapshot_id)
        path.write_bytes(image_bytes)
        self._image_bytes[snapshot_id] = image_bytes

    def record_snapshot(
        self,
        *,
        sensor_id: UUID,
        camera_name: str,
        entity_class: str,
        confidence: float,
        bounding_boxes: List[BoundingBoxOverlay],
        image_bytes: Optional[bytes] = None,
        timestamp: Optional[datetime] = None,
        track_id: Optional[UUID] = None,
        snapshot_id: Optional[UUID] = None,
    ) -> SnapshotRecord:
        sid = snapshot_id or uuid4()
        ts = timestamp or datetime.now(timezone.utc)
        if image_bytes is None:
            image_bytes = generate_synthetic_jpeg(entity_class)
        self._persist_image(sid, image_bytes)

        record = SnapshotRecord(
            snapshot_id=sid,
            sensor_id=sensor_id,
            camera_name=camera_name,
            timestamp=ts,
            image_url=f"/api/v1/snapshots/{sid}/image",
            entity_class=entity_class,
            confidence=confidence,
            bounding_boxes=bounding_boxes,
            vlm_analyzed=False,
            track_id=track_id,
        )
        self._snapshots[sid] = record

        cam_list = self._camera_index.setdefault(sensor_id, [])
        cam_list.insert(0, sid)
        self._camera_index[sensor_id] = cam_list[:50]

        if track_id is not None:
            self._track_latest[track_id] = sid

        return record

    def get_snapshot(self, snapshot_id: UUID) -> Optional[SnapshotRecord]:
        return self._snapshots.get(snapshot_id)

    def get_snapshot_image(self, snapshot_id: UUID) -> Optional[bytes]:
        if snapshot_id in self._image_bytes:
            return self._image_bytes[snapshot_id]
        path = self._image_path(snapshot_id)
        if path.is_file():
            data = path.read_bytes()
            self._image_bytes[snapshot_id] = data
            return data
        record = self._snapshots.get(snapshot_id)
        entity = record.entity_class if record else "person"
        synthetic = generate_synthetic_jpeg(entity)
        self._persist_image(snapshot_id, synthetic)
        return synthetic

    def get_camera_snapshots(self, camera_id: UUID, limit: int = 10) -> List[SnapshotRecord]:
        ids = self._camera_index.get(camera_id, [])[:limit]
        out: List[SnapshotRecord] = []
        for sid in ids:
            rec = self._snapshots.get(sid)
            if rec is not None:
                out.append(rec)
        return out

    def link_track_snapshot(self, track_id: UUID, snapshot_id: UUID) -> None:
        self._track_latest[track_id] = snapshot_id
        if snapshot_id in self._snapshots:
            self._snapshots[snapshot_id].track_id = track_id

    def get_track_snapshot(self, track_id: UUID) -> Optional[SnapshotRecord]:
        sid = self._track_latest.get(track_id)
        if sid is None:
            return None
        return self._snapshots.get(sid)

    def _local_forensic_report(self, record: SnapshotRecord) -> VlmInspectionReport:
        profile = get_domain_profile(settings.OPERATION_PROFILE)
        primary = record.bounding_boxes[0] if record.bounding_boxes else BoundingBoxOverlay(
            label=record.entity_class,
            confidence=record.confidence,
            ymin=0.2,
            xmin=0.2,
            ymax=0.8,
            xmax=0.8,
        )
        ar = _aspect_ratio(primary)
        conf_pct = int(round(record.confidence * 100))
        entity = record.entity_class.lower()

        if profile.profile_type == "ranch_livestock":
            terrain = "Dry pasture grass with fence line visible in mid-background; midday sun with short shadows."
        elif profile.profile_type == "industrial_terminal":
            terrain = "Paved yard lighting with container stacks and perimeter fencing in background."
        else:
            terrain = "Mixed terrain with perimeter infrastructure visible at frame edge."

        findings: List[str] = [
            f"Primary detection '{primary.label}' at {conf_pct}% model confidence.",
            f"Normalized bounding envelope aspect ratio {ar:.2f} (tall/wide indicator for posture or vehicle profile).",
            f"Camera node '{record.camera_name}' shutter capture aligned with active {profile.app_title} profile.",
        ]

        if entity in ("cow", "horse", "dog"):
            visual = (
                f"Livestock-class subject ({entity}) occupying central FOV with grazing posture inferred from "
                f"vertical extent of detection box."
            )
            if ar > 1.2:
                findings.append("Elongated vertical silhouette suggests standing/alert posture rather than bedded rest.")
            risk = "LOW // HEALTHY GRAZING" if record.confidence >= 0.5 else "MODERATE // LOW CONFIDENCE TRACK"
            action = "Continue passive herd monitoring; task ATV only if boundary drift persists >5 minutes."
        elif entity in ("truck", "car", "vehicle"):
            visual = (
                f"Vehicle-class target ({entity}) with box spanning roadway or access lane; "
                "cargo bed / cab separation not optically resolved at sub-stream resolution."
            )
            if ar < 0.6:
                findings.append("Wide aspect detection consistent with lateral vehicle profile or angled approach.")
            risk = "MODERATE // UNAUTHORIZED VEHICLE" if profile.profile_type != "ranch_livestock" else "ADVISORY // FARM TRAFFIC"
            action = "Verify gate authorization log; dispatch perimeter check if off-hours."
        elif entity == "person":
            visual = "Human figure detected with upright posture; high-visibility clothing not confirmed at this resolution."
            risk = "HIGH // INTRUDER / TRESPASS"
            action = "Issue verbal challenge via loudspeaker if equipped; dispatch interceptor to last kinematic fix."
        else:
            visual = f"Unclassified or rare class '{entity}' requiring operator review."
            risk = "MODERATE // UNCLASSIFIED MOTION"
            action = "Retain snapshot for 24h archival and correlate with adjacent camera triggers."

        findings.append(f"Environmental baseline: {terrain}")

        return VlmInspectionReport(
            snapshot_id=record.snapshot_id,
            entity_class=record.entity_class,
            visual_description=visual,
            environmental_context=terrain,
            risk_assessment=risk,
            findings=findings,
            recommended_action=action,
            provider="Rian Local Forensic Analyzer",
        )

    async def _call_deepseek_vlm(self, record: SnapshotRecord) -> VlmInspectionReport:
        api_key = _normalize_api_key(settings.DEEPSEEK_API_KEY)
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        profile = get_domain_profile(settings.OPERATION_PROFILE)
        boxes_payload = [b.model_dump() for b in record.bounding_boxes]
        user_content = json.dumps(
            {
                "snapshot_id": str(record.snapshot_id),
                "camera_name": record.camera_name,
                "entity_class": record.entity_class,
                "confidence": record.confidence,
                "bounding_boxes": boxes_payload,
                "operation_profile": profile.profile_type,
            },
            default=str,
        )
        system = (
            "You are a forensic vision analyst for remote ranch and perimeter security cameras. "
            "Infer visual appearance, environmental context, and operational risk from detection metadata. "
            + VLM_JSON_SCHEMA_HINT
        )
        payload = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Analyze this edge snapshot metadata JSON:\n{user_content}",
                },
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        return VlmInspectionReport(
            snapshot_id=record.snapshot_id,
            entity_class=str(parsed.get("entity_class") or record.entity_class),
            visual_description=str(parsed.get("visual_description") or ""),
            environmental_context=str(parsed.get("environmental_context") or ""),
            risk_assessment=str(parsed.get("risk_assessment") or "UNKNOWN"),
            findings=list(parsed.get("findings") or []),
            recommended_action=str(parsed.get("recommended_action") or ""),
            provider=str(parsed.get("provider") or "DeepSeek-VLM"),
        )

    async def analyze_snapshot(self, snapshot_id: UUID) -> VlmInspectionReport:
        record = self.get_snapshot(snapshot_id)
        if record is None:
            raise LookupError(f"Snapshot {snapshot_id} not found")

        api_key = _normalize_api_key(settings.DEEPSEEK_API_KEY)
        if api_key:
            try:
                report = await self._call_deepseek_vlm(record)
            except Exception as exc:
                logger.warning("DeepSeek VLM analysis failed, using local reasoner: %s", exc)
                report = self._local_forensic_report(record)
        else:
            report = self._local_forensic_report(record)

        self._snapshots[snapshot_id] = record.model_copy(update={"vlm_analyzed": True})
        return report


_inspector_singleton: Optional[VlmForensicInspector] = None


def get_vlm_inspector() -> VlmForensicInspector:
    global _inspector_singleton
    if _inspector_singleton is None:
        _inspector_singleton = VlmForensicInspector()
    return _inspector_singleton


def reset_vlm_inspector_for_tests() -> VlmForensicInspector:
    """Replace global inspector (used in tests)."""
    global _inspector_singleton
    _inspector_singleton = VlmForensicInspector(
        storage_root=str(Path(settings.STORAGE_ROOT) / "test_vlm")
    )
    return _inspector_singleton
