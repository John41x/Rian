"""
Deterministic input/output guards for tactical SITREP generation.

Cross-references LLM-produced entity names against verified COP telemetry
and clips user-supplied text to reduce prompt-injection surface.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

_MAX_FIELD_CHARS = 2000
_MAX_QUERY_CHARS = 512
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clip_text(value: str, max_len: int = _MAX_FIELD_CHARS) -> str:
    """Strip control characters and bound string length."""
    cleaned = _CONTROL_CHAR_RE.sub("", value or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len]


def sanitize_operator_query(query: str) -> str:
    """Sanitize natural-language operator input before LLM prompts."""
    clipped = clip_text(query, _MAX_QUERY_CHARS)
    lowered = clipped.lower()
    for token in ("ignore previous", "system:", "assistant:", "you are now"):
        if token in lowered:
            return clip_text(clipped[:200], 200)
    return clipped


def _name_from_record(record: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        val = record.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def collect_verified_entity_names(cop_state: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (track_codes, chokepoint_names, sensor_names) present in telemetry."""
    tracks: Set[str] = set()
    chokepoints: Set[str] = set()
    sensors: Set[str] = set()

    for track in cop_state.get("tracks") or []:
        code = _name_from_record(track, "track_code", "id")
        if code:
            tracks.add(code)

    for cp in cop_state.get("chokepoints") or []:
        name = _name_from_record(cp, "name", "chokepoint_name")
        if name:
            chokepoints.add(name)

    for directive in cop_state.get("directives") or []:
        name = _name_from_record(directive, "chokepoint_name")
        if name:
            chokepoints.add(name)
        code = _name_from_record(directive, "track_code")
        if code:
            tracks.add(code)

    for sensor in cop_state.get("sensors") or []:
        name = _name_from_record(sensor, "node_name", "name", "id")
        if name:
            sensors.add(str(name))

    return tracks, chokepoints, sensors


def build_fenced_evidence(cop_state: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot ground-truth telemetry used for evidence fencing."""
    tracks, chokepoints, sensors = collect_verified_entity_names(cop_state)
    return {
        "track_codes": sorted(tracks),
        "chokepoint_names": sorted(chokepoints),
        "sensor_names": sorted(sensors),
        "tracks": cop_state.get("tracks") or [],
        "chokepoints": cop_state.get("chokepoints") or [],
        "directives": cop_state.get("directives") or [],
        "sensors": cop_state.get("sensors") or [],
    }


def _chokepoint_phrases_in_text(text: str, valid_chokepoints: Iterable[str]) -> Set[str]:
    found: Set[str] = set()
    for name in valid_chokepoints:
        if name and name in text:
            found.add(name)
    return found


def _track_codes_in_text(text: str, valid_tracks: Iterable[str]) -> Set[str]:
    found: Set[str] = set()
    for code in valid_tracks:
        if code and re.search(rf"\b{re.escape(code)}\b", text):
            found.add(code)
    return found


def _hallucinated_chokepoint_phrases(
    text: str,
    valid_chokepoints: FrozenSet[str],
) -> List[str]:
    """Detect multi-word Title Case place names not in verified chokepoint set."""
    if not text:
        return []
    skip = {
        "UTC",
        "DEFCON",
        "RIAN",
        "SITREP",
        "COP",
        "LTE",
        "JSON",
        "TRK",
        "RED",
        "GREEN",
        "AMBER",
    }
    valid_lower = {c.lower() for c in valid_chokepoints}
    rejected: List[str] = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        phrase = match.group(0)
        if phrase in skip:
            continue
        if phrase.lower() in valid_lower:
            continue
        if any(phrase.lower() in v.lower() or v.lower() in phrase.lower() for v in valid_chokepoints):
            continue
        rejected.append(phrase)
    return rejected


def fence_text_entity_references(
    text: str,
    valid_chokepoints: FrozenSet[str],
    valid_tracks: FrozenSet[str],
    *,
    allow_unknown: bool = False,
) -> Tuple[str, List[str]]:
    """
    Remove references to chokepoints/tracks not present in verified telemetry.

    Returns (fenced_text, rejected_entity_names).
    """
    if allow_unknown or not text:
        return text, []

    rejected: List[str] = []
    result = text

    _track_codes_in_text(text, valid_tracks)
    _chokepoint_phrases_in_text(text, valid_chokepoints)

    for phrase in _hallucinated_chokepoint_phrases(text, valid_chokepoints):
        rejected.append(phrase)
        result = result.replace(phrase, "[REDACTED_UNVERIFIED]")

    track_pattern = re.compile(r"\b[A-Z]{2,}-\d+\b")
    for match in track_pattern.finditer(text):
        code = match.group(0)
        if code not in valid_tracks and code.upper() not in {t.upper() for t in valid_tracks}:
            rejected.append(code)
            result = re.sub(rf"\b{re.escape(code)}\b", "[REDACTED_UNVERIFIED]", result)

    return result, rejected


def sanitize_cop_state_for_prompt(cop_state: Dict[str, Any], max_json_chars: int = 12000) -> Dict[str, Any]:
    """Deep-copy COP state with clipped string fields for safe LLM prompts."""
    import json

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return clip_text(obj, 500)
        if isinstance(obj, list):
            return [_walk(item) for item in obj[:50]]
        if isinstance(obj, dict):
            return {str(k)[:64]: _walk(v) for k, v in list(obj.items())[:80]}
        return obj

    trimmed = _walk(dict(cop_state))
    serialized = json.dumps(trimmed, default=str)
    if len(serialized) <= max_json_chars:
        return trimmed
    return {
        "tracks": trimmed.get("tracks", [])[:10],
        "directives": trimmed.get("directives", [])[:10],
        "chokepoints": trimmed.get("chokepoints", [])[:20],
        "sensors": trimmed.get("sensors", [])[:20],
        "interceptor": trimmed.get("interceptor"),
        "generated_at": trimmed.get("generated_at"),
    }


def apply_output_evidence_guards(
    report: Dict[str, Any],
    cop_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fence LLM output fields and attach verified evidence metadata.

    Returns (guarded_report_dict, list_of_rejected_hallucinations).
    """
    tracks, chokepoints, sensors = collect_verified_entity_names(cop_state)
    track_frozen = frozenset(tracks)
    cp_frozen = frozenset(chokepoints)

    all_rejected: List[str] = []
    guarded = dict(report)

    for field in (
        "headline",
        "primary_objective",
        "target_summary",
        "threat_status",
    ):
        if field in guarded and isinstance(guarded[field], str):
            fenced, rejected = fence_text_entity_references(
                guarded[field], cp_frozen, track_frozen
            )
            guarded[field] = fenced
            all_rejected.extend(rejected)

    actions = guarded.get("recommended_actions") or []
    if isinstance(actions, list):
        fenced_actions: List[str] = []
        for action in actions:
            if not isinstance(action, str):
                continue
            fenced, rejected = fence_text_entity_references(action, cp_frozen, track_frozen)
            fenced_actions.append(fenced)
            all_rejected.extend(rejected)
        guarded["recommended_actions"] = fenced_actions

    objective_cp = guarded.get("primary_objective", "")
    for cp_name in chokepoints:
        if cp_name in objective_cp or cp_name.lower() in objective_cp.lower():
            break
    else:
        if chokepoints and guarded.get("primary_objective"):
            default_cp = sorted(chokepoints)[0]
            guarded["primary_objective"] = clip_text(
                f"Verified intercept at {default_cp}"
            )

    guarded["fenced_evidence"] = build_fenced_evidence(cop_state)
    if all_rejected:
        guarded["fenced_evidence"]["rejected_hallucinations"] = sorted(set(all_rejected))

    margin = guarded.get("tactical_margin_minutes")
    if margin is not None:
        try:
            m = float(margin)
            guarded["tactical_margin_minutes"] = max(0.0, min(m, 10_000.0))
        except (TypeError, ValueError):
            guarded["tactical_margin_minutes"] = 0.0

    defcon = guarded.get("defcon_level", 5)
    try:
        d = int(defcon)
        guarded["defcon_level"] = max(1, min(d, 5))
    except (TypeError, ValueError):
        guarded["defcon_level"] = 5

    return guarded, all_rejected
