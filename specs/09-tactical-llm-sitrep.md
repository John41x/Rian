# Spec 09: Tactical LLM Intelligence & Verified SITREP Engine

## 1. Module Overview & Purpose
Operating on the **Cold Path (Asynchronous Supervisor)**, the **Tactical LLM Intelligence Engine** ingests verified spatial state vectors from the Hot Path and translates them into actionable military-grade **Situation Reports (SITREPs)**, forensic incident analysis, and natural language C2 responses for field commanders.

To prevent hallucinations during live tactical emergencies, this module adapts:
1. **Deterministic Input/Output Guards** from **`dj9889/sitrep`** (`guards.ts`): Strict parameter clipping, range fencing, and verified Pydantic schema validation.
2. **Tactical C2 Persona Engineering** from **`raytheonbbn/tak-gpt`**: Authoritative, concise tactical voice adhering to military communication standards.
3. **DeepSeek-V3 / DeepSeek-R1 API Client**: Powered by the user's provided DeepSeek API key, using `https://api.deepseek.com`.
4. **Natural Language Spatial Tools** (inspired by `boettiger-lab/ca-30x30`): Function calling allowing operators to query PostGIS tables using natural language (e.g. *"Show all vehicle breaches in the last 4 hours"*).

---

## 2. Dependencies & Reference Architecture
- **Reference Repos:** `external_references/sitrep` (`src/main/guards.ts`), `external_references/tak-gpt`, `external_references/blackbox_node`.
- **Target Module Path:** `src/dispatch/llm_copilot.py` (enhanced).
- **Test Path:** `tests/test_dispatch/test_llm_copilot.py`.

---

## 3. Strict Schema Guardrails & Evidence Verification

### 3.1 Pydantic Output Contract (`TacticalSitrepReport`)
Every generated SITREP must strictly validate against:
```python
class TacticalSitrepReport(BaseModel):
    timestamp_utc: str
    defcon_level: int = Field(..., ge=1, le=5)
    threat_status: str  # e.g. "RED // ACTIVE BREACH"
    primary_objective: str  # e.g. "Intercept at South Gate"
    tactical_margin_minutes: float
    target_summary: str
    recommended_actions: List[str]
    fenced_evidence: Dict[str, Any]  # Unaltered telemetry used as ground truth
    engine_provider: str
```

### 3.2 Fencing & Hallucination Prevention (`guards.py`)
- Coordinates, speeds, and timestamps must be copied directly from the input spatial vector.
- Any attempt by the model to invent non-existent sensor names, cameras, or chokepoints is blocked by cross-referencing `sensor_nodes` and `strategic_chokepoints` in DB.

---

## 4. Implementation Requirements (`src/dispatch/llm_copilot.py`)

1. **DeepSeek API Provider**:
   - Implements `call_deepseek_chat(messages, temperature=0.2, tools=None)` using `httpx.AsyncClient`.
   - Uses `model: deepseek-chat` (DeepSeek-V3) or `deepseek-reasoner` (DeepSeek-R1).
   - Base URL: `https://api.deepseek.com/chat/completions`.
2. **Function Calling Tools**:
   - `get_active_tracks_tool`: Returns verified active track codes, locations, velocities.
   - `get_chokepoint_eta_tool`: Returns arrival differentials and safety margins.
   - `get_camera_health_tool`: Returns online status, battery levels, and FOV azimuths.
3. **Deterministic Fallback**:
   - If DeepSeek API is unreachable or network is offline, the local heuristic reasoner seamlessly generates valid SITREPs without crashing.

---

## 5. Acceptance Criteria
- [ ] Connects and authenticates successfully with DeepSeek API using `DEEPSEEK_API_KEY`.
- [ ] Generates valid `TacticalSitrepReport` matching Pydantic schema without JSON decode errors.
- [ ] Evidence guard blocks any hallucinated chokepoint or sensor names.
- [ ] Natural language tool execution answers queries with ground-truth PostGIS telemetry.
- [ ] 100% test pass rate in `tests/test_dispatch/test_llm_copilot.py`.
