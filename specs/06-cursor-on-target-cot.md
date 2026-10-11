# Spec 06: Cursor-on-Target (CoT) & ATAK Integration

## 1. Module Overview & Purpose
To deliver tactical situational awareness without requiring operators or patrol vehicles to stare at a web browser, the Rian Spatial State Engine emits standardized **Cursor-on-Target (CoT) XML / Protobuf** events compatible with the DoD / Public Safety Team Awareness Kit (ATAK, WinTAK, WebTAK, iTAK) and FreeTAKServer.

When an entity is tracked across blind acreage, or a tactical intercept directive is computed, Rian broadcasts:
1. Ground entity markers with NATO MIL-STD-2525 / CoT attitude and dimension (`a-f-G-E-V` for hostile/unknown vehicle, `a-f-G-U-C` for pedestrian).
2. Kinematic velocity vectors, course heading, and Circular Error Probable ($CE$).
3. Dynamic reachability polygons $R(t)$ transmitted as CoT shape / route overlays.
4. Response patrol vehicle intercept vectors directly to tactical client maps over UDP multicast / broadcast or TCP.

---

## 2. Dependencies & Reference Architecture
- **Reference Repo:** `external_references/ATAK_push_cots` (`push_cot.py`, `message.py`, `CotConfig.py`) and `ampledata/pytak`.
- **Contracts:** `specs/00-contracts-and-schemas.md` (`TrackState`, `TacticalDirective`).
- **Target Module Path:** `src/dispatch/cot_emitter.py`.
- **Test Path:** `tests/test_dispatch/test_cot_emitter.py`.

---

## 3. Cursor-on-Target (CoT) Data Contract

### 3.1 CoT Type Taxonomy
| Rian Entity Class | CoT Type Specifier | Description |
| :--- | :--- | :--- |
| `CAR` / `TRUCK` | `a-u-G-E-V` | Atom - Unknown - Ground - Equipment - Vehicle |
| `PERSON` | `a-u-G-U-C` | Atom - Unknown - Ground - Unit - Combatant / Intruder |
| `COW` / `HORSE` / `DOG` | `a-n-G` | Atom - Neutral - Ground |
| `INTERCEPT_DIRECTIVE` | `b-m-p-s-p-i` | Battle Management - Route / Waypoint - Intercept |

### 3.2 XML Envelope Specification (CoT v2.0)
```xml
<event version="2.0" uid="{track_code}" type="{cot_type}" how="m-g"
       time="{iso_utc}" start="{iso_utc}" stale="{iso_utc_plus_stale}">
  <point lat="{lat}" lon="{lon}" hae="0.0" ce="15.0" le="5.0"/>
  <detail>
    <contact callsign="{callsign}"/>
    <track speed="{speed_mps}" course="{heading_deg}"/>
    <remarks>{action_message}</remarks>
  </detail>
</event>
```

---

## 4. Implementation Requirements (`src/dispatch/cot_emitter.py`)

1. **`CotMessageBuilder`**:
   - `build_track_event(track: TrackState, stale_seconds: int = 120) -> str`: Formats track state into verified CoT XML.
   - `build_directive_event(directive: TacticalDirective, stale_seconds: int = 300) -> str`: Formats intercept route and target chokepoint into CoT event.
2. **`CotBroadcaster`**:
   - Asynchronous UDP broadcast on configurable IP (`COT_BROADCAST_IP`, default `239.2.3.1` or `127.0.0.1`) and port (`COT_PORT`, default `6969`).
   - Non-blocking async socket send using `asyncio.DatagramProtocol` or `loop.create_datagram_endpoint`.
   - Toggleable mock/test mode for unit testing without live network interfaces.

---

## 5. Acceptance Criteria
- [ ] `CotMessageBuilder.build_track_event` generates valid CoT v2.0 XML with correct UID, coordinates, and MIL-STD type.
- [ ] Speed in m/s and course heading in degrees correctly populated in `<track>` detail.
- [ ] Stale timestamp strictly enforces expiration window.
- [ ] UDP broadcaster sends packets asynchronously without blocking the FastAPI or tracking loops.
- [ ] 100% test pass rate in `tests/test_dispatch/test_cot_emitter.py`.
