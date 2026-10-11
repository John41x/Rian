"""Entity-targeted LLM diagnostics (spec 15)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.common.domains import (
    NORTH_PASTURE_TEST_LAT,
    NORTH_PASTURE_TEST_LON,
    get_domain_profile,
)
from src.dispatch.llm_copilot import (
    EntityDiagnosticResponse,
    EntityTrackNotFoundError,
    TacticalCopilot,
    build_entity_diagnostic_system_prompt,
    compute_entity_uncertainty_m,
    distance_to_demo_perimeter_m,
)
from src.web.app import app


def _cow_near_perimeter_state() -> dict:
    track_id = "3ad7325a-2fe3-40b3-88bd-8aa084f1d15d"
    return {
        "tracks": [
            {
                "track_id": track_id,
                "track_code": "HERD-COW-04",
                "object_class": "cow",
                "estimated_velocity_ms": 1.4,
                "estimated_heading_deg": 215.0,
                "confidence_score": 0.72,
                "elapsed_seconds": 45,
                "last_known_point": [NORTH_PASTURE_TEST_LON, 36.7579],
            }
        ],
        "chokepoints": [
            {
                "id": "cp-north",
                "name": "North Gate",
                "location": [-119.82, 36.757],
            }
        ],
        "directives": [],
        "sensors": [],
        "reachability": [],
        "interceptor": None,
        "generated_at": "2026-10-10T20:00:00+00:00",
    }


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestUncertaintyRadius:
    def test_increases_with_elapsed_time(self):
        low_t = compute_entity_uncertainty_m(10.0, 0.5)
        high_t = compute_entity_uncertainty_m(400.0, 0.5)
        assert high_t > low_t

    def test_shrinks_with_higher_confidence(self):
        low_conf = compute_entity_uncertainty_m(100.0, 0.2)
        high_conf = compute_entity_uncertainty_m(100.0, 0.95)
        assert high_conf < low_conf


class TestLocalEntityInsight:
    @pytest.mark.asyncio
    async def test_cow_near_perimeter_ranch_profile(self):
        copilot = TacticalCopilot(deepseek_key="", openai_key="", anthropic_key="")
        state = _cow_near_perimeter_state()
        track_id = state["tracks"][0]["track_id"]
        with patch("src.dispatch.llm_copilot.get_domain_profile") as mock_profile:
            mock_profile.return_value = get_domain_profile("ranch_livestock")
            result = await copilot.generate_entity_insight(track_id, state)

        assert isinstance(result, EntityDiagnosticResponse)
        assert result.track_code == "HERD-COW-04"
        assert result.entity_class == "cow"
        assert result.current_zone == "North Pasture"
        assert "ADVISORY" in result.risk_level
        assert result.telemetry["uncertainty_radius_m"] > 0
        assert distance_to_demo_perimeter_m(
            NORTH_PASTURE_TEST_LON, 36.7579
        ) < 120.0
        assert "fence" in result.insight.lower() or "perimeter" in result.insight.lower()
        assert result.provider.startswith("local_entity_reasoner")

    @pytest.mark.asyncio
    async def test_unknown_track_raises(self):
        copilot = TacticalCopilot(deepseek_key="", openai_key="", anthropic_key="")
        with pytest.raises(EntityTrackNotFoundError):
            await copilot.generate_entity_insight(
                "00000000-0000-0000-0000-000000000099",
                _cow_near_perimeter_state(),
            )


class TestDomainPrompts:
    def test_ranch_vs_industrial_system_prompts_differ(self):
        ranch = build_entity_diagnostic_system_prompt(get_domain_profile("ranch_livestock"))
        terminal = build_entity_diagnostic_system_prompt(
            get_domain_profile("industrial_terminal")
        )
        assert "Ranch Manager" in ranch
        assert "Port Facility Security Officer" in terminal
        assert ranch != terminal


class TestEntityInsightEndpoint:
    @pytest.mark.asyncio
    async def test_post_entity_insight_returns_200(self, client):
        state = _cow_near_perimeter_state()
        track_id = state["tracks"][0]["track_id"]
        mock_insight = EntityDiagnosticResponse(
            track_id=track_id,
            track_code="HERD-COW-04",
            entity_class="cow",
            risk_level="ADVISORY // BOUNDARY PROXIMITY",
            current_zone="North Pasture",
            headline="Test headline",
            insight="Test insight for endpoint.",
            recommended_actions=["Action one"],
            telemetry={
                "speed_kmh": 5.0,
                "heading_deg": 215.0,
                "elapsed_seconds": 45,
                "uncertainty_radius_m": 24.5,
            },
            provider="local_entity_reasoner (test)",
        )

        with patch(
            "src.web.app.build_cop_state",
            new=AsyncMock(
                return_value=type(
                    "CopStub",
                    (),
                    {"model_dump": lambda self, mode="json": state},
                )()
            ),
        ):
            with patch.object(
                TacticalCopilot,
                "generate_entity_insight",
                new=AsyncMock(return_value=mock_insight),
            ):
                resp = await client.post(f"/api/v1/cop/entity/{track_id}/insight")

        assert resp.status_code == 200
        body = resp.json()
        assert body["track_id"] == track_id
        assert body["track_code"] == "HERD-COW-04"
        assert body["telemetry"]["uncertainty_radius_m"] == 24.5

    @pytest.mark.asyncio
    async def test_unknown_track_returns_404(self, client):
        state = _cow_near_perimeter_state()
        missing = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        with patch(
            "src.web.app.build_cop_state",
            new=AsyncMock(
                return_value=type(
                    "CopStub",
                    (),
                    {"model_dump": lambda self, mode="json": state},
                )()
            ),
        ):
            resp = await client.post(f"/api/v1/cop/entity/{missing}/insight")

        assert resp.status_code == 404
