import math

import networkx as nx
import pytest

from src.spatial.routing import (
    build_networkx_from_edges,
    find_shortest_route,
    solve_shortest_path_networkx,
)


@pytest.fixture
def simple_road_graph() -> nx.Graph:
    return build_networkx_from_edges(
        [
            {"source": 1, "target": 2, "cost": 10.0, "lon_lat_path": [(-119.82, 36.74), (-119.819, 36.741)]},
            {"source": 2, "target": 3, "cost": 20.0, "lon_lat_path": [(-119.819, 36.741), (-119.818, 36.742)]},
            {
                "source": 1,
                "target": 3,
                "cost": 100.0,
                "lon_lat_path": [(-119.82, 36.74), (-119.818, 36.742)],
            },
        ]
    )


class TestNetworkXRouting:
    def test_shortest_path_prefers_lower_cost(self, simple_road_graph: nx.Graph):
        result = solve_shortest_path_networkx(simple_road_graph, 1, 3)
        assert math.isclose(result.transit_time_s, 30.0)
        assert result.used_fallback is True
        assert len(result.coordinates) >= 2
        assert result.coordinates[0][0] == pytest.approx(-119.82)

    def test_path_geometry_follows_edges(self, simple_road_graph: nx.Graph):
        result = solve_shortest_path_networkx(simple_road_graph, 1, 3)
        assert result.coordinates[-1] == (-119.818, 36.742)


@pytest.mark.asyncio
async def test_find_shortest_route_fallback_when_db_empty():
    result = await find_shortest_route(
        (-119.82, 36.74),
        (-119.818, 36.742),
        travel_mode="vehicle",
    )
    assert result.used_fallback is True
    assert math.isclose(result.transit_time_s, 30.0)
    assert len(result.coordinates) >= 2
