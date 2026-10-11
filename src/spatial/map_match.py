"""
HMM map-matching and road snapping on property_network_edges via PostGIS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import networkx as nx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.common.database import async_session_factory
from src.spatial.routing import build_networkx_from_edges, _haversine_m

logger = logging.getLogger(__name__)

EdgesTable = "property_network_edges"
VerticesTable = "property_network_edges_vertices_pgr"

SessionFactory = async_sessionmaker[AsyncSession]


@dataclass(frozen=True)
class CandidateEdge:
    edge_id: int
    source: int
    target: int
    distance_m: float
    snapped_point: Tuple[float, float]


def _default_offline_edge_records() -> List[dict[str, Any]]:
    """Synthetic ranch corridor for offline tests and DB fallback."""
    return [
        {
            "id": 1,
            "source": 1,
            "target": 2,
            "lon_lat_path": [(-119.82, 36.74), (-119.819, 36.741)],
        },
        {
            "id": 2,
            "source": 2,
            "target": 3,
            "lon_lat_path": [(-119.819, 36.741), (-119.818, 36.742)],
        },
    ]


def _closest_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> Tuple[float, float, float]:
    """Project (px, py) onto segment AB; return (lon, lat, distance_m)."""
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return ax, ay, _haversine_m(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return cx, cy, _haversine_m(px, py, cx, cy)


def _offline_candidate_edges(
    lon: float,
    lat: float,
    radius_m: float,
    edge_records: Sequence[dict[str, Any]],
) -> List[CandidateEdge]:
    candidates: List[CandidateEdge] = []
    for rec in edge_records:
        path = rec["lon_lat_path"]
        best_dist = float("inf")
        best_pt = (lon, lat)
        for a, b in zip(path[:-1], path[1:]):
            cx, cy, dist = _closest_on_segment(lon, lat, a[0], a[1], b[0], b[1])
            if dist < best_dist:
                best_dist = dist
                best_pt = (cx, cy)
        if best_dist <= radius_m:
            candidates.append(
                CandidateEdge(
                    edge_id=int(rec["id"]),
                    source=int(rec["source"]),
                    target=int(rec["target"]),
                    distance_m=best_dist,
                    snapped_point=best_pt,
                )
            )
    candidates.sort(key=lambda c: c.distance_m)
    return candidates


def _build_offline_graph(edge_records: Sequence[dict[str, Any]]) -> nx.Graph:
    edges: List[dict[str, Any]] = []
    for rec in edge_records:
        path = rec["lon_lat_path"]
        length_m = 0.0
        for a, b in zip(path[:-1], path[1:]):
            length_m += _haversine_m(a[0], a[1], b[0], b[1])
        edges.append(
            {
                "source": rec["source"],
                "target": rec["target"],
                "cost": max(length_m, 1.0),
                "lon_lat_path": path,
            }
        )
    return build_networkx_from_edges(edges)


def _offline_network_distance_m(
    graph: nx.Graph,
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    if not graph.nodes:
        return _haversine_m(lon1, lat1, lon2, lat2)

    def nearest_node(lon: float, lat: float) -> int:
        best: Optional[int] = None
        best_d = float("inf")
        for u, v, data in graph.edges(data=True):
            path = data.get("geom") or []
            for pt in path:
                d = _haversine_m(lon, lat, pt[0], pt[1])
                if d < best_d:
                    best_d = d
                    best = int(u)
            for node_id, pt in ((int(u), path[0]), (int(v), path[-1])):
                if path:
                    d = _haversine_m(lon, lat, pt[0], pt[1])
                    if d < best_d:
                        best_d = d
                        best = node_id
        return best if best is not None else int(next(iter(graph.nodes)))

    start = nearest_node(lon1, lat1)
    end = nearest_node(lon2, lat2)
    try:
        path = nx.dijkstra_path(graph, start, end, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return _haversine_m(lon1, lat1, lon2, lat2)

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        segment = graph[u][v].get("geom")
        if segment and len(segment) >= 2:
            for a, b in zip(segment[:-1], segment[1:]):
                total += _haversine_m(a[0], a[1], b[0], b[1])
        else:
            total += graph[u][v].get("weight", _haversine_m(float(u), float(u), float(v), float(v)))
    return total


class RoadMapMatcher:
    """PostGIS-backed HMM road snapping with offline geometric fallback."""

    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        offline_edge_records: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._session_factory = session_factory or async_session_factory
        self._offline_edges = list(offline_edge_records or _default_offline_edge_records())
        self._offline_graph = _build_offline_graph(self._offline_edges)

    @staticmethod
    def calculate_emission_probability(distance_m: float, sigma_z: float = 20.0) -> float:
        """Gaussian likelihood of observation given perpendicular distance to edge."""
        if sigma_z <= 0:
            raise ValueError("sigma_z must be positive")
        coef = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_z)
        return coef * math.exp(-(distance_m ** 2) / (2.0 * sigma_z ** 2))

    @staticmethod
    def calculate_transition_probability(
        network_dist_m: float,
        euclidean_dist_m: float,
        beta: float = 30.0,
    ) -> float:
        """Exponential likelihood of route length discrepancy (network vs straight-line)."""
        if beta <= 0:
            raise ValueError("beta must be positive")
        discrepancy = abs(network_dist_m - euclidean_dist_m)
        return (1.0 / beta) * math.exp(-discrepancy / beta)

    async def find_candidate_edges(
        self,
        lon: float,
        lat: float,
        radius_m: float = 150.0,
    ) -> List[CandidateEdge]:
        sql = text(
            f"""
            SELECT
                e.id AS edge_id,
                e.source,
                e.target,
                ST_Distance(
                    ST_Transform(e.geom, 3857),
                    ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 3857)
                ) AS distance_m,
                ST_X(ST_ClosestPoint(e.geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))) AS snapped_lon,
                ST_Y(ST_ClosestPoint(e.geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))) AS snapped_lat
            FROM {EdgesTable} e
            WHERE ST_DWithin(
                ST_Transform(e.geom, 3857),
                ST_Transform(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 3857),
                :radius_m
            )
            ORDER BY distance_m ASC
            """
        )
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    sql,
                    {"lon": lon, "lat": lat, "radius_m": radius_m},
                )
                rows = result.fetchall()
                if not rows:
                    return _offline_candidate_edges(lon, lat, radius_m, self._offline_edges)
                return [
                    CandidateEdge(
                        edge_id=int(row.edge_id),
                        source=int(row.source),
                        target=int(row.target),
                        distance_m=float(row.distance_m),
                        snapped_point=(float(row.snapped_lon), float(row.snapped_lat)),
                    )
                    for row in rows
                ]
        except Exception as exc:
            logger.info("Candidate edge query failed (%s); using offline fallback", exc)
            return _offline_candidate_edges(lon, lat, radius_m, self._offline_edges)

    async def _network_distance_m(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> float:
        sql = text(
            f"""
            WITH start_v AS (
                SELECT id
                FROM {VerticesTable}
                ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(:lon1, :lat1), 4326)
                LIMIT 1
            ),
            end_v AS (
                SELECT id
                FROM {VerticesTable}
                ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(:lon2, :lat2), 4326)
                LIMIT 1
            ),
            route AS (
                SELECT edge
                FROM pgr_dijkstra(
                    'SELECT id, source, target, cost, reverse_cost FROM {EdgesTable}',
                    (SELECT id FROM start_v),
                    (SELECT id FROM end_v),
                    directed := false
                )
                WHERE edge <> -1
            )
            SELECT COALESCE(
                SUM(ST_Length(ST_Transform(e.geom, 3857))),
                0
            ) AS distance_m
            FROM route r
            JOIN {EdgesTable} e ON e.id = r.edge
            """
        )
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    sql,
                    {"lon1": lon1, "lat1": lat1, "lon2": lon2, "lat2": lat2},
                )
                row = result.first()
                if row and row.distance_m is not None and float(row.distance_m) > 0:
                    return float(row.distance_m)
        except Exception as exc:
            logger.debug("Network distance query failed: %s", exc)
        return _offline_network_distance_m(self._offline_graph, lon1, lat1, lon2, lat2)

    async def snap_point(
        self,
        lon: float,
        lat: float,
        max_radius_m: float = 150.0,
    ) -> Tuple[float, float, Optional[int]]:
        candidates = await self.find_candidate_edges(lon, lat, radius_m=max_radius_m)
        if not candidates:
            return lon, lat, None

        best = max(
            candidates,
            key=lambda c: self.calculate_emission_probability(c.distance_m),
        )
        if best.distance_m > max_radius_m:
            return lon, lat, None
        return best.snapped_point[0], best.snapped_point[1], best.edge_id

    async def match_trajectory(
        self,
        points: List[Tuple[float, float]],
        radius_m: float = 150.0,
        sigma_z: float = 20.0,
        beta: float = 30.0,
    ) -> List[Tuple[float, float]]:
        """Viterbi HMM decode: snap a GPS sequence to the road network."""
        if not points:
            return []

        if len(points) == 1:
            slon, slat, _ = await self.snap_point(points[0][0], points[0][1], max_radius_m=radius_m)
            return [(slon, slat)]

        candidate_lists: List[List[CandidateEdge]] = []
        for lon, lat in points:
            candidate_lists.append(
                await self.find_candidate_edges(lon, lat, radius_m=radius_m)
            )

        n = len(points)
        log_v: List[List[float]] = []
        backptr: List[List[Optional[int]]] = []

        for t in range(n):
            cands = candidate_lists[t]
            if not cands:
                log_v.append([])
                backptr.append([])
                continue

            emissions = [
                self.calculate_emission_probability(c.distance_m, sigma_z=sigma_z) for c in cands
            ]
            log_em = [math.log(max(e, 1e-300)) for e in emissions]

            if t == 0:
                log_v.append(log_em)
                backptr.append([None] * len(cands))
                continue

            prev_cands = candidate_lists[t - 1]
            prev_log = log_v[t - 1]
            curr_log: List[float] = []
            curr_back: List[Optional[int]] = []

            if not prev_cands or not prev_log:
                log_v.append(log_em)
                backptr.append([None] * len(cands))
                continue

            plon, plat = points[t - 1][0], points[t - 1][1]
            clon, clat = points[t][0], points[t][1]
            eucl = _haversine_m(plon, plat, clon, clat)

            for j, cand_j in enumerate(cands):
                best_score = float("-inf")
                best_i: Optional[int] = None
                for i, cand_i in enumerate(prev_cands):
                    si = cand_i.snapped_point
                    sj = cand_j.snapped_point
                    net_dist = await self._network_distance_m(si[0], si[1], sj[0], sj[1])
                    trans = self.calculate_transition_probability(net_dist, eucl, beta=beta)
                    score = prev_log[i] + math.log(max(trans, 1e-300)) + log_em[j]
                    if score > best_score:
                        best_score = score
                        best_i = i
                curr_log.append(best_score)
                curr_back.append(best_i)

            log_v.append(curr_log)
            backptr.append(curr_back)

        snapped: List[Tuple[float, float]] = []
        last_t = n - 1
        if not candidate_lists[last_t] or not log_v[last_t]:
            for lon, lat in points:
                slon, slat, _ = await self.snap_point(lon, lat, max_radius_m=radius_m)
                snapped.append((slon, slat))
            return snapped

        j = max(range(len(candidate_lists[last_t])), key=lambda idx: log_v[last_t][idx])
        path_indices: List[Optional[int]] = [None] * n
        path_indices[last_t] = j
        for t in range(last_t, 0, -1):
            if path_indices[t] is None:
                break
            prev_j = backptr[t][path_indices[t]] if backptr[t] else None
            path_indices[t - 1] = prev_j

        for t, (lon, lat) in enumerate(points):
            cands = candidate_lists[t]
            if not cands:
                snapped.append((lon, lat))
                continue
            idx = path_indices[t]
            if idx is None:
                slon, slat, _ = await self.snap_point(lon, lat, max_radius_m=radius_m)
                snapped.append((slon, slat))
            else:
                snapped.append(cands[idx].snapped_point)
        return snapped
