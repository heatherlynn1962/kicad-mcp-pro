"""Deterministic, clearance-aware planning for one PCB connection.

This module deliberately has no KiCad or FastMCP imports.  Live-board adapters
translate KiCad objects into the small geometry contract below, which keeps the
planner fast to test and safe to run as a read-only preview.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count


@dataclass(frozen=True, slots=True)
class RoutePoint:
    x_mm: float
    y_mm: float

    def rounded(self) -> RoutePoint:
        return RoutePoint(round(self.x_mm, 6), round(self.y_mm, 6))


@dataclass(frozen=True, slots=True)
class CircleObstacle:
    center: RoutePoint
    radius_mm: float
    label: str = ""


@dataclass(frozen=True, slots=True)
class SegmentObstacle:
    start: RoutePoint
    end: RoutePoint
    radius_mm: float
    label: str = ""


@dataclass(frozen=True, slots=True)
class RectObstacle:
    min_x_mm: float
    min_y_mm: float
    max_x_mm: float
    max_y_mm: float
    label: str = ""


type RouteObstacle = CircleObstacle | SegmentObstacle | RectObstacle
type RouteBounds = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PlannedRoute:
    points: tuple[RoutePoint, ...]
    length_mm: float
    explored_nodes: int
    strategy: str


class RouteNotFoundError(ValueError):
    """Raised when no clearance-safe single-layer path is available."""


def _point_segment_distance(point: RoutePoint, start: RoutePoint, end: RoutePoint) -> float:
    dx = end.x_mm - start.x_mm
    dy = end.y_mm - start.y_mm
    if dx == 0 and dy == 0:
        return math.hypot(point.x_mm - start.x_mm, point.y_mm - start.y_mm)
    ratio = ((point.x_mm - start.x_mm) * dx + (point.y_mm - start.y_mm) * dy) / (dx * dx + dy * dy)
    ratio = max(0.0, min(1.0, ratio))
    closest = RoutePoint(start.x_mm + ratio * dx, start.y_mm + ratio * dy)
    return math.hypot(point.x_mm - closest.x_mm, point.y_mm - closest.y_mm)


def _orientation(a: RoutePoint, b: RoutePoint, c: RoutePoint) -> float:
    return (b.x_mm - a.x_mm) * (c.y_mm - a.y_mm) - (b.y_mm - a.y_mm) * (c.x_mm - a.x_mm)


def _segments_intersect(a: RoutePoint, b: RoutePoint, c: RoutePoint, d: RoutePoint) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    epsilon = 1e-9
    if abs(o1) <= epsilon and _point_segment_distance(c, a, b) <= epsilon:
        return True
    if abs(o2) <= epsilon and _point_segment_distance(d, a, b) <= epsilon:
        return True
    if abs(o3) <= epsilon and _point_segment_distance(a, c, d) <= epsilon:
        return True
    if abs(o4) <= epsilon and _point_segment_distance(b, c, d) <= epsilon:
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _segment_distance(a: RoutePoint, b: RoutePoint, c: RoutePoint, d: RoutePoint) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _point_in_rect(point: RoutePoint, rect: RectObstacle) -> bool:
    return (
        rect.min_x_mm <= point.x_mm <= rect.max_x_mm
        and rect.min_y_mm <= point.y_mm <= rect.max_y_mm
    )


def _segment_hits_rect(start: RoutePoint, end: RoutePoint, rect: RectObstacle) -> bool:
    if _point_in_rect(start, rect) or _point_in_rect(end, rect):
        return True
    corners = (
        RoutePoint(rect.min_x_mm, rect.min_y_mm),
        RoutePoint(rect.max_x_mm, rect.min_y_mm),
        RoutePoint(rect.max_x_mm, rect.max_y_mm),
        RoutePoint(rect.min_x_mm, rect.max_y_mm),
    )
    return any(
        _segments_intersect(start, end, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )


def segment_is_clear(
    start: RoutePoint,
    end: RoutePoint,
    obstacles: tuple[RouteObstacle, ...],
) -> bool:
    """Return whether a candidate centerline clears every inflated obstacle."""
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            if _point_segment_distance(obstacle.center, start, end) <= obstacle.radius_mm:
                return False
        elif isinstance(obstacle, SegmentObstacle):
            if _segment_distance(start, end, obstacle.start, obstacle.end) <= obstacle.radius_mm:
                return False
        elif _segment_hits_rect(start, end, obstacle):
            return False
    return True


def _route_length(points: tuple[RoutePoint, ...]) -> float:
    return sum(
        math.hypot(right.x_mm - left.x_mm, right.y_mm - left.y_mm)
        for left, right in zip(points, points[1:], strict=False)
    )


def _simplify(points: list[RoutePoint]) -> tuple[RoutePoint, ...]:
    if len(points) < 3:
        return tuple(point.rounded() for point in points)
    simplified = [points[0]]
    for point, following in zip(points[1:-1], points[2:], strict=False):
        previous = simplified[-1]
        if abs(_orientation(previous, point, following)) > 1e-9:
            simplified.append(point)
    simplified.append(points[-1])
    return tuple(point.rounded() for point in simplified)


def _dogleg_candidates(start: RoutePoint, end: RoutePoint) -> tuple[tuple[RoutePoint, ...], ...]:
    dx = end.x_mm - start.x_mm
    dy = end.y_mm - start.y_mm
    ax = abs(dx)
    ay = abs(dy)
    sx = 1.0 if dx >= 0 else -1.0
    sy = 1.0 if dy >= 0 else -1.0
    if math.isclose(ax, 0.0) or math.isclose(ay, 0.0) or math.isclose(ax, ay):
        return ((start, end),)
    if ax > ay:
        return (
            (start, RoutePoint(start.x_mm + sx * (ax - ay), start.y_mm), end),
            (start, RoutePoint(start.x_mm + sx * ay, end.y_mm), end),
        )
    return (
        (start, RoutePoint(start.x_mm, start.y_mm + sy * (ay - ax)), end),
        (start, RoutePoint(end.x_mm, start.y_mm + sy * ax), end),
    )


def _path_is_clear(points: tuple[RoutePoint, ...], obstacles: tuple[RouteObstacle, ...]) -> bool:
    return all(
        segment_is_clear(start, end, obstacles)
        for start, end in zip(points, points[1:], strict=False)
    )


def _default_bounds(
    start: RoutePoint,
    end: RoutePoint,
    obstacles: tuple[RouteObstacle, ...],
    margin_mm: float,
) -> RouteBounds:
    xs = [start.x_mm, end.x_mm]
    ys = [start.y_mm, end.y_mm]
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            xs.extend(
                [
                    obstacle.center.x_mm - obstacle.radius_mm,
                    obstacle.center.x_mm + obstacle.radius_mm,
                ]
            )
            ys.extend(
                [
                    obstacle.center.y_mm - obstacle.radius_mm,
                    obstacle.center.y_mm + obstacle.radius_mm,
                ]
            )
        elif isinstance(obstacle, SegmentObstacle):
            xs.extend([obstacle.start.x_mm, obstacle.end.x_mm])
            ys.extend([obstacle.start.y_mm, obstacle.end.y_mm])
        else:
            xs.extend([obstacle.min_x_mm, obstacle.max_x_mm])
            ys.extend([obstacle.min_y_mm, obstacle.max_y_mm])
    return (
        min(xs) - margin_mm,
        min(ys) - margin_mm,
        max(xs) + margin_mm,
        max(ys) + margin_mm,
    )


def plan_route(
    *,
    start: RoutePoint,
    end: RoutePoint,
    obstacles: tuple[RouteObstacle, ...],
    grid_mm: float = 0.25,
    bounds: RouteBounds | None = None,
    margin_mm: float = 3.0,
    max_iterations: int = 50_000,
) -> PlannedRoute:
    """Plan a 45-degree, single-layer route through inflated obstacles."""
    if grid_mm <= 0:
        raise ValueError("grid_mm must be greater than zero.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")
    for candidate in _dogleg_candidates(start, end):
        if _path_is_clear(candidate, obstacles):
            points = _simplify(list(candidate))
            return PlannedRoute(points, _route_length(points), 0, "direct-45")

    active_bounds = bounds or _default_bounds(start, end, obstacles, margin_mm)
    min_x, min_y, max_x, max_y = active_bounds

    def to_grid(point: RoutePoint) -> tuple[int, int]:
        return (round(point.x_mm / grid_mm), round(point.y_mm / grid_mm))

    def from_grid(node: tuple[int, int]) -> RoutePoint:
        return RoutePoint(node[0] * grid_mm, node[1] * grid_mm)

    min_gx = math.ceil(min_x / grid_mm)
    min_gy = math.ceil(min_y / grid_mm)
    max_gx = math.floor(max_x / grid_mm)
    max_gy = math.floor(max_y / grid_mm)
    start_node = to_grid(start)
    end_node = to_grid(end)
    if not (min_gx <= start_node[0] <= max_gx and min_gy <= start_node[1] <= max_gy):
        raise RouteNotFoundError("The start point is outside the available routing bounds.")
    if not (min_gx <= end_node[0] <= max_gx and min_gy <= end_node[1] <= max_gy):
        raise RouteNotFoundError("The end point is outside the available routing bounds.")

    sequence = count()
    frontier: list[tuple[float, int, tuple[int, int]]] = [(0.0, next(sequence), start_node)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    cost: dict[tuple[int, int], float] = {start_node: 0.0}
    directions = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    explored = 0
    while frontier and explored < max_iterations:
        _, _, current = heapq.heappop(frontier)
        explored += 1
        if current == end_node:
            break
        current_point = from_grid(current)
        for dx, dy, step_cost in directions:
            following = (current[0] + dx, current[1] + dy)
            if not (min_gx <= following[0] <= max_gx and min_gy <= following[1] <= max_gy):
                continue
            following_point = from_grid(following)
            if not segment_is_clear(current_point, following_point, obstacles):
                continue
            new_cost = cost[current] + step_cost
            if new_cost >= cost.get(following, math.inf):
                continue
            cost[following] = new_cost
            came_from[following] = current
            delta_x = abs(end_node[0] - following[0])
            delta_y = abs(end_node[1] - following[1])
            heuristic = max(delta_x, delta_y) + (math.sqrt(2.0) - 1.0) * min(delta_x, delta_y)
            heapq.heappush(frontier, (new_cost + heuristic, next(sequence), following))

    if end_node not in came_from:
        raise RouteNotFoundError(
            f"No clearance-safe route was found after exploring {explored} grid nodes."
        )

    nodes: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = end_node
    while cursor is not None:
        nodes.append(cursor)
        cursor = came_from[cursor]
    nodes.reverse()
    grid_points = [from_grid(node) for node in nodes]
    grid_points[0] = start
    grid_points[-1] = end
    simplified = _simplify(grid_points)
    if not _path_is_clear(simplified, obstacles):
        raise RouteNotFoundError("The simplified route failed final clearance validation.")
    return PlannedRoute(simplified, _route_length(simplified), explored, "a-star-45")
