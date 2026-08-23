from __future__ import annotations

import math

import pytest

from kicad_mcp.pcb.route_planning import (
    CircleObstacle,
    RectObstacle,
    RouteNotFoundError,
    RoutePoint,
    SegmentObstacle,
    plan_route,
    segment_is_clear,
)


def test_direct_45_route_is_preferred_when_clear() -> None:
    route = plan_route(
        start=RoutePoint(0, 0),
        end=RoutePoint(10, 5),
        obstacles=(),
        grid_mm=0.25,
    )

    assert route.strategy == "direct-45"
    assert route.points == (
        RoutePoint(0, 0),
        RoutePoint(5, 0),
        RoutePoint(10, 5),
    )
    assert route.length_mm == pytest.approx(5 + math.sqrt(50))


def test_a_star_routes_around_inflated_pad_obstacle() -> None:
    obstacle = CircleObstacle(RoutePoint(5, 0), 1.0, "U1.1")
    route = plan_route(
        start=RoutePoint(0, 0),
        end=RoutePoint(10, 0),
        obstacles=(obstacle,),
        grid_mm=0.5,
        bounds=(-1, -4, 11, 4),
    )

    assert route.strategy == "a-star-45"
    assert route.explored_nodes > 0
    assert all(
        segment_is_clear(left, right, (obstacle,))
        for left, right in zip(route.points, route.points[1:], strict=False)
    )


def test_segment_and_rule_area_obstacles_are_honored() -> None:
    obstacles = (
        SegmentObstacle(RoutePoint(2, -1), RoutePoint(2, 1), 0.4, "track"),
        RectObstacle(3, -1, 4, 1, "keepout"),
    )
    route = plan_route(
        start=RoutePoint(0, 0),
        end=RoutePoint(6, 0),
        obstacles=obstacles,
        grid_mm=0.25,
        bounds=(-1, -3, 7, 3),
    )

    assert route.strategy == "a-star-45"
    assert all(
        segment_is_clear(left, right, obstacles)
        for left, right in zip(route.points, route.points[1:], strict=False)
    )


def test_no_route_fails_without_mutating_any_state() -> None:
    wall = RectObstacle(2, -2, 3, 2, "wall")

    with pytest.raises(RouteNotFoundError, match="No clearance-safe route"):
        plan_route(
            start=RoutePoint(0, 0),
            end=RoutePoint(5, 0),
            obstacles=(wall,),
            grid_mm=0.5,
            bounds=(0, -2, 5, 2),
            max_iterations=1_000,
        )


@pytest.mark.parametrize("grid", [0.0, -0.25])
def test_invalid_grid_is_rejected(grid: float) -> None:
    with pytest.raises(ValueError, match="grid_mm"):
        plan_route(
            start=RoutePoint(0, 0),
            end=RoutePoint(1, 1),
            obstacles=(),
            grid_mm=grid,
        )
