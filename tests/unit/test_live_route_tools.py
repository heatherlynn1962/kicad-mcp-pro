from __future__ import annotations

from types import SimpleNamespace

import pytest
from kipy.board_types import Net
from mcp.server.fastmcp import FastMCP

from kicad_mcp.tools import routing
from tests.conftest import call_tool_payload


class _Position:
    def __init__(self, x_mm: float, y_mm: float) -> None:
        self.x_nm = int(x_mm * 1_000_000)
        self.y_nm = int(y_mm * 1_000_000)


class _Pad:
    def __init__(self, reference: str, number: str, x_mm: float, y_mm: float, net: Net) -> None:
        self.id = SimpleNamespace(value=f"{reference}-{number}")
        self.number = number
        self.position = _Position(x_mm, y_mm)
        self.net = net
        self.padstack = SimpleNamespace(copper_layers=[])


class _FakeBoard:
    def __init__(self) -> None:
        self.net = Net()
        self.net.name = "CAN1H"
        self.pads = [
            _Pad("J10", "1", 0, 0, self.net),
            _Pad("U11", "7", 10, 0, self.net),
        ]
        self.footprints = [
            SimpleNamespace(
                reference_field=SimpleNamespace(text=SimpleNamespace(value=reference)),
                definition=SimpleNamespace(
                    pads=[pad for pad in self.pads if pad.id.value.startswith(f"{reference}-")]
                ),
            )
            for reference in ("J10", "U11")
        ]
        self.selected: list[object] = []
        self.state = "before"
        self.effects: list[object] = []
        self.created_count = 0

    def get_pads(self) -> list[object]:
        return self.pads

    def get_footprints(self) -> list[object]:
        return self.footprints

    def get_tracks(self) -> list[object]:
        return []

    def get_vias(self) -> list[object]:
        return []

    def get_zones(self) -> list[object]:
        return []

    def get_shapes(self) -> list[object]:
        return []

    def get_selection(self) -> list[object]:
        return self.selected

    def get_netclass_for_nets(self, _net: Net) -> dict[str, object]:
        return {
            "CAN1H": SimpleNamespace(
                track_width=300_000,
                clearance=200_000,
            )
        }

    def get_as_string(self) -> str:
        return self.state

    def begin_commit(self) -> object:
        commit = object()
        self.effects.append(("begin", commit))
        return commit

    def create_items(self, tracks: list[object]) -> list[object]:
        self.effects.append(("create", len(tracks)))
        created = []
        for _track in tracks:
            self.created_count += 1
            created.append(SimpleNamespace(id=SimpleNamespace(value=f"route-{self.created_count}")))
        return created

    def push_commit(self, commit: object, message: str = "") -> None:
        self.effects.append(("push", commit, message))
        self.state += ":changed"

    def drop_commit(self, commit: object) -> None:
        self.effects.append(("drop", commit))

    def remove_items_by_id(self, ids: list[object]) -> None:
        self.effects.append(("remove", tuple(item.value for item in ids)))


@pytest.fixture
def live_route_server(monkeypatch: pytest.MonkeyPatch) -> tuple[FastMCP, _FakeBoard]:
    board = _FakeBoard()
    monkeypatch.setattr(routing, "get_board", lambda: board)
    monkeypatch.setattr(
        routing,
        "_run_queued_ipc_mutation",
        lambda _operation, command: command(),
    )
    routing._ROUTE_PLANS.clear()
    routing._ROUTE_RECEIPTS.clear()
    server = FastMCP("route-test")
    routing.register(server)
    return server, board


@pytest.mark.anyio
async def test_plan_apply_and_exact_revert_are_one_commit_each(
    live_route_server: tuple[FastMCP, _FakeBoard],
) -> None:
    server, board = live_route_server
    plan = await call_tool_payload(
        server,
        "pcb_plan_route",
        {"ref1": "J10", "pad1": "1", "ref2": "U11", "pad2": "7"},
    )
    assert isinstance(plan, dict)
    assert plan["dry_run"] is True
    route_plan = plan["extra"]["route_plan"]
    assert route_plan["layer"] == "F.Cu"
    assert route_plan["width_mm"] == pytest.approx(0.3)
    assert route_plan["clearance_mm"] == pytest.approx(0.2)

    plan_id = route_plan["plan_id"]
    applied = await call_tool_payload(server, "pcb_apply_route_plan", {"plan_id": plan_id})
    assert isinstance(applied, dict)
    assert applied["ok"] is True
    assert applied["rollback_token"] == plan_id
    assert applied["extra"]["created_item_ids"] == ["route-1"]
    assert [effect[0] for effect in board.effects] == ["begin", "create", "push"]

    duplicate = await call_tool_payload(server, "pcb_apply_route_plan", {"plan_id": plan_id})
    assert isinstance(duplicate, dict)
    assert duplicate["ok"] is False
    assert "already applied" in duplicate["errors"][0]

    reverted = await call_tool_payload(server, "pcb_revert_route_plan", {"plan_id": plan_id})
    assert isinstance(reverted, dict)
    assert reverted["ok"] is True
    assert reverted["extra"]["removed_item_ids"] == ["route-1"]
    assert [effect[0] for effect in board.effects[-3:]] == ["begin", "remove", "push"]


@pytest.mark.anyio
async def test_stale_plan_is_rejected_before_any_mutation(
    live_route_server: tuple[FastMCP, _FakeBoard],
) -> None:
    server, board = live_route_server
    plan = await call_tool_payload(
        server,
        "pcb_plan_route",
        {"ref1": "J10", "pad1": "1", "ref2": "U11", "pad2": "7"},
    )
    assert isinstance(plan, dict)
    plan_id = plan["extra"]["route_plan"]["plan_id"]
    board.state = "user-edited-board"

    applied = await call_tool_payload(server, "pcb_apply_route_plan", {"plan_id": plan_id})

    assert isinstance(applied, dict)
    assert applied["ok"] is False
    assert "changed after this route was planned" in applied["errors"][0]
    assert board.effects == []


@pytest.mark.anyio
async def test_two_selected_pads_can_define_the_route(
    live_route_server: tuple[FastMCP, _FakeBoard],
) -> None:
    server, board = live_route_server
    board.selected = list(board.pads)

    plan = await call_tool_payload(server, "pcb_plan_route", {})

    assert isinstance(plan, dict)
    assert plan["ok"] is True
    assert plan["extra"]["route_plan"]["source"] == "J10.1"
    assert plan["extra"]["route_plan"]["target"] == "U11.7"
