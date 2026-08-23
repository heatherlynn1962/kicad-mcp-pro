from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from kicad_mcp.pcb.board_initialization import (
    BoardInitializationProfile,
    ManufacturingMinimums,
    NetClassPattern,
    NetClassProfile,
    StackupExpectation,
    ViaPreset,
    apply_profile,
    capture_profile,
    dump_profile,
    load_profile,
    render_ai_directives,
    render_ai_notes,
)
from kicad_mcp.tools import board_initialization
from tests.conftest import call_tool_payload, call_tool_text


def _project_payload() -> dict[str, object]:
    return {
        "board": {
            "design_settings": {
                "rules": {
                    "min_track_width": 0.2,
                    "min_clearance": 0.2,
                    "min_via_diameter": 0.5,
                    "min_via_drill": 0.3,
                },
                "track_widths": [0.0, 0.2, 0.3],
                "via_dimensions": [{"diameter": 0.0, "drill": 0.0}],
                "preserved_setting": {"enabled": True},
            }
        },
        "meta": {"filename": "demo.kicad_pro", "version": 1},
        "net_settings": {
            "classes": [
                {
                    "name": "Default",
                    "track_width": 0.2,
                    "clearance": 0.2,
                    "via_diameter": 0.5,
                    "via_drill": 0.3,
                    "priority": 2_147_483_647,
                },
                {
                    "name": "POWER",
                    "track_width": 0.75,
                    "clearance": 0.2,
                    "via_diameter": 0.8,
                    "via_drill": 0.4,
                    "priority": 0,
                },
            ],
            "netclass_patterns": [{"netclass": "POWER", "pattern": "VCC"}],
            "unknown_net_setting": "preserve-me",
        },
        "unrelated": {"keep": "yes"},
    }


def _write_project(project_dir: Path) -> tuple[Path, Path]:
    project_file = project_dir / "demo.kicad_pro"
    project_file.write_text(json.dumps(_project_payload(), indent=2), encoding="utf-8")
    board_file = project_dir / "demo.kicad_pcb"
    board_file.write_text(
        """(kicad_pcb
  (layers
    (0 "F.Cu" signal)
    (2 "In1.Cu" power)
    (31 "B.Cu" signal)
  )
  (net 0 "")
  (net 1 "VCC")
  (net 2 "GND")
)\n""",
        encoding="utf-8",
    )
    return project_file, board_file


def _desired_profile() -> BoardInitializationProfile:
    return BoardInitializationProfile(
        manufacturing=ManufacturingMinimums(
            min_track_width_mm=0.25,
            min_clearance_mm=0.2,
            min_via_diameter_mm=0.6,
            min_via_drill_mm=0.3,
        ),
        track_width_presets_mm=[0.25, 0.5, 1.0],
        via_presets=[ViaPreset(diameter_mm=0.6, drill_mm=0.3)],
        net_classes=[
            NetClassProfile(
                name="Default",
                track_width_mm=0.25,
                clearance_mm=0.2,
                via_diameter_mm=0.6,
                via_drill_mm=0.3,
            ),
            NetClassProfile(
                name="POWER",
                track_width_mm=1.0,
                clearance_mm=0.25,
                via_diameter_mm=1.0,
                via_drill_mm=0.5,
            ),
        ],
        net_class_patterns=[NetClassPattern(pattern="VCC", net_class="POWER")],
        required_nets=["VCC", "GND"],
        stackup=StackupExpectation(copper_layers=["F.Cu", "In1.Cu", "B.Cu"]),
    )


def test_capture_and_profile_round_trip_are_project_agnostic(tmp_path: Path) -> None:
    project_file, board_file = _write_project(tmp_path)
    captured = capture_profile(json.loads(project_file.read_text()), board_file=board_file)

    assert [item.name for item in captured.net_classes] == ["Default", "POWER"]
    assert captured.net_class_patterns[0].pattern == "VCC"
    assert captured.stackup.copper_layers == ["F.Cu", "In1.Cu", "B.Cu"]

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(dump_profile(captured), encoding="utf-8")
    assert load_profile(profile_path) == captured


def test_profile_merge_is_idempotent_and_preserves_unmanaged_settings() -> None:
    original = _project_payload()
    profile = _desired_profile()

    updated, changes = apply_profile(original, profile)
    repeated, repeated_changes = apply_profile(updated, profile)

    assert changes
    assert repeated_changes == []
    assert repeated == updated
    assert updated["unrelated"] == {"keep": "yes"}
    settings = updated["board"]["design_settings"]  # type: ignore[index]
    assert settings["preserved_setting"] == {"enabled": True}  # type: ignore[index]
    assert updated["net_settings"]["unknown_net_setting"] == "preserve-me"  # type: ignore[index]


def test_ai_templates_separate_directives_from_working_notes() -> None:
    directives = render_ai_directives("demo")
    notes = render_ai_notes()

    assert "Project purpose and scope" in directives
    assert "Fabrication release requires explicit human approval" in directives
    assert "append-only" in notes
    assert "not authoritative" in notes


@pytest.fixture
def initialization_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastMCP, Path, Path]:
    project_file, board_file = _write_project(tmp_path)
    profile_path = tmp_path / ".kicad-mcp" / "board-profile.yaml"
    profile_path.parent.mkdir()
    profile_path.write_text(dump_profile(_desired_profile()), encoding="utf-8")
    config = SimpleNamespace(
        project_dir=tmp_path,
        project_file=project_file,
        pcb_file=board_file,
        max_text_response_chars=50_000,
    )
    monkeypatch.setattr(board_initialization, "get_config", lambda: config)
    board_initialization._INITIALIZATION_PLANS.clear()
    board_initialization._INITIALIZATION_RECEIPTS.clear()
    server = FastMCP("initialization-test")
    board_initialization.register(server)
    return server, project_file, profile_path


@pytest.mark.anyio
async def test_capture_defaults_to_preview_and_never_overwrites_ai(
    initialization_server: tuple[FastMCP, Path, Path],
) -> None:
    server, project_file, profile_path = initialization_server
    ai_path = project_file.parent / "AI.md"
    ai_path.write_text("owner directives\n", encoding="utf-8")

    preview = await call_tool_payload(server, "pcb_capture_board_profile", {})
    refused = await call_tool_payload(
        server,
        "pcb_capture_board_profile",
        {"write": True},
    )

    assert preview["dry_run"] is True
    assert preview["extra"]["profile_path"] == str(profile_path)
    assert refused["ok"] is False
    assert ai_path.read_text() == "owner directives\n"

    context = await call_tool_text(
        server,
        "project_get_ai_context",
        {"include_notes": False},
    )
    assert "owner directives" in context


@pytest.mark.anyio
async def test_plan_binds_ai_hash_apply_is_idempotent_and_revert_is_exact(
    initialization_server: tuple[FastMCP, Path, Path],
) -> None:
    server, project_file, _profile_path = initialization_server
    original_project = project_file.read_bytes()

    first_plan = await call_tool_payload(server, "pcb_plan_board_initialization", {})
    first_plan_id = first_plan["extra"]["plan_id"]
    ai_path = project_file.parent / "AI.md"
    ai_path.write_text("changed after planning\n", encoding="utf-8")
    stale = await call_tool_payload(
        server,
        "pcb_apply_board_initialization",
        {"plan_id": first_plan_id, "confirm_project_saved": True},
    )
    assert stale["ok"] is False
    assert "changed after planning" in stale["errors"][0]

    ai_path.unlink()
    second_plan = await call_tool_payload(server, "pcb_plan_board_initialization", {})
    applied = await call_tool_payload(
        server,
        "pcb_apply_board_initialization",
        {
            "plan_id": second_plan["extra"]["plan_id"],
            "confirm_project_saved": True,
        },
    )
    assert applied["ok"] is True
    assert ai_path.is_file()
    assert (project_file.parent / "AI_NOTES.md").is_file()
    payload = json.loads(project_file.read_text())
    settings = payload["board"]["design_settings"]
    assert settings["track_widths"] == [0.0, 0.25, 0.5, 1.0]
    assert settings["via_dimensions"] == [
        {"diameter": 0.0, "drill": 0.0},
        {"diameter": 0.6, "drill": 0.3},
    ]

    receipt_id = applied["extra"]["receipt_id"]
    reverted = await call_tool_payload(
        server,
        "pcb_revert_board_initialization",
        {"receipt_id": receipt_id},
    )
    assert reverted["ok"] is True
    assert project_file.read_bytes() == original_project
    assert not ai_path.exists()
    assert not (project_file.parent / "AI_NOTES.md").exists()


@pytest.mark.anyio
async def test_apply_requires_saved_confirmation_when_project_settings_change(
    initialization_server: tuple[FastMCP, Path, Path],
) -> None:
    server, _project_file, _profile_path = initialization_server
    plan = await call_tool_payload(server, "pcb_plan_board_initialization", {})

    result = await call_tool_payload(
        server,
        "pcb_apply_board_initialization",
        {"plan_id": plan["extra"]["plan_id"]},
    )

    assert result["ok"] is False
    assert "Save the KiCad project first" in result["errors"][0]
