"""Project-agnostic KiCad board initialization profiles and AI directives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROFILE_SCHEMA_VERSION = "1.0"
PROFILE_RELATIVE_PATH = Path(".kicad-mcp") / "board-profile.yaml"
AI_DIRECTIVES_FILENAME = "AI.md"
AI_NOTES_FILENAME = "AI_NOTES.md"
MISSING_FINGERPRINT = "missing"


class ViaPreset(BaseModel):
    """One selectable KiCad via diameter/drill pair."""

    model_config = ConfigDict(extra="forbid")

    diameter_mm: float = Field(gt=0.0, le=25.0)
    drill_mm: float = Field(gt=0.0, le=25.0)

    @model_validator(mode="after")
    def drill_fits_diameter(self) -> ViaPreset:
        if self.drill_mm >= self.diameter_mm:
            raise ValueError("Via drill must be smaller than via diameter.")
        return self


class DifferentialPairPreset(BaseModel):
    """One selectable differential-pair geometry."""

    model_config = ConfigDict(extra="forbid")

    width_mm: float = Field(gt=0.0, le=25.0)
    gap_mm: float = Field(ge=0.0, le=25.0)
    via_gap_mm: float = Field(ge=0.0, le=25.0)


class ManufacturingMinimums(BaseModel):
    """Optional project manufacturing and DRC floors."""

    model_config = ConfigDict(extra="forbid")

    min_track_width_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_clearance_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_via_diameter_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_via_drill_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_via_annular_width_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_hole_clearance_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_hole_to_hole_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    min_copper_edge_clearance_mm: float | None = Field(default=None, ge=0.0, le=25.0)


class NetClassProfile(BaseModel):
    """Routing defaults for a named KiCad net class."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    track_width_mm: float = Field(gt=0.0, le=25.0)
    clearance_mm: float = Field(ge=0.0, le=25.0)
    via_diameter_mm: float = Field(gt=0.0, le=25.0)
    via_drill_mm: float = Field(gt=0.0, le=25.0)
    diff_pair_width_mm: float | None = Field(default=None, gt=0.0, le=25.0)
    diff_pair_gap_mm: float | None = Field(default=None, ge=0.0, le=25.0)
    diff_pair_via_gap_mm: float | None = Field(default=None, ge=0.0, le=25.0)

    @model_validator(mode="after")
    def drill_fits_diameter(self) -> NetClassProfile:
        if self.via_drill_mm >= self.via_diameter_mm:
            raise ValueError(f"Net class {self.name!r} has drill >= via diameter.")
        return self


class NetClassPattern(BaseModel):
    """A KiCad net-name pattern bound to a net class."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1, max_length=500)
    net_class: str = Field(min_length=1, max_length=100)


class StackupExpectation(BaseModel):
    """Read-only stackup expectations validated during initialization."""

    model_config = ConfigDict(extra="forbid")

    copper_layers: list[str] = Field(default_factory=list)
    layer_roles: dict[str, str] = Field(default_factory=dict)


class BoardInitializationProfile(BaseModel):
    """Portable, project-owned board initialization policy."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PROFILE_SCHEMA_VERSION
    manufacturing: ManufacturingMinimums = Field(default_factory=ManufacturingMinimums)
    track_width_presets_mm: list[float] = Field(default_factory=list)
    via_presets: list[ViaPreset] = Field(default_factory=list)
    differential_pair_presets: list[DifferentialPairPreset] = Field(default_factory=list)
    net_classes: list[NetClassProfile] = Field(default_factory=list)
    net_class_patterns: list[NetClassPattern] = Field(default_factory=list)
    required_nets: list[str] = Field(default_factory=list)
    stackup: StackupExpectation = Field(default_factory=StackupExpectation)

    @model_validator(mode="after")
    def validate_references(self) -> BoardInitializationProfile:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported board profile schema {self.schema_version!r}; "
                f"expected {PROFILE_SCHEMA_VERSION!r}."
            )
        class_names = [item.name for item in self.net_classes]
        if len(class_names) != len(set(class_names)):
            raise ValueError("Net class names must be unique.")
        unknown = sorted(
            {item.net_class for item in self.net_class_patterns}.difference(class_names)
        )
        if unknown:
            raise ValueError(f"Net patterns reference undefined classes: {', '.join(unknown)}.")
        if any(width <= 0.0 or width > 25.0 for width in self.track_width_presets_mm):
            raise ValueError("Track-width presets must be greater than zero and at most 25 mm.")
        return self


_MANUFACTURING_PROJECT_KEYS = {
    "min_track_width_mm": "min_track_width",
    "min_clearance_mm": "min_clearance",
    "min_via_diameter_mm": "min_via_diameter",
    "min_via_drill_mm": "min_via_drill",
    "min_via_annular_width_mm": "min_via_annular_width",
    "min_hole_clearance_mm": "min_hole_clearance",
    "min_hole_to_hole_mm": "min_hole_to_hole",
    "min_copper_edge_clearance_mm": "min_copper_edge_clearance",
}


def file_fingerprint(path: Path) -> str:
    """Hash a file, preserving missing-file state as a stable sentinel."""
    if not path.is_file():
        return MISSING_FINGERPRINT
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace one project file from the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_ai_directives(project_name: str) -> str:
    """Render a conservative project-boundary contract with no invented facts."""
    return f"""# AI Project Directives

This file is the project-owned contract for AI-assisted work on `{project_name}`.
Edit it deliberately. AI tools must preserve existing content and stop at stated hard stops.

## Project purpose and scope

- Purpose: TODO - define the board's intended function.
- In scope: TODO - define the work this project permits.
- Out of scope: TODO - define excluded systems, revisions, or authority.

## Architecture and subsystem boundaries

- TODO - identify subsystem ownership, external interfaces, and authority boundaries.

## Electrical and safety constraints

- TODO - identify voltage, current, isolation, protection, and safety constraints.

## Layer, plane, and routing intent

- TODO - identify layer roles, reference planes, critical nets, and routing constraints.
- Routing sizes and net-class assignments are defined in `.kicad-mcp/board-profile.yaml`.

## Mechanical and component constraints

- TODO - identify board outline, mounting, connector access, antenna, and placement constraints.

## Must preserve

- Existing verified design work unless a reviewed plan explicitly changes it.
- Project files unrelated to the approved task.

## Agent permissions and hard stops

- Preview project mutations before applying them.
- Do not invent project-specific facts or silently create electrical nets.
- Stop before fabrication release, destructive replacement, or an unresolved safety decision.
- Do not treat `AI_NOTES.md` as authoritative project truth.

## Verification and release gates

- Run the relevant ERC, DRC, visual, 3D, and manufacturing checks after changes.
- Fabrication release requires explicit human approval.

## Unresolved decisions

- TODO - list decisions that remain open.
"""


def render_ai_notes() -> str:
    """Render the non-authoritative working-log header."""
    return """# AI Working Notes

This is an append-only activity and handoff log. It is not authoritative project truth.
Project intent and boundaries belong in `AI.md`; implemented state belongs in the KiCad files.
"""


def load_project_payload(project_file: Path) -> dict[str, Any]:
    try:
        payload = json.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read KiCad project file {project_file}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"KiCad project file {project_file} does not contain a JSON object.")
    return payload


def render_project_payload(payload: Mapping[str, Any]) -> str:
    """Render deterministic KiCad project JSON and validate the round trip."""
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Rendered KiCad project payload did not round-trip as an object.")
    return text


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def copper_layers_from_board(board_file: Path | None) -> list[str]:
    """Read copper-layer names from the board's layer table without mutating it."""
    if board_file is None or not board_file.is_file():
        return []
    text = board_file.read_text(encoding="utf-8", errors="replace")
    layers_match = re.search(r"\(layers\s+(.*?)\n\s*\)", text, flags=re.DOTALL)
    if layers_match is None:
        return []
    layers = []
    for name, kind in re.findall(r'\(\s*\d+\s+"([^"]+)"\s+([^)\s]+)', layers_match.group(1)):
        if name.endswith(".Cu") or kind in {"signal", "power", "mixed"}:
            layers.append(name)
    return layers


def net_names_from_board(board_file: Path | None) -> set[str]:
    """Read named nets from a KiCad board file for required-net validation."""
    if board_file is None or not board_file.is_file():
        return set()
    text = board_file.read_text(encoding="utf-8", errors="replace")
    names: set[str] = set()
    for encoded_name in re.findall(r'\(net\s+\d+\s+"((?:\\.|[^"\\])*)"\)', text):
        if not encoded_name:
            continue
        try:
            names.add(json.loads(f'"{encoded_name}"'))
        except json.JSONDecodeError:
            names.add(encoded_name)
    return names


def capture_profile(
    project_payload: Mapping[str, Any],
    *,
    board_file: Path | None = None,
) -> BoardInitializationProfile:
    """Capture portable routing policy from any KiCad project payload."""
    board = _mapping(project_payload.get("board"))
    design_settings = _mapping(board.get("design_settings"))
    rules = _mapping(design_settings.get("rules"))
    manufacturing_values = {
        profile_key: rules.get(project_key)
        for profile_key, project_key in _MANUFACTURING_PROJECT_KEYS.items()
        if isinstance(rules.get(project_key), int | float)
    }

    via_presets = []
    for item in _list_of_mappings(design_settings.get("via_dimensions")):
        diameter = item.get("diameter")
        drill = item.get("drill")
        if isinstance(diameter, int | float) and isinstance(drill, int | float):
            if diameter > 0 and 0 < drill < diameter:
                via_presets.append(ViaPreset(diameter_mm=diameter, drill_mm=drill))

    differential_pair_presets = []
    for item in _list_of_mappings(design_settings.get("diff_pair_dimensions")):
        width = item.get("width")
        gap = item.get("gap")
        via_gap = item.get("via_gap")
        if (
            isinstance(width, int | float)
            and isinstance(gap, int | float)
            and isinstance(via_gap, int | float)
            and width > 0
        ):
            differential_pair_presets.append(
                DifferentialPairPreset(width_mm=width, gap_mm=gap, via_gap_mm=via_gap)
            )

    net_settings = _mapping(project_payload.get("net_settings"))
    classes = []
    for item in _list_of_mappings(net_settings.get("classes")):
        try:
            classes.append(
                NetClassProfile(
                    name=str(item["name"]),
                    track_width_mm=float(item["track_width"]),
                    clearance_mm=float(item["clearance"]),
                    via_diameter_mm=float(item["via_diameter"]),
                    via_drill_mm=float(item["via_drill"]),
                    diff_pair_width_mm=item.get("diff_pair_width"),
                    diff_pair_gap_mm=item.get("diff_pair_gap"),
                    diff_pair_via_gap_mm=item.get("diff_pair_via_gap"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    patterns = []
    for item in _list_of_mappings(net_settings.get("netclass_patterns")):
        pattern = item.get("pattern")
        net_class = item.get("netclass")
        if isinstance(pattern, str) and pattern and isinstance(net_class, str) and net_class:
            patterns.append(NetClassPattern(pattern=pattern, net_class=net_class))

    widths = [
        float(value)
        for value in design_settings.get("track_widths", [])
        if isinstance(value, int | float) and value > 0
    ]
    return BoardInitializationProfile(
        manufacturing=ManufacturingMinimums.model_validate(manufacturing_values),
        track_width_presets_mm=widths,
        via_presets=via_presets,
        differential_pair_presets=differential_pair_presets,
        net_classes=classes,
        net_class_patterns=patterns,
        required_nets=[],
        stackup=StackupExpectation(copper_layers=copper_layers_from_board(board_file)),
    )


def dump_profile(profile: BoardInitializationProfile) -> str:
    """Render a stable, human-editable YAML profile."""
    return yaml.safe_dump(
        profile.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )


def load_profile(path: Path) -> BoardInitializationProfile:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read board profile {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"Board profile {path} must contain a YAML mapping.")
    return BoardInitializationProfile.model_validate(data)


def apply_profile(
    project_payload: Mapping[str, Any],
    profile: BoardInitializationProfile,
) -> tuple[dict[str, Any], list[str]]:
    """Merge only profile-owned settings into a KiCad project payload."""
    updated = deepcopy(dict(project_payload))
    changes: list[str] = []

    board = updated.setdefault("board", {})
    if not isinstance(board, dict):
        raise ValueError("KiCad project 'board' field is not an object.")
    design_settings = board.setdefault("design_settings", {})
    if not isinstance(design_settings, dict):
        raise ValueError("KiCad project board design settings are not an object.")
    rules = design_settings.setdefault("rules", {})
    if not isinstance(rules, dict):
        raise ValueError("KiCad project board rules are not an object.")

    for profile_key, project_key in _MANUFACTURING_PROJECT_KEYS.items():
        value = getattr(profile.manufacturing, profile_key)
        if value is not None and rules.get(project_key) != value:
            rules[project_key] = value
            changes.append(f"manufacturing.{profile_key}")

    desired_widths = [0.0, *sorted(set(profile.track_width_presets_mm))]
    if profile.track_width_presets_mm and design_settings.get("track_widths") != desired_widths:
        design_settings["track_widths"] = desired_widths
        changes.append("track_width_presets_mm")

    desired_vias = [
        {"diameter": 0.0, "drill": 0.0},
        *[{"diameter": item.diameter_mm, "drill": item.drill_mm} for item in profile.via_presets],
    ]
    if profile.via_presets and design_settings.get("via_dimensions") != desired_vias:
        design_settings["via_dimensions"] = desired_vias
        changes.append("via_presets")

    desired_pairs = [
        {"gap": 0.0, "via_gap": 0.0, "width": 0.0},
        *[
            {"gap": item.gap_mm, "via_gap": item.via_gap_mm, "width": item.width_mm}
            for item in profile.differential_pair_presets
        ],
    ]
    if (
        profile.differential_pair_presets
        and design_settings.get("diff_pair_dimensions") != desired_pairs
    ):
        design_settings["diff_pair_dimensions"] = desired_pairs
        changes.append("differential_pair_presets")

    net_settings = updated.setdefault("net_settings", {})
    if not isinstance(net_settings, dict):
        raise ValueError("KiCad project net settings are not an object.")
    existing_classes = _list_of_mappings(net_settings.get("classes"))
    default_class = next(
        (deepcopy(item) for item in existing_classes if item.get("name") == "Default"),
        {},
    )
    by_name = {
        str(item.get("name")): deepcopy(item)
        for item in existing_classes
        if isinstance(item.get("name"), str)
    }
    for index, item in enumerate(profile.net_classes):
        is_new_class = item.name not in by_name
        candidate = by_name.get(item.name, deepcopy(default_class))
        candidate.update(
            {
                "name": item.name,
                "track_width": item.track_width_mm,
                "clearance": item.clearance_mm,
                "via_diameter": item.via_diameter_mm,
                "via_drill": item.via_drill_mm,
            }
        )
        if item.diff_pair_width_mm is not None:
            candidate["diff_pair_width"] = item.diff_pair_width_mm
        if item.diff_pair_gap_mm is not None:
            candidate["diff_pair_gap"] = item.diff_pair_gap_mm
        if item.diff_pair_via_gap_mm is not None:
            candidate["diff_pair_via_gap"] = item.diff_pair_via_gap_mm
        if is_new_class:
            candidate["priority"] = index
        by_name[item.name] = candidate
    desired_classes = list(by_name.values())
    if profile.net_classes and existing_classes != desired_classes:
        net_settings["classes"] = desired_classes
        changes.append("net_classes")

    existing_patterns = _list_of_mappings(net_settings.get("netclass_patterns"))
    by_pattern = {
        str(item.get("pattern")): deepcopy(item)
        for item in existing_patterns
        if isinstance(item.get("pattern"), str)
    }
    for pattern_entry in profile.net_class_patterns:
        by_pattern[pattern_entry.pattern] = {
            "netclass": pattern_entry.net_class,
            "pattern": pattern_entry.pattern,
        }
    desired_patterns = list(by_pattern.values())
    if profile.net_class_patterns and existing_patterns != desired_patterns:
        net_settings["netclass_patterns"] = desired_patterns
        changes.append("net_class_patterns")

    return updated, changes


def validate_profile_against_board(
    profile: BoardInitializationProfile,
    *,
    board_file: Path | None,
) -> list[str]:
    """Return non-mutating project mismatches that require review."""
    warnings: list[str] = []
    available_nets = net_names_from_board(board_file)
    if profile.required_nets and not available_nets:
        warnings.append("Required nets could not be checked because the PCB file was unavailable.")
    else:
        missing_nets = sorted(set(profile.required_nets).difference(available_nets))
        if missing_nets:
            warnings.append(f"Required nets are missing from the PCB: {', '.join(missing_nets)}.")

    actual_layers = copper_layers_from_board(board_file)
    if profile.stackup.copper_layers and actual_layers != profile.stackup.copper_layers:
        warnings.append(
            "Copper-layer expectation differs: expected "
            f"{profile.stackup.copper_layers}, found {actual_layers or '(unavailable)'}."
        )
    return warnings
