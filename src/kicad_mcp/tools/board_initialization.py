"""Generic project board-profile initialization tools."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..config import get_config
from ..models.tool_result import StateDelta, ToolResult
from ..path_safety import assert_within, resolve_under
from ..pcb.board_initialization import (
    AI_DIRECTIVES_FILENAME,
    AI_NOTES_FILENAME,
    PROFILE_RELATIVE_PATH,
    BoardInitializationProfile,
    ViaPreset,
    apply_profile,
    atomic_write_text,
    capture_profile,
    dump_profile,
    file_fingerprint,
    load_profile,
    load_project_payload,
    render_ai_directives,
    render_ai_notes,
    render_project_payload,
    validate_profile_against_board,
)
from .metadata import headless_compatible


@dataclass(frozen=True, slots=True)
class _InitializationPlan:
    plan_id: str
    project_dir: Path
    project_file: Path
    profile_path: Path
    expected_fingerprints: dict[Path, str]
    updated_project_text: str
    changes: tuple[str, ...]
    warnings: tuple[str, ...]
    create_ai_directives: bool
    create_ai_notes: bool


@dataclass(frozen=True, slots=True)
class _InitializationReceipt:
    receipt_id: str
    originals: dict[Path, bytes | None]
    post_fingerprints: dict[Path, str]
    summary: str


_INITIALIZATION_PLANS: dict[str, _InitializationPlan] = {}
_INITIALIZATION_RECEIPTS: dict[str, _InitializationReceipt] = {}


def _project_paths(profile_path: str = "") -> tuple[Path, Path, Path, Path, Path]:
    cfg = get_config()
    if cfg.project_dir is None or cfg.project_file is None:
        raise ValueError(
            "No KiCad project is configured. Call kicad_set_project() before initialization."
        )
    project_dir = cfg.project_dir.expanduser().resolve()
    project_file = cfg.project_file.expanduser().resolve()
    assert_within(project_dir, project_file)
    resolved_profile = (
        resolve_under(project_dir, profile_path, allow_absolute=True)
        if profile_path
        else (project_dir / PROFILE_RELATIVE_PATH).resolve()
    )
    assert_within(project_dir, resolved_profile)
    return (
        project_dir,
        project_file,
        resolved_profile,
        project_dir / AI_DIRECTIVES_FILENAME,
        project_dir / AI_NOTES_FILENAME,
    )


def _notes_entry(action: str, identifier: str, details: list[str]) -> str:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = ["", f"## {timestamp} - {action}", "", f"- Receipt/plan: `{identifier}`"]
    lines.extend(f"- {detail}" for detail in details)
    return "\n".join(lines) + "\n"


def _read_optional(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    atomic_write_text(path, content.decode("utf-8"))


def _write_set(writes: dict[Path, str]) -> dict[Path, bytes | None]:
    originals = {path: _read_optional(path) for path in writes}
    completed: list[Path] = []
    try:
        for path, text in writes.items():
            atomic_write_text(path, text)
            completed.append(path)
    except Exception:
        for path in reversed(completed):
            original = originals[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(path, original)
        raise
    return originals


def _restore_originals(originals: dict[Path, bytes | None]) -> None:
    for path, original in originals.items():
        if original is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(path, original)


def _store_receipt(
    *,
    originals: dict[Path, bytes | None],
    summary: str,
    receipt_id: str | None = None,
) -> _InitializationReceipt:
    receipt = _InitializationReceipt(
        receipt_id=receipt_id or str(uuid.uuid4()),
        originals=originals,
        post_fingerprints={path: file_fingerprint(path) for path in originals},
        summary=summary,
    )
    _INITIALIZATION_RECEIPTS[receipt.receipt_id] = receipt
    return receipt


def _profile_payload(profile: BoardInitializationProfile) -> dict[str, object]:
    return profile.model_dump(mode="json", exclude_none=True)


def register(mcp: FastMCP) -> None:
    """Register generic board initialization and directive-file tools."""

    @mcp.tool()
    @headless_compatible
    def project_get_ai_context(include_notes: bool = False) -> str:
        """Read the project-owned AI directives and optionally its working notes."""
        try:
            _project_dir, _project_file, _profile, ai_path, notes_path = _project_paths()
        except ValueError as exc:
            return str(exc)
        if not ai_path.is_file():
            return (
                "AI.md is not initialized for this project. "
                "Use pcb_capture_board_profile or pcb_plan_board_initialization first."
            )
        sections = [
            "# Project AI context",
            "",
            "Authority: KiCad files describe implemented state; AI.md defines project intent and "
            "boundaries; AI_NOTES.md is non-authoritative history.",
            "",
            f"## {AI_DIRECTIVES_FILENAME}",
            "",
            ai_path.read_text(encoding="utf-8"),
        ]
        if include_notes:
            sections.extend(
                [
                    "",
                    f"## {AI_NOTES_FILENAME}",
                    "",
                    (
                        notes_path.read_text(encoding="utf-8")
                        if notes_path.is_file()
                        else "(not initialized)"
                    ),
                ]
            )
        rendered = "\n".join(sections)
        limit = get_config().max_text_response_chars
        if len(rendered) > limit:
            return rendered[:limit] + "\n... [truncated]"
        return rendered

    @mcp.tool()
    @headless_compatible
    def pcb_capture_board_profile(
        profile_path: str = "",
        write: bool = False,
        overwrite: bool = False,
        include_net_class_sizes_as_presets: bool = True,
    ) -> ToolResult:
        """Capture any project's current board policy into a portable YAML profile.

        The default call is a preview. With ``write=True``, the tool also creates
        missing AI.md and AI_NOTES.md files, but never overwrites either directive file.
        """
        try:
            project_dir, project_file, profile, ai_path, notes_path = _project_paths(profile_path)
            project_payload = load_project_payload(project_file)
            cfg = get_config()
            captured = capture_profile(project_payload, board_file=cfg.pcb_file)
            if include_net_class_sizes_as_presets:
                widths = sorted(
                    {
                        *captured.track_width_presets_mm,
                        *(item.track_width_mm for item in captured.net_classes),
                    }
                )
                via_pairs = {(item.diameter_mm, item.drill_mm) for item in captured.via_presets}
                via_pairs.update(
                    (item.via_diameter_mm, item.via_drill_mm) for item in captured.net_classes
                )
                captured = captured.model_copy(
                    update={
                        "track_width_presets_mm": widths,
                        "via_presets": [
                            ViaPreset(diameter_mm=diameter, drill_mm=drill)
                            for diameter, drill in sorted(via_pairs)
                        ],
                    }
                )
            rendered = dump_profile(captured)
        except (OSError, ValueError) as exc:
            return ToolResult.failure("pcb_capture_board_profile", str(exc))

        if not write:
            return ToolResult.dry_run_result(
                "pcb_capture_board_profile",
                f"Previewed board initialization profile for {project_file.name}.",
                extra={
                    "profile_path": str(profile),
                    "profile": _profile_payload(captured),
                    "yaml": rendered,
                    "would_create": [
                        str(path) for path in (profile, ai_path, notes_path) if not path.exists()
                    ],
                },
            )
        if profile.exists() and not overwrite:
            return ToolResult.failure(
                "pcb_capture_board_profile",
                f"Board profile already exists at {profile}; set overwrite=True to replace it.",
            )

        capture_id = str(uuid.uuid4())
        writes = {profile: rendered}
        if not ai_path.exists():
            writes[ai_path] = render_ai_directives(project_file.stem)
        notes_text = (
            notes_path.read_text(encoding="utf-8") if notes_path.is_file() else render_ai_notes()
        )
        writes[notes_path] = (
            notes_text.rstrip()
            + "\n"
            + _notes_entry(
                "Captured board initialization profile",
                capture_id,
                [
                    f"Project: `{project_file.name}`",
                    f"Profile: `{profile.relative_to(project_dir)}`",
                    "Source: existing KiCad project settings",
                ],
            )
        )
        try:
            originals = _write_set(writes)
        except (OSError, UnicodeError) as exc:
            return ToolResult.failure("pcb_capture_board_profile", f"Capture failed: {exc}")
        receipt = _store_receipt(
            originals=originals,
            summary=f"Captured board profile for {project_file.name}.",
            receipt_id=capture_id,
        )
        return ToolResult.success(
            "pcb_capture_board_profile",
            changed=True,
            rollback_token=receipt.receipt_id,
            state_delta=StateDelta(summary=receipt.summary),
            extra={
                "profile_path": str(profile),
                "created_or_updated": [str(path) for path in writes],
                "receipt_id": receipt.receipt_id,
            },
        )

    @mcp.tool()
    @headless_compatible
    def pcb_plan_board_initialization(profile_path: str = "") -> ToolResult:
        """Compare a project-owned board profile with the active KiCad project."""
        try:
            project_dir, project_file, profile_path_obj, ai_path, notes_path = _project_paths(
                profile_path
            )
            profile = load_profile(profile_path_obj)
            original_payload = load_project_payload(project_file)
            updated_payload, changes = apply_profile(original_payload, profile)
            cfg = get_config()
            warnings = validate_profile_against_board(profile, board_file=cfg.pcb_file)
            updated_text = render_project_payload(updated_payload)
            plan = _InitializationPlan(
                plan_id=str(uuid.uuid4()),
                project_dir=project_dir,
                project_file=project_file,
                profile_path=profile_path_obj,
                expected_fingerprints={
                    project_file: file_fingerprint(project_file),
                    profile_path_obj: file_fingerprint(profile_path_obj),
                    ai_path: file_fingerprint(ai_path),
                    notes_path: file_fingerprint(notes_path),
                },
                updated_project_text=updated_text,
                changes=tuple(changes),
                warnings=tuple(warnings),
                create_ai_directives=not ai_path.exists(),
                create_ai_notes=not notes_path.exists(),
            )
            _INITIALIZATION_PLANS[plan.plan_id] = plan
        except (OSError, ValueError) as exc:
            return ToolResult.failure("pcb_plan_board_initialization", str(exc))

        return ToolResult.dry_run_result(
            "pcb_plan_board_initialization",
            (
                f"Planned board initialization for {project_file.name}: "
                f"{len(changes)} project-setting change(s)."
            ),
            warnings=warnings,
            extra={
                "plan_id": plan.plan_id,
                "profile_path": str(profile_path_obj),
                "changes": changes,
                "create_ai_md": plan.create_ai_directives,
                "create_ai_notes_md": plan.create_ai_notes,
                "directive_fingerprint": plan.expected_fingerprints[ai_path],
                "reload_required_after_apply": bool(changes),
            },
        )

    @mcp.tool()
    @headless_compatible
    def pcb_apply_board_initialization(
        plan_id: str,
        confirm_project_saved: bool = False,
    ) -> ToolResult:
        """Atomically apply one current initialization plan and return a rollback receipt."""
        plan = _INITIALIZATION_PLANS.get(plan_id)
        if plan is None:
            return ToolResult.failure(
                "pcb_apply_board_initialization", "Unknown, expired, or already-applied plan ID."
            )
        stale = [
            str(path)
            for path, expected in plan.expected_fingerprints.items()
            if file_fingerprint(path) != expected
        ]
        if stale:
            return ToolResult.failure(
                "pcb_apply_board_initialization",
                "Project state changed after planning. Re-plan before applying: "
                + ", ".join(stale),
            )
        if plan.changes and not confirm_project_saved:
            return ToolResult.failure(
                "pcb_apply_board_initialization",
                "Save the KiCad project first, then retry with confirm_project_saved=True. "
                "Project settings must be reloaded after apply.",
            )

        ai_path = plan.project_dir / AI_DIRECTIVES_FILENAME
        notes_path = plan.project_dir / AI_NOTES_FILENAME
        writes: dict[Path, str] = {}
        if plan.changes:
            writes[plan.project_file] = plan.updated_project_text
        if plan.create_ai_directives:
            writes[ai_path] = render_ai_directives(plan.project_file.stem)
        if plan.create_ai_notes:
            notes_text = render_ai_notes()
        else:
            notes_text = notes_path.read_text(encoding="utf-8")

        if writes or plan.create_ai_notes:
            details = [
                f"Project: `{plan.project_file.name}`",
                f"Profile: `{plan.profile_path.relative_to(plan.project_dir)}`",
            ]
            if plan.changes:
                details.append("Applied settings: " + ", ".join(plan.changes))
            if plan.create_ai_directives:
                details.append("Created `AI.md` from the generic directive template")
            writes[notes_path] = (
                notes_text.rstrip()
                + "\n"
                + _notes_entry(
                    "Applied board initialization",
                    plan.plan_id,
                    details,
                )
            )

        if not writes:
            _INITIALIZATION_PLANS.pop(plan_id, None)
            return ToolResult.success(
                "pcb_apply_board_initialization",
                changed=False,
                state_delta=StateDelta(summary="Board initialization already matches the profile."),
                extra={"plan_id": plan_id, "reload_required": False},
            )

        originals: dict[Path, bytes | None] = {}
        try:
            originals = _write_set(writes)
            # Parse the project after replacement before declaring success.
            if plan.project_file in writes:
                load_project_payload(plan.project_file)
        except (OSError, UnicodeError, ValueError) as exc:
            if originals:
                _restore_originals(originals)
            return ToolResult.failure(
                "pcb_apply_board_initialization", f"Initialization failed: {exc}"
            )

        receipt = _store_receipt(
            originals=originals,
            summary=(
                f"Applied {len(plan.changes)} board-setting change(s) and initialized project "
                "AI directives."
            ),
        )
        _INITIALIZATION_PLANS.pop(plan_id, None)
        return ToolResult.success(
            "pcb_apply_board_initialization",
            changed=True,
            rollback_token=receipt.receipt_id,
            state_delta=StateDelta(
                pre_fingerprint=plan.expected_fingerprints[plan.project_file],
                post_fingerprint=file_fingerprint(plan.project_file),
                summary=receipt.summary,
            ),
            extra={
                "receipt_id": receipt.receipt_id,
                "changed_settings": list(plan.changes),
                "files_written": [str(path) for path in writes],
                "reload_required": bool(plan.changes),
            },
        )

    @mcp.tool()
    @headless_compatible
    def pcb_revert_board_initialization(receipt_id: str) -> ToolResult:
        """Restore exactly the files changed by one initialization or capture receipt."""
        receipt = _INITIALIZATION_RECEIPTS.get(receipt_id)
        if receipt is None:
            return ToolResult.failure(
                "pcb_revert_board_initialization", "Unknown or already-reverted receipt ID."
            )
        stale = [
            str(path)
            for path, expected in receipt.post_fingerprints.items()
            if file_fingerprint(path) != expected
        ]
        if stale:
            return ToolResult.failure(
                "pcb_revert_board_initialization",
                "A managed file changed after initialization; refusing to overwrite it: "
                + ", ".join(stale),
            )
        try:
            _restore_originals(receipt.originals)
        except (OSError, UnicodeError) as exc:
            return ToolResult.failure(
                "pcb_revert_board_initialization", f"Initialization revert failed: {exc}"
            )
        _INITIALIZATION_RECEIPTS.pop(receipt_id, None)
        return ToolResult.success(
            "pcb_revert_board_initialization",
            changed=True,
            state_delta=StateDelta(summary=f"Reverted: {receipt.summary}"),
            extra={"restored_files": [str(path) for path in receipt.originals]},
        )


def debug_state() -> str:
    """Return compact internal state for tests without exposing it as an MCP tool."""
    return json.dumps(
        {
            "plans": sorted(_INITIALIZATION_PLANS),
            "receipts": sorted(_INITIALIZATION_RECEIPTS),
        }
    )
