from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from kicad_mcp.pcb.transaction_lifecycle import PcbTransactionLifecycleService


class FakeConnectionError(Exception):
    pass


def _run_direct[T](operation: str, command: Callable[[], T]) -> T:
    del operation
    return command()


def _service(board: object, calls: list[str]) -> PcbTransactionLifecycleService:
    def run_mutation[T](operation: str, command: Callable[[], T]) -> T:
        calls.append(operation)
        return command()

    return PcbTransactionLifecycleService(
        get_board=lambda: board,
        run_mutation=run_mutation,
        connection_errors=(FakeConnectionError, OSError),
    )


def test_transaction_operations_use_the_existing_queue_names() -> None:
    calls: list[str] = []
    effects: list[object] = []
    commit = object()
    board = SimpleNamespace(
        begin_commit=lambda: (effects.append("begin"), commit)[1],
        push_commit=lambda active: effects.append(("push", active)),
        drop_commit=lambda active: effects.append(("drop", active)),
        revert=lambda: effects.append("revert"),
    )
    service = _service(board, calls)

    assert service.begin() == (
        "Transaction group started. Use pcb_push_commit to apply or pcb_drop_commit to discard."
    )
    assert service.push() == "Transaction group committed successfully."
    assert service.drop() == "No active transaction group to discard."
    assert service.begin() == (
        "Transaction group started. Use pcb_push_commit to apply or pcb_drop_commit to discard."
    )
    assert service.drop() == "Transaction group discarded successfully."
    assert service.revert() == (
        "Board reverted to last saved state. All unsaved changes have been discarded."
    )
    assert calls == [
        "pcb_begin_commit",
        "pcb_push_commit",
        "pcb_begin_commit",
        "pcb_drop_commit",
        "pcb_revert",
    ]
    assert effects == ["begin", ("push", commit), "begin", ("drop", commit), "revert"]


def test_begin_refuses_to_replace_an_active_commit() -> None:
    calls: list[str] = []
    commit = object()
    board = SimpleNamespace(begin_commit=lambda: commit)
    service = _service(board, calls)

    assert service.begin().startswith("Transaction group started")
    assert service.begin() == (
        "A transaction group is already active. Push or drop it before starting another."
    )
    assert calls == ["pcb_begin_commit"]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            "begin",
            "Transaction grouping is not supported by the current KiCad IPC version. "
            "Mutations will be applied individually without atomic grouping.",
        ),
        ("push", "No active transaction group to commit."),
        ("drop", "No active transaction group to discard."),
        (
            "revert",
            "Revert is not supported by the current KiCad IPC version. "
            "Please save and reload the board manually.",
        ),
    ],
)
def test_unsupported_operations_preserve_legacy_messages(method: str, expected: str) -> None:
    service = _service(SimpleNamespace(), [])

    assert getattr(service, method)() == expected


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("begin", "Failed to begin transaction: offline"),
        ("push", "Failed to commit transaction: offline"),
        ("drop", "Failed to discard transaction: offline"),
        ("revert", "Failed to revert board: offline"),
    ],
)
def test_connection_errors_preserve_legacy_messages(method: str, expected: str) -> None:
    def fail() -> object:
        raise FakeConnectionError("offline")

    service = PcbTransactionLifecycleService(
        get_board=fail,
        run_mutation=_run_direct,
        connection_errors=(FakeConnectionError, OSError),
    )

    if method == "begin" or method == "revert":
        assert getattr(service, method)() == expected
    else:
        # Push/drop are intentionally local no-ops until begin() returns a commit handle.
        assert getattr(service, method)() == (
            "No active transaction group to commit."
            if method == "push"
            else "No active transaction group to discard."
        )


@pytest.mark.parametrize("method", ["push", "drop"])
def test_finish_passes_commit_handle_and_clears_failed_connection(method: str) -> None:
    commit = object()

    def fail(_commit: object) -> None:
        assert _commit is commit
        raise FakeConnectionError("offline")

    board = SimpleNamespace(
        begin_commit=lambda: commit,
        push_commit=fail,
        drop_commit=fail,
    )
    service = _service(board, [])
    assert service.begin().startswith("Transaction group started")

    expected = (
        "Failed to commit transaction: offline"
        if method == "push"
        else "Failed to discard transaction: offline"
    )
    assert getattr(service, method)() == expected
    assert getattr(service, method)() == (
        "No active transaction group to commit."
        if method == "push"
        else "No active transaction group to discard."
    )


def test_unexpected_errors_are_not_hidden() -> None:
    def fail() -> object:
        raise RuntimeError("bug")

    service = PcbTransactionLifecycleService(
        get_board=fail,
        run_mutation=_run_direct,
        connection_errors=(FakeConnectionError, OSError),
    )

    with pytest.raises(RuntimeError, match="bug"):
        service.begin()
