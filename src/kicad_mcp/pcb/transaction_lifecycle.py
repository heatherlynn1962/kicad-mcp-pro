"""FastMCP-free PCB transaction and revert lifecycle behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


class RunMutation(Protocol):
    """Serialize one live-board mutation through the existing command queue."""

    def __call__[T](self, operation: str, command: Callable[[], T]) -> T: ...


type GetBoard = Callable[[], object]
type ConnectionErrors = tuple[type[Exception], ...]


@dataclass
class PcbTransactionLifecycleService:
    """Manage transaction grouping and board revert through injected dependencies."""

    get_board: GetBoard
    run_mutation: RunMutation
    connection_errors: ConnectionErrors
    _active_board: object | None = field(default=None, init=False, repr=False)
    _active_commit: object | None = field(default=None, init=False, repr=False)

    def _run(
        self,
        *,
        method_name: str,
        operation: str,
        unsupported: str,
        success: str,
        failure_prefix: str,
    ) -> str:
        try:
            board = self.get_board()
            command = getattr(board, method_name, None)
            if not callable(command):
                return unsupported
            self.run_mutation(operation, command)
            return success
        except self.connection_errors as exc:
            return f"{failure_prefix}: {exc}"

    def begin(self) -> str:
        """Begin one atomic transaction group when supported by KiCad."""
        if self._active_commit is not None:
            return "A transaction group is already active. Push or drop it before starting another."
        try:
            board = self.get_board()
            command = getattr(board, "begin_commit", None)
            if not callable(command):
                return (
                    "Transaction grouping is not supported by the current KiCad IPC version. "
                    "Mutations will be applied individually without atomic grouping."
                )
            commit = self.run_mutation("pcb_begin_commit", command)
            if commit is None:
                return (
                    "Transaction grouping is not supported by the current KiCad IPC version. "
                    "Mutations will be applied individually without atomic grouping."
                )
            self._active_board = board
            self._active_commit = commit
            return (
                "Transaction group started. Use pcb_push_commit to apply or "
                "pcb_drop_commit to discard."
            )
        except self.connection_errors as exc:
            self._clear_active()
            return f"Failed to begin transaction: {exc}"

    def push(self) -> str:
        """Commit the active transaction group."""
        return self._finish(commit=True)

    def drop(self) -> str:
        """Discard the active transaction group."""
        return self._finish(commit=False)

    def _clear_active(self) -> None:
        self._active_board = None
        self._active_commit = None

    def _finish(self, *, commit: bool) -> str:
        board = self._active_board
        active = self._active_commit
        if board is None or active is None:
            return (
                "No active transaction group to commit."
                if commit
                else "No active transaction group to discard."
            )
        method_name = "push_commit" if commit else "drop_commit"
        operation = "pcb_push_commit" if commit else "pcb_drop_commit"
        failure_prefix = (
            "Failed to commit transaction" if commit else "Failed to discard transaction"
        )
        command = getattr(board, method_name, None)
        if not callable(command):
            self._clear_active()
            return (
                "No active transaction group to commit."
                if commit
                else "No active transaction group to discard."
            )
        try:
            self.run_mutation(operation, lambda: command(active))
        except self.connection_errors as exc:
            self._clear_active()
            return f"{failure_prefix}: {exc}"
        self._clear_active()
        return (
            "Transaction group committed successfully."
            if commit
            else "Transaction group discarded successfully."
        )

    def revert(self) -> str:
        """Revert the board to its last saved state when supported by KiCad."""
        return self._run(
            method_name="revert",
            operation="pcb_revert",
            unsupported=(
                "Revert is not supported by the current KiCad IPC version. "
                "Please save and reload the board manually."
            ),
            success=(
                "Board reverted to last saved state. All unsaved changes have been discarded."
            ),
            failure_prefix="Failed to revert board",
        )
