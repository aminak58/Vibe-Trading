"""Server-owned obligations for an explicitly requested workflow topology.

Execution identity answers *what* may execute.  This module deliberately
answers only *how* the current user required it to execute.  It never reads
memory, assembled prompts, or model-generated prose.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


WORKFLOW_OBLIGATION_ARTIFACT = "workflow_obligation.json"


class WorkflowMode(str, Enum):
    NONE = "none"
    SWARM_REQUIRED = "swarm_required"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    SWARM_STARTED = "swarm_started"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowObligation(BaseModel):
    """Immutable-ish persisted state for a current-user workflow requirement."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "workflow-obligation/v1"
    mode: WorkflowMode = WorkflowMode.NONE
    required_preset: str | None = None
    authority: str = "current_user"
    status: WorkflowStatus = WorkflowStatus.PENDING
    identity_hash: str | None = None
    swarm_run_id: str | None = None
    evidence: str = "current_raw_user_message"

    def transition(self, status: WorkflowStatus, **updates: str | None) -> "WorkflowObligation":
        return self.model_copy(update={"status": status, **updates})


# This deliberately has a narrow vocabulary.  Do not turn ordinary historic
# prose, summaries, or broad words such as "analysis" into an obligation.
_SWARM_RE = re.compile(r"\b(?:run|use|start|launch)\s+(?:the\s+)?swarm\b|\bswarm\s+(?:workflow|team|analysis)\b", re.IGNORECASE)
_PRESET_RE = re.compile(r"\b(quant_scalp_desk)\b", re.IGNORECASE)


def obligation_from_current_user_message(user_message: str) -> WorkflowObligation:
    """Parse only this turn's raw user message into a bounded obligation."""
    text = user_message or ""
    preset_match = _PRESET_RE.search(text)
    preset = preset_match.group(1).casefold() if preset_match else None
    if _SWARM_RE.search(text) or preset is not None:
        return WorkflowObligation(
            mode=WorkflowMode.SWARM_REQUIRED,
            required_preset=preset,
        )
    return WorkflowObligation(mode=WorkflowMode.NONE)


class WorkflowObligationLedger:
    """Persist and enforce the server-side workflow obligation for one run."""

    def __init__(self, *, run_dir: Path, user_message: str) -> None:
        self.run_dir = Path(run_dir)
        restored = self._load()
        # A running/waiting swarm survives a server restart.  A terminal
        # artifact must never impose a previous turn's workflow on this turn.
        if restored and restored.mode is WorkflowMode.SWARM_REQUIRED and restored.status in {
            WorkflowStatus.SWARM_STARTED,
            WorkflowStatus.WAITING,
        }:
            self._obligation = restored
        else:
            self._obligation = obligation_from_current_user_message(user_message)
            self.persist()

    @property
    def obligation(self) -> WorkflowObligation:
        return self._obligation

    def bind_identity(self, identity_hash: str) -> None:
        if self._obligation.mode is WorkflowMode.SWARM_REQUIRED and not self._obligation.identity_hash:
            self._obligation = self._obligation.transition(self._obligation.status, identity_hash=identity_hash)
            self.persist()

    def record_swarm_result(self, result: str) -> None:
        if self._obligation.mode is not WorkflowMode.SWARM_REQUIRED:
            return
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            self._obligation = self._obligation.transition(WorkflowStatus.FAILED)
            self.persist()
            return
        if not isinstance(payload, dict):
            self._obligation = self._obligation.transition(WorkflowStatus.FAILED)
        else:
            status = str(payload.get("status") or "").casefold()
            run_id = str(payload.get("run_id") or "") or None
            if payload.get("wait_budget_exhausted") or status in {"pending", "running"}:
                self._obligation = self._obligation.transition(WorkflowStatus.WAITING, swarm_run_id=run_id)
            elif status == "completed":
                self._obligation = self._obligation.transition(WorkflowStatus.COMPLETED, swarm_run_id=run_id)
            elif status in {"cancelled", "canceled"}:
                self._obligation = self._obligation.transition(WorkflowStatus.CANCELLED, swarm_run_id=run_id)
            else:
                self._obligation = self._obligation.transition(WorkflowStatus.FAILED, swarm_run_id=run_id)
        self.persist()

    def mark_swarm_started(self) -> None:
        """Record dispatch before a long-running Swarm tool returns."""
        if self._obligation.mode is WorkflowMode.SWARM_REQUIRED:
            self._obligation = self._obligation.transition(WorkflowStatus.SWARM_STARTED)
            self.persist()

    def record_status_result(self, result: str) -> None:
        if self._obligation.mode is not WorkflowMode.SWARM_REQUIRED:
            return
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict) or payload.get("run_id") != self._obligation.swarm_run_id:
            return
        status = str(payload.get("status") or "").casefold()
        terminal = {"completed": WorkflowStatus.COMPLETED, "failed": WorkflowStatus.FAILED,
                    "cancelled": WorkflowStatus.CANCELLED, "canceled": WorkflowStatus.CANCELLED}
        if status in terminal:
            self._obligation = self._obligation.transition(terminal[status])
            self.persist()

    def block(self, tool_name: str, arguments: dict) -> str | None:
        """Return a structured denial for a topology-changing tool call."""
        obligation = self._obligation
        if obligation.mode is not WorkflowMode.SWARM_REQUIRED:
            return None
        if tool_name == "run_swarm":
            supplied = str(arguments.get("preset_name") or "").casefold() or None
            if obligation.required_preset and supplied != obligation.required_preset:
                return _denial(obligation, "An explicit current-user preset must be used exactly.")
            return None
        # Source resolution, document extraction, market-data inspection, and
        # contract discovery are prerequisites; only replacement execution is
        # denied here.  Identity/source policy continues to enforce their args.
        if tool_name == "backtest":
            return _denial(obligation, "Run the required Swarm workflow or report it unavailable; do not substitute direct backtest execution.")
        return None

    def _load(self) -> WorkflowObligation | None:
        try:
            return WorkflowObligation.model_validate_json((self.run_dir / WORKFLOW_OBLIGATION_ARTIFACT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def persist(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / WORKFLOW_OBLIGATION_ARTIFACT).write_text(
            self._obligation.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


def _denial(obligation: WorkflowObligation, message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_code": "denied_by_orchestration_intent",
            "message": message,
            "required_mode": obligation.mode.value,
            "required_preset": obligation.required_preset,
            "obligation_status": obligation.status.value,
        },
        ensure_ascii=False,
    )
