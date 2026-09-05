"""Server-owned obligations for an explicitly requested workflow topology.

Execution identity answers *what* may execute.  This module deliberately
answers only *how* the current user required it to execute.  It never reads
memory, assembled prompts, or model-generated prose.
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

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


class PresetRequirementKind(str, Enum):
    """How a current user constrained the required Swarm preset."""

    NONE = "none"
    EXACT = "exact"
    CAPABILITY = "capability"


class PresetRequirement(BaseModel):
    """Bounded, current-user-only preset requirement.

    ``CAPABILITY`` intentionally contains no prose-derived provider, market,
    or preset name.  The only P0 capability directive is an existing preset
    metadata field: source-scoped execution.  Identity compatibility is
    evaluated separately against the verified server-owned identity.
    """

    model_config = ConfigDict(frozen=True)

    kind: PresetRequirementKind = PresetRequirementKind.NONE
    exact_name: str | None = None
    source_scoped_execution: bool = False


class WorkflowObligation(BaseModel):
    """Immutable-ish persisted state for a current-user workflow requirement."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "workflow-obligation/v3"
    mode: WorkflowMode = WorkflowMode.NONE
    preset_requirement: PresetRequirement = PresetRequirement()
    # ``requested`` is the literal exact user constraint, if any.  A
    # capability directive deliberately has no requested preset; its selected
    # preset is a later, server-side result that remains auditable on restart.
    requested_preset: str | None = None
    selected_preset: str | None = None
    selection_reason: str | None = None
    authority: str = "current_user"
    status: WorkflowStatus = WorkflowStatus.PENDING
    identity_hash: str | None = None
    swarm_run_id: str | None = None
    dispatch_attempted: bool = False
    launch_id: str | None = None
    terminal_result_status: str | None = None
    terminal_reason: str | None = None
    evidence: str = "current_raw_user_message"

    @property
    def required_preset(self) -> str | None:
        """Compatibility accessor for exact-preset callers and diagnostics."""
        return self.requested_preset

    def transition(self, status: WorkflowStatus, **updates: str | None) -> "WorkflowObligation":
        return self.model_copy(update={"status": status, **updates})


# This deliberately has a narrow vocabulary.  Do not turn ordinary historic
# prose, summaries, or broad words such as "analysis" into an obligation.
_ACTION = r"(?:run|use|start|launch|execute)"
_SWARM_DIRECTIVE_RE = re.compile(
    rf"\b{_ACTION}\b(?:\s+(?:this|it|the\s+(?:task|request|workflow)))?\s+(?:(?:with|through)\s+)?(?:a\s+)?swarm\b(?:\s+(?:workflow|team|preset))?",
    re.IGNORECASE,
)
_CAPABILITY_DIRECTIVE_RE = re.compile(
    rf"\b{_ACTION}\b(?:\s+(?:this|it|the\s+(?:task|request|workflow)))?\s+(?:with\s+)?(?:the\s+|a\s+)?(?:strict-capable|source-aware)\s+swarm\s+preset\b",
    re.IGNORECASE,
)
_PRESET_DIRECTIVE_RE = re.compile(
    rf"\b{_ACTION}\b(?:\s+(?:this|it|the\s+(?:task|request|workflow)))?\s+(?:with\s+)?(?:the\s+)?(quant_scalp_desk)\b",
    re.IGNORECASE,
)
_NEGATED_SWARM_RE = re.compile(r"\b(?:do\s+not|don't|without|avoid|never)\s+(?:(?:use|run|start|launch|execute)\s+)?(?:the\s+)?swarm\b", re.IGNORECASE)
_NON_DIRECTIVE_SWARM_RE = re.compile(
    r"\b(?:should\s+(?:we|i)\s+use|may\s+be\s+useful|previous|last|failed\s+yesterday)\s+(?:the\s+)?swarm\b"
    r"|\bswarm\s+(?:may\s+be\s+useful|failed\s+yesterday)\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"(?P<quote>['\"])(?:\\.|(?!\1).)*(?P=quote)")
_CODE_SPAN_RE = re.compile(r"`[^`]*`")
_BLOCK_QUOTE_RE = re.compile(r"(?m)^\s*>.*$")


def _directive_text(user_message: str) -> str:
    """Remove quoted examples; only an affirmative current instruction counts."""
    text = user_message or ""
    text = _BLOCK_QUOTE_RE.sub(" ", text)
    text = _CODE_SPAN_RE.sub(" ", text)
    return _QUOTED_RE.sub(" ", text)


def obligation_from_current_user_message(user_message: str) -> WorkflowObligation:
    """Parse only this turn's raw user message into a bounded obligation."""
    text = _directive_text(user_message)
    if _NEGATED_SWARM_RE.search(text) or _NON_DIRECTIVE_SWARM_RE.search(text):
        return WorkflowObligation(mode=WorkflowMode.NONE)
    preset_match = _PRESET_DIRECTIVE_RE.search(text)
    if preset_match:
        return WorkflowObligation(
            mode=WorkflowMode.SWARM_REQUIRED,
            preset_requirement=PresetRequirement(
                kind=PresetRequirementKind.EXACT,
                exact_name=preset_match.group(1).casefold(),
            ),
            requested_preset=preset_match.group(1).casefold(),
            selected_preset=preset_match.group(1).casefold(),
            selection_reason="explicit_user_request",
        )
    if _CAPABILITY_DIRECTIVE_RE.search(text):
        return WorkflowObligation(
            mode=WorkflowMode.SWARM_REQUIRED,
            preset_requirement=PresetRequirement(
                kind=PresetRequirementKind.CAPABILITY,
                source_scoped_execution=True,
            ),
        )
    if _SWARM_DIRECTIVE_RE.search(text):
        return WorkflowObligation(mode=WorkflowMode.SWARM_REQUIRED)
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
            if self._obligation.swarm_run_id:
                return
            self._obligation = self._obligation.transition(WorkflowStatus.FAILED)
            self.persist()
            return
        # A denied invocation is not a result of the owning execution.
        if isinstance(payload, dict) and payload.get("error_code") == "denied_by_orchestration_intent":
            return
        if self._obligation.swarm_run_id and (
            not isinstance(payload, dict)
            or payload.get("run_id") != self._obligation.swarm_run_id
        ):
            return
        if not isinstance(payload, dict):
            self._obligation = self._obligation.transition(WorkflowStatus.FAILED)
        else:
            status = str(payload.get("status") or "").casefold()
            run_id = str(payload.get("run_id") or "") or None
            if self._obligation.swarm_run_id and run_id and run_id != self._obligation.swarm_run_id:
                return
            if payload.get("wait_budget_exhausted") or status in {"pending", "running"}:
                self._obligation = self._obligation.transition(WorkflowStatus.WAITING, swarm_run_id=run_id)
            elif status == "completed":
                self._obligation = self._obligation.transition(WorkflowStatus.COMPLETED, swarm_run_id=run_id, terminal_result_status="completed")
            elif status in {"cancelled", "canceled"}:
                self._obligation = self._obligation.transition(WorkflowStatus.CANCELLED, swarm_run_id=run_id, terminal_result_status="cancelled")
            else:
                self._obligation = self._obligation.transition(WorkflowStatus.FAILED, swarm_run_id=run_id, terminal_result_status="failed")
        self.persist()

    def mark_swarm_started(self) -> None:
        """Record dispatch before a long-running Swarm tool returns."""
        if self._obligation.mode is WorkflowMode.SWARM_REQUIRED and not self._obligation.dispatch_attempted:
            self._obligation = self._obligation.transition(WorkflowStatus.SWARM_STARTED, dispatch_attempted=True, launch_id="launch-" + uuid.uuid4().hex)
            self.persist()

    def bind_swarm_run(self, run_id: str) -> None:
        if self._obligation.mode is WorkflowMode.SWARM_REQUIRED and self._obligation.dispatch_attempted and not self._obligation.swarm_run_id:
            self._obligation = self._obligation.transition(self._obligation.status, swarm_run_id=run_id)
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

    def prepare_run_swarm(self, arguments: dict[str, Any], identity: Any | None) -> str | None:
        """Resolve a bounded capability requirement into one server-selected preset.

        The model never supplies the selection.  A capability directive needs
        a verified identity so preset metadata can be checked deterministically.
        """
        requirement = self._obligation.preset_requirement
        if requirement.kind is not PresetRequirementKind.CAPABILITY:
            return None
        if identity is None or getattr(getattr(identity, "status", None), "value", None) != "verified":
            return _denial(self._obligation, "Resolve the execution identity before selecting a capability-required Swarm preset.")
        from src.swarm.presets import resolve_source_scoped_preset

        selected, error = resolve_source_scoped_preset(
            identity,
            require_synthetic_forbid=requirement.source_scoped_execution,
        )
        if error:
            return _denial(self._obligation, error)
        assert selected is not None
        supplied = str(arguments.get("preset_name") or "").casefold() or None
        if supplied is not None and supplied != selected:
            return _denial(self._obligation, "The capability-required Swarm preset is selected server-side and cannot be replaced.")
        self._obligation = self._obligation.transition(
            self._obligation.status,
            selected_preset=selected,
            selection_reason="capability_match",
        )
        self.persist()
        arguments["preset_name"] = selected
        return None

    def block(self, tool_name: str, arguments: dict) -> str | None:
        """Return a structured denial for a topology-changing tool call."""
        obligation = self._obligation
        if obligation.mode is not WorkflowMode.SWARM_REQUIRED:
            return None
        if tool_name == "run_swarm":
            if obligation.dispatch_attempted:
                return _denial(obligation, "This current-user Swarm objective has already consumed its one dispatch attempt.")
            supplied = str(arguments.get("preset_name") or "").casefold() or None
            if obligation.preset_requirement.kind is PresetRequirementKind.EXACT and obligation.required_preset and supplied != obligation.required_preset:
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
            raw = json.loads((self.run_dir / WORKFLOW_OBLIGATION_ARTIFACT).read_text(encoding="utf-8"))
            # Preserve the only v1 constraint while safely loading a live
            # running/waiting Swarm after deployment.  New artifacts always
            # persist v2's requested/selected split.
            if isinstance(raw, dict) and raw.get("schema_version") == "workflow-obligation/v1":
                exact = raw.get("required_preset")
                if exact:
                    raw.update(
                        {
                            "schema_version": "workflow-obligation/v2",
                            "preset_requirement": {
                                "kind": PresetRequirementKind.EXACT.value,
                                "exact_name": str(exact).casefold(),
                                "source_scoped_execution": False,
                            },
                            "requested_preset": str(exact).casefold(),
                            "selected_preset": str(exact).casefold(),
                            "selection_reason": "legacy_v1_exact_preset",
                        }
                    )
            return WorkflowObligation.model_validate(raw)
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
            "preset_requirement": obligation.preset_requirement.model_dump(mode="json"),
            "obligation_status": obligation.status.value,
        },
        ensure_ascii=False,
    )
