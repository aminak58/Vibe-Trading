"""Server-owned convergence of a terminal Swarm into its owning parent state."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.workflow_obligation import WorkflowObligationLedger
from src.config.paths import get_runs_dir, get_sessions_dir


logger = logging.getLogger(__name__)
_TERMINAL = {"completed", "failed", "cancelled", "canceled", "rejected"}


def _safe_component(value: object) -> str | None:
    if not isinstance(value, str) or not value or Path(value).name != value:
        return None
    return value


def _contained_child(root: Path, child: str) -> Path | None:
    try:
        resolved_root = root.resolve()
        candidate = (root / child).resolve()
        candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return candidate


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        logger.warning("could not persist terminal Swarm ownership reconciliation", exc_info=True)
        return False
    return True


def _status(run: Any) -> str:
    raw = getattr(getattr(run, "status", None), "value", None) or getattr(run, "status", "")
    return str(raw).casefold()


def finalize_terminal_owned_swarm(
    run: Any,
    *,
    runs_dir: Path | None = None,
    sessions_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Converge one terminal Swarm into exact, previously persisted ownership.

    The canonical run supplies only its run/session/launch identity.  The
    parent directory comes exclusively from the trusted session index and is
    containment-checked beneath the server-owned runs root.  This function
    never dispatches tools, retries work, or calls a model.
    """
    terminal_status = _status(run)
    if terminal_status not in _TERMINAL:
        return None
    run_id = _safe_component(getattr(run, "id", None))
    session_id = _safe_component(getattr(run, "owner_session_id", None))
    launch_id = _safe_component(getattr(run, "launch_id", None))
    if not run_id or not session_id or not launch_id:
        return None

    resolved_runs_dir = Path(runs_dir) if runs_dir is not None else get_runs_dir()
    resolved_sessions_dir = Path(sessions_dir) if sessions_dir is not None else get_sessions_dir()
    index_path = _contained_child(resolved_sessions_dir, session_id)
    if index_path is None:
        return None
    index_path = index_path / "swarm_ownership.json"
    index = _read_json(index_path)
    if index is None or (
        index.get("session_id") != session_id
        or index.get("swarm_run_id") != run_id
        or index.get("launch_id") != launch_id
    ):
        return None

    parent_run_id = _safe_component(index.get("parent_run_id"))
    if parent_run_id is None:
        return None
    parent_dir = _contained_child(resolved_runs_dir, parent_run_id)
    if parent_dir is None or not parent_dir.is_dir():
        return None
    ownership_path = parent_dir / "swarm_ownership.json"
    ownership = _read_json(ownership_path)
    if ownership is None or (
        ownership.get("owner_session_id") != session_id
        or ownership.get("run_id") != run_id
        or ownership.get("launch_id") != launch_id
    ):
        return None

    ledger = WorkflowObligationLedger(run_dir=parent_dir, user_message="")
    result = ledger.reconcile_owned_swarm(run, owner_session_id=session_id)
    if result is None:
        persisted_result = index.get("terminal_result")
        if (
            str(index.get("status") or "").casefold() == terminal_status
            and isinstance(persisted_result, dict)
            and persisted_result.get("status")
        ):
            return persisted_result
        return None

    # A terminal research failure may still follow an official completed
    # backtest.  Preserve only registered, provenance-passed backtest types so
    # recovery/UI consumers can distinguish that partial success from no
    # backtest having run, without treating worker prose as evidence.
    result["official_backtest_artifact_types"] = sorted(
        {
            str(getattr(ref, "artifact_type", ""))
            for task in getattr(run, "tasks", [])
            for ref in getattr(task, "artifact_refs", [])
            if str(getattr(ref, "artifact_type", "")).startswith("backtest.")
            and getattr(ref, "producer_run_id", None) == run_id
            and getattr(ref, "provenance_status", None) == "passed"
        }
    )

    result_status = str(result.get("status") or "").casefold()
    ownership_status = (
        "rejected"
        if terminal_status == "rejected"
        else "failed" if result_status == "completed_with_artifact_gap" else result_status
    )
    terminal_reason = str(result.get("terminal_reason") or result_status)
    ownership.update(status=ownership_status, terminal_reason=terminal_reason)
    if not _write_json(ownership_path, ownership):
        return None

    index.update(
        status=ownership_status,
        terminal_reason=terminal_reason,
        terminal_result=result,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    if not _write_json(index_path, index):
        return None
    return result
