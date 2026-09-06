"""Strict report binding admits only the strategy file actually executed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.execution_identity import (
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionResolution,
    FallbackPolicy,
    SourceMode,
    SyntheticDataPolicy,
)
from src.swarm.models import ArtifactManifest, ArtifactRef, RunStatus, SwarmRun, SwarmTask
from src.swarm.report_binding import build_executed_strategy_binding
from src.swarm.runtime import _classify_authoritative_artifacts


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-xau",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
            synthetic=SyntheticDataPolicy.FORBID,
        ),
        requests=(ExecutionRequest(request_id="xau", symbol="XAUUSD", source="mt5"),),
        resolutions=(ExecutionResolution(
            request_id="xau", resolved_symbol="XAUUSD_o", source="mt5", resolver_evidence_ref="resolver:1"
        ),),
    )


def _write_ref(
    root: Path, *, artifact_id: str, path: str, text: str, generation: str = "G1"
) -> ArtifactRef:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))
    return ArtifactRef(
        artifact_id=artifact_id,
        producer_run_id="strict-strategy-run",
        producer_task_id="task-backtest",
        producer_agent_id="backtester",
        run_relative_path=path,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        byte_size=len(text.encode()),
        execution_identity_hash="",
        artifact_type="worker.file",
        provenance_status="unclassified",
        producer_attempt_id="attempt-1",
        manifest_generation_id=generation,
    )


def _strict_backtest_bundle(tmp_path: Path) -> tuple[SwarmRun, SwarmTask, list[ArtifactRef], ArtifactRef, ArtifactRef]:
    identity = _identity()
    task = SwarmTask(id="task-backtest", agent_id="backtester", prompt_template="x")
    run = SwarmRun(
        id="strict-strategy-run",
        preset_name="quant_scalp_desk",
        status=RunStatus.running,
        created_at="2026-09-06T00:00:00+00:00",
        execution_identity=identity,
        identity_hash=identity.identity_hash,
        tasks=[task],
    )
    strategy = '''class SignalEngine:
    """VWAP bands with BOS confirmation and ATR exits."""
'''
    config = json.dumps({
        "source": "mt5", "codes": ["XAUUSD_o"], "interval": "15m",
        "_execution_identity_hash": identity.identity_hash,
    })
    run_card = json.dumps({
        "reproducibility": {
            "strategy_hash": hashlib.sha256(strategy.encode()).hexdigest(),
            "config_hash": hashlib.sha256(config.encode()).hexdigest(),
        },
        "artifacts": [{"path": "code/signal_engine.py", "sha256": hashlib.sha256(strategy.encode()).hexdigest()}],
    })
    provenance = json.dumps({
        "identity_hash": identity.identity_hash,
        "owning_run_id": run.id,
        "requested_symbol": "XAUUSD",
        "resolved_symbol": "XAUUSD_o",
        "effective_source": "mt5",
        "synthetic": False,
        "fallback_used": False,
    })
    refs = [
        _write_ref(tmp_path, artifact_id="provenance", path="artifacts/backtester/execution_provenance.json", text=provenance),
        _write_ref(tmp_path, artifact_id="config", path="artifacts/backtester/config.json", text=config),
        _write_ref(tmp_path, artifact_id="run-card", path="artifacts/backtester/run_card.json", text=run_card),
    ]
    canonical = _write_ref(tmp_path, artifact_id="canonical-strategy", path="artifacts/backtester/code/signal_engine.py", text=strategy)
    duplicate = _write_ref(tmp_path, artifact_id="worker-copy", path="artifacts/backtester/signal_engine.py", text=strategy)
    refs.extend((canonical, duplicate))
    refs = [ref.model_copy(update={"execution_identity_hash": identity.identity_hash}) for ref in refs]
    canonical = next(ref for ref in refs if ref.artifact_id == canonical.artifact_id)
    duplicate = next(ref for ref in refs if ref.artifact_id == duplicate.artifact_id)
    manifest_dir = tmp_path / "artifact_manifests"
    manifest_dir.mkdir()
    (manifest_dir / "task-backtest.json").write_text(
        ArtifactManifest(
            generation_id="G1", producer_attempt_id="attempt-1", producer_task_id=task.id,
            run_id=run.id, identity_hash=identity.identity_hash,
            artifact_ids=[ref.artifact_id for ref in refs],
        ).model_dump_json(),
        encoding="utf-8",
    )
    return run, task, refs, canonical, duplicate


def test_run_card_declared_strategy_is_the_only_strict_authoritative_strategy(tmp_path: Path) -> None:
    run, task, refs, canonical, duplicate = _strict_backtest_bundle(tmp_path)

    classified = _classify_authoritative_artifacts(run, task, refs, tmp_path)

    by_id = {ref.artifact_id: ref for ref in classified}
    assert by_id[canonical.artifact_id].artifact_type == "backtest.strategy"
    assert by_id[canonical.artifact_id].provenance_status == "passed"
    assert by_id[duplicate.artifact_id].artifact_type == "worker.file"
    assert by_id[duplicate.artifact_id].provenance_status == "unclassified"

    binding = build_executed_strategy_binding(
        run_dir=tmp_path,
        refs=classified,
        run_id=run.id,
        identity_hash=run.identity_hash,
    )
    assert binding["strategy_sha256"] == canonical.sha256


def test_multiple_canonical_strategy_refs_remain_an_explicit_report_block(tmp_path: Path) -> None:
    run, task, refs, _, _ = _strict_backtest_bundle(tmp_path)
    classified = _classify_authoritative_artifacts(run, task, refs, tmp_path)
    canonical = next(ref for ref in classified if ref.artifact_type == "backtest.strategy")

    with pytest.raises(ValueError, match="ambiguous executed artifact: backtest.strategy"):
        build_executed_strategy_binding(
            run_dir=tmp_path,
            refs=[*classified, canonical],
            run_id=run.id,
            identity_hash=run.identity_hash,
        )
