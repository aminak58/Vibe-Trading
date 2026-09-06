"""Strict final-strategy text must be bound to executed, registered artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.swarm.models import ArtifactManifest, ArtifactRef
from src.swarm.report_binding import (
    NarrativeMismatch,
    build_executed_strategy_binding,
    render_bound_strict_report,
)
from src.swarm.runtime import _is_strict_report_task
from src.swarm.models import RunStatus, SwarmRun, SwarmTask


def _ref(run_id: str, kind: str, path: str, text: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact-{kind}", producer_run_id=run_id,
        producer_task_id="task-backtest", producer_agent_id="backtester",
        run_relative_path=path, sha256=hashlib.sha256(text.encode()).hexdigest(),
        byte_size=len(text), execution_identity_hash="identity-1",
        artifact_type=kind, provenance_status="passed",
    )


def _bundle(tmp_path: Path) -> tuple[list[ArtifactRef], str]:
    run_id = "strict-run"
    config = json.dumps({
        "source": "mt5", "codes": ["XAUUSD_o"], "interval": "5m",
        "_execution_identity_hash": "identity-1",
    })
    strategy = '''class SignalEngine:
    """VWAP deviation bands with BOS confirmation and ATR risk controls."""
'''
    values = {
        "backtest.config": ("artifacts/backtester/config.json", config),
        "backtest.run_card": ("artifacts/backtester/run_card.json", json.dumps({
            "reproducibility": {
                "strategy_hash": hashlib.sha256(strategy.encode()).hexdigest(),
                "config_hash": hashlib.sha256(config.encode()).hexdigest(),
            },
        })),
        "backtest.execution_provenance": ("artifacts/backtester/execution_provenance.json", json.dumps({
            "identity_hash": "identity-1", "effective_source": "mt5", "synthetic": False,
        })),
        "backtest.strategy": ("artifacts/backtester/code/signal_engine.py", strategy),
    }
    refs: list[ArtifactRef] = []
    for kind, (path, text) in values.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
        refs.append(_ref(run_id, kind, path, text).model_copy(update={
            "producer_attempt_id": "attempt-G1", "manifest_generation_id": "G1",
        }))
    manifest_dir = tmp_path / "artifact_manifests"
    manifest_dir.mkdir()
    (manifest_dir / "task-backtest.json").write_text(
        ArtifactManifest(
            generation_id="G1", producer_attempt_id="attempt-G1",
            producer_task_id="task-backtest", run_id=run_id,
            identity_hash="identity-1", artifact_ids=[ref.artifact_id for ref in refs],
        ).model_dump_json(),
        encoding="utf-8",
    )
    return refs, run_id


def test_registered_executed_artifacts_render_canonical_final_strategy(tmp_path: Path) -> None:
    refs, run_id = _bundle(tmp_path)
    binding = build_executed_strategy_binding(
        run_dir=tmp_path, refs=refs, run_id=run_id, identity_hash="identity-1"
    )
    report = render_bound_strict_report(binding, "## Backtest Summary\nOfficial metrics only.")
    assert "## Final Strategy (Executed Contract)" in report
    assert "VWAP deviation bands with BOS confirmation and ATR risk controls." in report
    assert "XAUUSD_o" in report and "mt5" in report


def test_freeform_final_strategy_heading_is_rejected_in_strict_report(tmp_path: Path) -> None:
    refs, run_id = _bundle(tmp_path)
    binding = build_executed_strategy_binding(
        run_dir=tmp_path, refs=refs, run_id=run_id, identity_hash="identity-1"
    )
    with pytest.raises(NarrativeMismatch, match="Final Strategy"):
        render_bound_strict_report(binding, "## Final Strategy\nEMA(8/21) crossover with RSI filter.")


def test_identity_mismatched_executed_config_cannot_build_binding(tmp_path: Path) -> None:
    refs, run_id = _bundle(tmp_path)
    refs[0] = refs[0].model_copy(update={"execution_identity_hash": "other"})
    with pytest.raises(ValueError, match="identity"):
        build_executed_strategy_binding(run_dir=tmp_path, refs=refs, run_id=run_id, identity_hash="identity-1")


def test_missing_executed_strategy_artifact_cannot_build_binding(tmp_path: Path) -> None:
    refs, run_id = _bundle(tmp_path)
    with pytest.raises(ValueError, match="backtest.strategy"):
        build_executed_strategy_binding(
            run_dir=tmp_path,
            refs=[ref for ref in refs if ref.artifact_type != "backtest.strategy"],
            run_id=run_id,
            identity_hash="identity-1",
        )


def test_stale_executed_generation_cannot_build_binding(tmp_path: Path) -> None:
    refs, run_id = _bundle(tmp_path)
    refs[-1] = refs[-1].model_copy(update={"manifest_generation_id": "old-generation"})
    with pytest.raises(ValueError, match="generation"):
        build_executed_strategy_binding(run_dir=tmp_path, refs=refs, run_id=run_id, identity_hash="identity-1")


def test_quant_scalp_report_declares_all_executed_strategy_dependencies() -> None:
    preset = yaml.safe_load(
        (Path(__file__).parents[1] / "src" / "swarm" / "presets" / "quant_scalp_desk.yaml").read_text(encoding="utf-8")
    )
    report = next(task for task in preset["tasks"] if task["id"] == "task-report")
    requirements = {
        (item["producer_task_id"], item["artifact_type"])
        for item in report["artifact_requirements"]
    }
    assert {
        ("task-backtest", "backtest.config"),
        ("task-backtest", "backtest.run_card"),
        ("task-backtest", "backtest.execution_provenance"),
        ("task-backtest", "backtest.strategy"),
    }.issubset(requirements)


def test_non_strict_or_generic_report_is_not_subject_to_strict_binding() -> None:
    generic = SwarmRun(
        id="generic-run", preset_name="quant_strategy_desk", status=RunStatus.running,
        created_at="2026-09-06T00:00:00+00:00",
    )
    task = SwarmTask(id="task-report", agent_id="report_aggregator", prompt_template="x")
    assert _is_strict_report_task(generic, task) is False
