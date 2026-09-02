"""PR-E: quantitative consumers must be gated by typed evidence, never prose."""
from __future__ import annotations

import threading
import hashlib
from datetime import datetime, timezone

import src.swarm.runtime as rt
from src.swarm.models import ArtifactRef, ArtifactRequirement, RunStatus, SwarmAgentSpec, SwarmRun, SwarmTask, TaskStatus, WorkerResult
from src.swarm.store import SwarmStore
from src.swarm.task_store import TaskStore


_TYPES = ["backtest.execution_provenance", "backtest.config", "backtest.metrics", "backtest.equity", "backtest.trades"]


def _ref(run_id: str, kind: str, *, valid=True, identity="ih", text="x") -> ArtifactRef:
    return ArtifactRef(artifact_id="id-" + kind, producer_run_id=run_id, producer_task_id="task-backtest", producer_agent_id="backtester", run_relative_path="artifacts/backtester/" + kind, sha256=hashlib.sha256(text.encode()).hexdigest(), byte_size=len(text), execution_identity_hash=identity, artifact_type=kind, provenance_status="passed" if valid else "unclassified")


def _run(tmp_path, refs):
    store=SwarmStore(base_dir=tmp_path); runtime=rt.SwarmRuntime(store=store)
    agents=[SwarmAgentSpec(id="backtester",role="b",system_prompt="x",max_retries=0),SwarmAgentSpec(id="risk",role="r",system_prompt="x",max_retries=0)]
    upstream=SwarmTask(id="task-backtest",agent_id="backtester",prompt_template="x",status=TaskStatus.completed,summary="metrics.csv generated successfully",artifact_refs=refs)
    risk=SwarmTask(id="task-risk",agent_id="risk",prompt_template="x",depends_on=["task-backtest"],blocked_by=[],input_from={"backtest":"task-backtest"},artifact_requirements=[ArtifactRequirement(producer_task_id="task-backtest",artifact_type=t) for t in _TYPES])
    run=SwarmRun(id="r",preset_name="quant_scalp_desk",status=RunStatus.running,created_at=datetime.now(timezone.utc).isoformat(),agents=agents,tasks=[upstream,risk],identity_hash="ih")
    rd=store.create_run(run)
    for ref in refs:
        p=rd/ref.run_relative_path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text("x")
    task_store=TaskStore(store.run_dir("r")/"tasks")
    task_store.save_task(upstream); task_store.save_task(risk)
    return store,runtime,run


def test_summary_cannot_satisfy_missing_artifacts_before_dispatch(tmp_path, monkeypatch):
    store,runtime,run=_run(tmp_path, []); calls=[]
    monkeypatch.setattr(rt,"run_worker",lambda *a,**k: calls.append(a) or WorkerResult(status="completed",summary="bad"))
    runtime._execute_layer(run=run,task_store=TaskStore(store.run_dir("r")/"tasks"),agent_map={a.id:a for a in run.agents},layer_task_ids=["task-risk"],task_summaries={"task-backtest":"metrics.csv generated successfully"},run_dir=store.run_dir("r"),cancel_event=threading.Event(),include_shell_tools=False,grounding_block="")
    task=TaskStore(store.run_dir("r")/"tasks").load_task("task-risk"); assert calls == []; assert task.status is TaskStatus.blocked; assert task.error.startswith("missing_required_artifacts")


def test_valid_refs_dispatch_even_when_summary_empty(tmp_path, monkeypatch):
    refs=[_ref("r",t) for t in _TYPES]; store,runtime,run=_run(tmp_path, refs); calls=[]
    monkeypatch.setattr(rt,"run_worker",lambda *a,**k: calls.append(a) or WorkerResult(status="completed",summary="ok"))
    runtime._execute_layer(run=run,task_store=TaskStore(store.run_dir("r")/"tasks"),agent_map={a.id:a for a in run.agents},layer_task_ids=["task-risk"],task_summaries={"task-backtest":""},run_dir=store.run_dir("r"),cancel_event=threading.Event(),include_shell_tools=False,grounding_block="")
    assert len(calls)==1


def test_mutated_registered_artifact_blocks_before_dispatch(tmp_path, monkeypatch):
    refs=[_ref("r",t) for t in _TYPES]; store,runtime,run=_run(tmp_path, refs); calls=[]
    (store.run_dir("r") / refs[0].run_relative_path).write_text("tampered")
    monkeypatch.setattr(rt,"run_worker",lambda *a,**k: calls.append(a) or WorkerResult(status="completed",summary="bad"))
    runtime._execute_layer(run=run,task_store=TaskStore(store.run_dir("r")/"tasks"),agent_map={a.id:a for a in run.agents},layer_task_ids=["task-risk"],task_summaries={},run_dir=store.run_dir("r"),cancel_event=threading.Event(),include_shell_tools=False,grounding_block="")
    assert calls == []
    assert TaskStore(store.run_dir("r")/"tasks").load_task("task-risk").status is TaskStatus.blocked
