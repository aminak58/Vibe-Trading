"""Server-side registration for durable Swarm worker artifacts.

The worker reports candidate paths after it completes.  This module turns only
safe, regular files inside that worker's own artifact directory into typed,
content-addressed records.  It intentionally does *not* grant another worker
read access; dependency-scoped transport is a later runtime concern.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Mapping

from src.swarm.models import ArtifactManifest, ArtifactRef
from src.agent.tools import BaseTool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_id(task_id: str, relative_path: str, *, run_id: str = "", attempt_id: str = "", generation_id: str = "") -> str:
    """Return a stable opaque ID for a producer/task path pair."""
    seed = f"{run_id}\0{task_id}\0{attempt_id}\0{generation_id}\0{relative_path}".encode("utf-8")
    return "artifact-" + hashlib.sha256(seed).hexdigest()[:24]


def _has_symlink_component(root: Path, relative: Path) -> bool:
    """Return whether *relative* reaches a file through any symlink."""
    current = root
    for part in relative.parts:
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _producer_artifact_dir(run_dir: Path, agent_id: str) -> Path:
    """Resolve the one permissible artifact directory for an agent.

    Kept here instead of importing ``worker.agent_artifact_dir`` so this
    server-side module can also provide the private reader that workers load.
    The worker's pre-existing helper remains the writer-side authority.
    """
    if not agent_id or agent_id in {".", ".."} or Path(agent_id).name != agent_id:
        raise ValueError(f"Invalid swarm agent id {agent_id!r}")
    run_root = run_dir.resolve()
    artifact_root = (run_root / "artifacts").resolve()
    candidate = artifact_root / agent_id
    resolved = candidate.resolve()
    if not resolved.is_relative_to(artifact_root) or resolved.parent != artifact_root:
        raise ValueError(f"Invalid swarm agent id {agent_id!r}: artifact path escapes root")
    return resolved


def register_task_artifacts(
    *,
    run_dir: Path,
    task_id: str,
    agent_id: str,
    artifact_paths: list[str],
    execution_identity_hash: str | None,
    producer_attempt_id: str | None = None,
    manifest_generation_id: str | None = None,
) -> list[ArtifactRef]:
    """Validate and content-address a completed worker's artifact candidates.

    Invalid, stale, symlinked, duplicate, or cross-worker candidates are
    omitted rather than becoming durable task state.  The return order is
    deterministic and all paths are normalized to POSIX run-relative form.
    """
    run_root = run_dir.resolve()
    producer_root = _producer_artifact_dir(run_dir, agent_id)
    artifact_root = (run_root / "artifacts").resolve()
    if not producer_root.is_relative_to(artifact_root):
        raise ValueError("producer artifact directory escapes swarm artifact root")

    refs: list[ArtifactRef] = []
    seen: set[str] = set()
    for raw_path in sorted(set(artifact_paths or [])):
        if not isinstance(raw_path, str) or not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        # Candidate paths are persisted in POSIX form.  Resolving through the
        # run root also rejects a model-provided ../ escape.
        try:
            if _has_symlink_component(run_root, candidate):
                continue
            resolved = (run_root / candidate).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        try:
            if (
                not resolved.is_relative_to(producer_root)
                or not resolved.is_relative_to(run_root)
                or not resolved.is_file()
                or resolved.is_symlink()
            ):
                continue
        except OSError:
            continue

        relative_path = resolved.relative_to(run_root).as_posix()
        if relative_path in seen:
            continue
        seen.add(relative_path)
        try:
            refs.append(
                ArtifactRef(
                    artifact_id=_artifact_id(task_id, relative_path, run_id=run_root.name, attempt_id=producer_attempt_id or "legacy", generation_id=manifest_generation_id or "legacy"),
                    producer_run_id=run_root.name,
                    producer_task_id=task_id,
                    producer_agent_id=agent_id,
                    run_relative_path=relative_path,
                    sha256=_sha256_file(resolved),
                    byte_size=resolved.stat().st_size,
                    execution_identity_hash=execution_identity_hash,
                    producer_attempt_id=producer_attempt_id,
                    manifest_generation_id=manifest_generation_id,
                )
            )
        except OSError:
            # A concurrently removed/changed file is not an authoritative
            # artifact.  A later reader must never receive a stale record.
            continue
    return refs


def finalize_artifact_generation(*, run_dir: Path, task_id: str, identity_hash: str | None, refs: list[ArtifactRef]) -> tuple[ArtifactManifest, list[ArtifactRef]]:
    """Atomically publish one validated producer generation as current."""
    attempt_id = "attempt-" + uuid.uuid4().hex
    generation_id = "generation-" + uuid.uuid4().hex
    refreshed = [ref.model_copy(update={
        "artifact_id": _artifact_id(task_id, ref.run_relative_path, run_id=run_dir.name, attempt_id=attempt_id, generation_id=generation_id),
        "producer_attempt_id": attempt_id, "manifest_generation_id": generation_id,
    }) for ref in refs]
    manifest = ArtifactManifest(generation_id=generation_id, producer_attempt_id=attempt_id, producer_task_id=task_id, run_id=run_dir.name, identity_hash=identity_hash, artifact_ids=[r.artifact_id for r in refreshed])
    root = run_dir / "artifact_manifests"; root.mkdir(parents=True, exist_ok=True)
    path = root / f"{task_id}.json"; tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2)+"\n", encoding="utf-8"); tmp.replace(path)
    return manifest, refreshed


def current_artifact_manifest(run_dir: Path, task_id: str) -> ArtifactManifest | None:
    try:
        return ArtifactManifest.model_validate_json((run_dir / "artifact_manifests" / f"{task_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_registered_artifact(run_dir: Path, ref: ArtifactRef) -> bool:
    """Return true only while a registered ref still names the same safe bytes."""
    try:
        relative = Path(ref.run_relative_path)
        if relative.is_absolute() or ".." in relative.parts or _has_symlink_component(run_dir.resolve(), relative):
            return False
        path = (run_dir.resolve() / relative).resolve(strict=True)
        root = _producer_artifact_dir(run_dir.resolve(), ref.producer_agent_id)
        return path.is_file() and path.is_relative_to(root) and path.stat().st_size == ref.byte_size and _sha256_file(path) == ref.sha256
    except (OSError, RuntimeError, ValueError):
        return False


class ReadDependencyArtifactTool(BaseTool):
    """Read a server-authorized immutable artifact from a declared DAG edge.

    This tool is constructed privately for an individual worker.  Its opaque
    IDs come from the runtime's task ``input_from`` map; callers cannot pass a
    filesystem path, producer task, or another run ID.
    """

    name = "read_dependency_artifact"
    description = (
        "Read one immutable artifact supplied by a declared upstream task. "
        "Use an artifact_id from the Upstream Artifact Manifest; paths and "
        "other workers' artifacts are not accepted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Opaque artifact ID from the Upstream Artifact Manifest.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Zero-based line offset (default 0).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5000,
                "description": "Maximum text lines to return (default 500).",
            },
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }
    repeatable = True
    is_readonly = True

    def __init__(
        self,
        *,
        run_dir: Path,
        allowed_refs: Mapping[str, ArtifactRef],
        execution_identity_hash: str | None,
    ) -> None:
        self._run_dir = run_dir.resolve()
        self._allowed_refs = dict(allowed_refs)
        self._execution_identity_hash = execution_identity_hash

    def _error(self, code: str, message: str) -> str:
        return json.dumps({"status": "error", "error_code": code, "error": message}, ensure_ascii=False)

    def _resolve_ref(self, ref: ArtifactRef) -> Path | None:
        relative = Path(ref.run_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        if _has_symlink_component(self._run_dir, relative):
            return None
        try:
            path = (self._run_dir / relative).resolve(strict=True)
            producer_root = _producer_artifact_dir(self._run_dir, ref.producer_agent_id)
            if not path.is_relative_to(producer_root) or not path.is_file() or path.is_symlink():
                return None
            return path
        except (OSError, RuntimeError, ValueError):
            return None

    def execute(self, **kwargs: object) -> str:
        artifact_id = kwargs.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return self._error("invalid_argument", "artifact_id must be a non-empty string")
        ref = self._allowed_refs.get(artifact_id)
        if ref is None:
            return self._error("dependency_artifact_denied", "Artifact is not authorized for this task")
        if ref.producer_run_id != self._run_dir.name:
            return self._error("dependency_artifact_denied", "Artifact belongs to another swarm run")
        if (
            self._execution_identity_hash is not None
            and ref.execution_identity_hash != self._execution_identity_hash
        ):
            return self._error("provenance_conflict", "Artifact identity hash does not match this execution")

        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 500)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return self._error("invalid_argument", "offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            return self._error("invalid_argument", "limit must be an integer from 1 through 5000")

        path = self._resolve_ref(ref)
        if path is None:
            return self._error("dependency_artifact_invalid", "Authorized artifact path is missing or invalid")
        try:
            if path.stat().st_size != ref.byte_size or _sha256_file(path) != ref.sha256:
                return self._error("provenance_conflict", "Artifact content no longer matches its registered hash")
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return self._error("unsupported_artifact", "Artifact is binary and cannot be read as text")
        except OSError:
            return self._error("dependency_artifact_invalid", "Authorized artifact could not be read")

        lines = content.splitlines(keepends=True)
        selected = "".join(lines[offset : offset + limit])
        return json.dumps(
            {
                "status": "ok",
                "artifact": ref.model_dump(),
                "offset": offset,
                "line_count": len(lines),
                "content": selected,
            },
            ensure_ascii=False,
        )
