"""Server-side registration for durable Swarm worker artifacts.

The worker reports candidate paths after it completes.  This module turns only
safe, regular files inside that worker's own artifact directory into typed,
content-addressed records.  It intentionally does *not* grant another worker
read access; dependency-scoped transport is a later runtime concern.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.swarm.models import ArtifactRef
from src.swarm.worker import agent_artifact_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_id(task_id: str, relative_path: str) -> str:
    """Return a stable opaque ID for a producer/task path pair."""
    seed = f"{task_id}\0{relative_path}".encode("utf-8")
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


def register_task_artifacts(
    *,
    run_dir: Path,
    task_id: str,
    agent_id: str,
    artifact_paths: list[str],
    execution_identity_hash: str | None,
) -> list[ArtifactRef]:
    """Validate and content-address a completed worker's artifact candidates.

    Invalid, stale, symlinked, duplicate, or cross-worker candidates are
    omitted rather than becoming durable task state.  The return order is
    deterministic and all paths are normalized to POSIX run-relative form.
    """
    run_root = run_dir.resolve()
    producer_root = agent_artifact_dir(run_dir, agent_id).resolve()
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
                    artifact_id=_artifact_id(task_id, relative_path),
                    producer_task_id=task_id,
                    producer_agent_id=agent_id,
                    run_relative_path=relative_path,
                    sha256=_sha256_file(resolved),
                    byte_size=resolved.stat().st_size,
                    execution_identity_hash=execution_identity_hash,
                )
            )
        except OSError:
            # A concurrently removed/changed file is not an authoritative
            # artifact.  A later reader must never receive a stale record.
            continue
    return refs
