"""Authoritative, server-owned execution identity for market workflows.

This module deliberately contains no model-facing parsing or tool dispatch.
The AgentLoop creates identities from current-task evidence, resolvers append
same-source facts, and execution boundaries consume immutable snapshots.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionMode(str, Enum):
    GENERIC = "generic"
    SOURCE_SCOPED = "source_scoped"


class ExecutionIdentityStatus(str, Enum):
    UNRESOLVED = "unresolved"
    PARTIALLY_SPECIFIED = "partially_specified"
    RESOLVING = "resolving"
    VERIFIED = "verified"
    CONFLICT = "conflict"
    REJECTED = "rejected"


class SourceMode(str, Enum):
    AUTO = "auto"
    STRICT = "strict"


class FallbackPolicy(str, Enum):
    DENY = "deny"
    ALLOW = "allow"


class SyntheticDataPolicy(str, Enum):
    FORBID = "forbid"
    EXPLICIT_ONLY = "explicit_only"
    ALLOW_CONTROL = "allow_control"


class ProvenanceAuthority(str, Enum):
    CURRENT_USER = "current_user"
    CURRENT_ATTACHMENT = "current_attachment"
    RUNTIME_CONFIGURATION = "runtime_configuration"
    SOURCE_RESOLVER = "source_resolver"
    SESSION_EVIDENCE = "session_evidence"
    PERSISTENT_MEMORY = "persistent_memory"
    MODEL_SUGGESTION = "model_suggestion"


class AttachmentAssertionType(str, Enum):
    REQUIREMENT = "requirement"
    PREFERENCE = "preference"
    REFERENCE = "reference"


class TimeframeRole(str, Enum):
    DATA = "data"
    ANALYSIS = "analysis"
    EXECUTION = "execution"


class IdentityProvenance(BaseModel):
    """Non-volatile explanation of an execution-critical field's origin."""

    model_config = ConfigDict(frozen=True)

    authority: ProvenanceAuthority
    origin: str
    evidence_ref: str | None = None
    assertion_type: AttachmentAssertionType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TimeframeRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    role: TimeframeRole = TimeframeRole.DATA


class ExecutionRequest(BaseModel):
    """Immutable current-task request, before provider-native resolution."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    asset: str | None = None
    symbol: str | None = None
    source: str | None = None
    platform: str | None = None
    timeframes: tuple[TimeframeRequirement, ...] = ()
    provenance: tuple[IdentityProvenance, ...] = ()


class ExecutionResolution(BaseModel):
    """A verified same-source resolution for exactly one request."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    canonical_asset: str | None = None
    resolved_symbol: str | None = None
    source: str | None = None
    broker: str | None = None
    venue: str | None = None
    asset_class: str | None = None
    market: str | None = None
    resolver_evidence_ref: str
    provenance: tuple[IdentityProvenance, ...] = ()
    status: ExecutionIdentityStatus = ExecutionIdentityStatus.VERIFIED


class ExecutionPolicy(BaseModel):
    """Frozen data-source and synthetic-data permissions for one identity."""

    model_config = ConfigDict(frozen=True)

    source_mode: SourceMode = SourceMode.AUTO
    fallback: FallbackPolicy = FallbackPolicy.ALLOW
    cross_source_fallback: bool = True
    synthetic: SyntheticDataPolicy = SyntheticDataPolicy.EXPLICIT_ONLY

    @model_validator(mode="after")
    def strict_source_is_fail_closed(self) -> "ExecutionPolicy":
        if self.source_mode is SourceMode.STRICT and (
            self.fallback is not FallbackPolicy.DENY or self.cross_source_fallback
        ):
            raise ValueError("strict source mode requires fallback=deny and cross_source_fallback=false")
        return self


class IdentityConflict(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    existing_value: str | None = None
    incoming_value: str | None = None
    provenance: IdentityProvenance
    message: str


class ExecutionIdentity(BaseModel):
    """Immutable revision of the authoritative execution identity.

    Callers create a new revision through :meth:`with_resolution` or
    :meth:`with_conflict`; they must never mutate a verified snapshot in place.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = "execution-identity/v1"
    identity_id: str
    revision: int = Field(default=1, ge=1)
    mode: ExecutionMode = ExecutionMode.GENERIC
    status: ExecutionIdentityStatus = ExecutionIdentityStatus.UNRESOLVED
    requests: tuple[ExecutionRequest, ...] = ()
    resolutions: tuple[ExecutionResolution, ...] = ()
    policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    conflicts: tuple[IdentityConflict, ...] = ()
    provenance: tuple[IdentityProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_request_resolution_graph(self) -> "ExecutionIdentity":
        request_ids = [request.request_id for request in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("execution identity request_id values must be unique")
        known = set(request_ids)
        seen_resolutions: set[str] = set()
        request_by_id = {request.request_id: request for request in self.requests}
        for resolution in self.resolutions:
            if resolution.request_id not in known:
                raise ValueError("execution resolution references an unknown request_id")
            if resolution.request_id in seen_resolutions:
                raise ValueError("each request_id may have exactly one resolution per identity revision")
            seen_resolutions.add(resolution.request_id)
            request = request_by_id[resolution.request_id]
            if (
                self.policy.source_mode is SourceMode.STRICT
                and request.source
                and resolution.source
                and request.source.casefold() != resolution.source.casefold()
            ):
                raise ValueError("strict source resolution must remain in the requested source namespace")
        if self.status is ExecutionIdentityStatus.VERIFIED and not self.resolutions:
            raise ValueError("verified execution identity requires at least one resolution")
        return self

    @property
    def identity_hash(self) -> str:
        """Return a deterministic digest excluding volatile evidence details."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def canonical_json(self) -> str:
        return json.dumps(
            _canonicalize(self.model_dump(mode="json")),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def with_resolution(self, resolution: ExecutionResolution) -> "ExecutionIdentity":
        """Append one verified resolution as a new immutable revision."""
        if any(item.request_id == resolution.request_id for item in self.resolutions):
            raise ValueError("resolved request_id cannot be silently overwritten")
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "revision": self.revision + 1,
                "status": ExecutionIdentityStatus.VERIFIED,
                "resolutions": (*self.resolutions, resolution),
            }
        )

    def with_conflict(self, conflict: IdentityConflict) -> "ExecutionIdentity":
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "revision": self.revision + 1,
                "status": ExecutionIdentityStatus.CONFLICT,
                "conflicts": (*self.conflicts, conflict),
            }
        )

    def with_status(self, status: ExecutionIdentityStatus) -> "ExecutionIdentity":
        """Return a new revision for a terminal or resolving state transition."""
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "revision": self.revision + 1,
                "status": status,
            }
        )


def _canonicalize(value: Any) -> Any:
    """Strip volatile provenance fields before hashing an identity revision."""
    if isinstance(value, dict):
        return {
            key: _canonicalize(item)
            for key, item in sorted(value.items())
            if key not in {"evidence_ref", "resolver_evidence_ref"}
        }
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


__all__ = [
    "AttachmentAssertionType",
    "ExecutionIdentity",
    "ExecutionIdentityStatus",
    "ExecutionMode",
    "ExecutionPolicy",
    "ExecutionRequest",
    "ExecutionResolution",
    "FallbackPolicy",
    "IdentityConflict",
    "IdentityProvenance",
    "ProvenanceAuthority",
    "SourceMode",
    "SyntheticDataPolicy",
    "TimeframeRequirement",
    "TimeframeRole",
]
