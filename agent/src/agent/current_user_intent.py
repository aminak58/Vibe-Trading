"""Bounded parsing of execution-critical intent in one raw user turn.

This module intentionally recognizes only explicit, whitelisted source
phrases.  It does not inspect memory, assembled context, or model prose.
"""

from __future__ import annotations

import re


_EXPLICIT_SOURCE_RE = re.compile(r"\bsource\s*(?:=|:)\s*['\"]?([a-z0-9_-]+)", re.IGNORECASE)
_PERSIAN_MT5_SOURCE_RE = re.compile(
    r"(?:"
    r"\u062f\u0627\u062f\u0647(?:\u0654|\u200c)?\s*(?:mt5|metatrader|\u0645\u062a\u0627\u062a\u0631\u06cc\u062f\u0631)(?:\s*(?:5|\u06f5))?"
    r"|\u0627\u0632\s+mt5(?:\s+\u0628\u0631\u0627\u06cc\s+\S+)?\s+\u0627\u0633\u062a\u0641\u0627\u062f\u0647\s+\u06a9\u0646"
    r")",
    re.IGNORECASE,
)


def explicit_source_from_current_user_message(user_message: str) -> str | None:
    """Return one explicit source from the raw current user request.

    Existing ``source=provider`` syntax remains general.  The only
    natural-language alias admitted in this bounded P0 grammar is MT5; it is
    mapped to the repository's canonical source identifier rather than a
    model-inferred provider name.
    """
    text = user_message or ""
    match = _EXPLICIT_SOURCE_RE.search(text)
    if match:
        return match.group(1).casefold()
    if _PERSIAN_MT5_SOURCE_RE.search(text):
        return "mt5"
    return None
