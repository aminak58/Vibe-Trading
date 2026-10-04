---
name: external-orchestration
description: Tells the agent that an external orchestrator (MetaTrader Assistant / another agent / CLI) may converse with it via the local HTTP API — keep context in the same session, expect follow-up messages, and use vibe-ops-workarounds for known infra bugs.
category: ops
---

# External Orchestration

An **external orchestrator** (e.g. MetaTrader Assistant, Codex, a CLI tool, or
another agent harness) can send you messages and drive your research through
the local Vibe-Trading HTTP API:

- `POST /sessions` — create a session
- `POST /sessions/{id}/messages` — send you a user message (HTTP 409 = you are busy; the caller waits and retries)
- `GET /sessions/{id}/messages` — read the conversation
- `POST /sessions/{id}/cancel` — cancel your in-flight loop

## What this means for you

1. **Stay in the same session.** Messages to the same session keep history, so
   the orchestrator can correct you mid-run, ask follow-ups, or add steps —
   treat each incoming message as a natural continuation, not a fresh task.
2. **Expect short, directive messages.** The orchestrator may say "continue",
   "run the swarm now", or point you to a workaround. Answer the instruction,
   then continue the current task (data gathering → backtest → swarm → report).
3. **If a message is a correction** (e.g. "don't use get_market_data"), apply it
   immediately and note it in your output; do not restart from scratch.
4. **Known infra bugs and safe paths** are documented in the skill
   `vibe-ops-workarounds`. Load it if the orchestrator references it, or if you
   hit `identity_required` / `identity_mismatch` / `insufficient capital` /
   `read_url 401`.
5. **If the orchestrator asks for a swarm report**, call
   `run_swarm(preset_name=...)` explicitly (e.g. `quant_scalp_desk`) once your
   backtest data is real — do not claim a swarm ran when it did not.

## Bridge tool

`agent/scripts/talk_agent.py` (see `agent/scripts/TALK_AGENT.md`) is the
reference client the orchestrator uses. You do not need to call it yourself;
you just answer messages normally.
