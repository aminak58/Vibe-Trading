# Talk to a Vibe-Trading Research Agent from Outside

Any external agent (e.g. MetaTrader Assistant, Codex, a CLI tool, another agent
harness) can hold a **conversation with the Vibe-Trading research agent** over
its local HTTP API. This is how a second brain drives the research harness
without being inside it.

## What it is

| Endpoint | Purpose |
|---|---|
| `POST /sessions` | create a session → returns `session_id` |
| `POST /sessions/<id>/messages` | send a user message → starts the agent loop (HTTP `409` = agent busy) |
| `GET /sessions/<id>/messages?limit=N` | read messages (list; newest last) |
| `POST /sessions/<id>/cancel` | cancel the in-flight loop |

Messages sent to the **same session** keep conversation history → the agent
remembers previous turns (multi-step dialogue, corrections, follow-ups).

## The bridge script

`agent/scripts/talk_agent.py` wraps the API into one command.

```bash
# create a new session + ask for a swarm research on mt5 broker data
pythonw agent/scripts/talk_agent.py \
  "Investigate WaveTrend 3D MTF scalping on XAUUSD_o/BTCUSD/YM: run backtest source=mt5 position_adjustment=hold per symbol, then run swarm preset quant_scalp_desk" \
  --title wt3d-mtf --timeout 600

# continue an existing conversation (keeps context)
pythonw agent/scripts/talk_agent.py "continue: now run the swarm report" --session <SESSION_ID>

# no-window launcher (Windows): talk_agent_w.cmd
agent/scripts/talk_agent_w.cmd "message" --session <SESSION_ID>
```

Every call is logged to `~/.vibe-trading/talk_agent.jsonl`.

## How other agents discover this capability

Any Vibe-Trading agent that loads the skill **`external-orchestration`** learns
that an external orchestrator may send it messages/sessions at any time, that it
should keep working in the same session, and that workarounds for known
infra bugs live in **`vibe-ops-workarounds`**.

## Notes for the external caller

- The agent has its own tool budget (default 50 ReAct iterations per message).
  For long research, split the work into steps and send follow-up messages.
- If you want a **swarm** report, say so explicitly:
  `... then run_swarm(preset_name="quant_scalp_desk") ...`.
- If the API returns 409, the agent is busy — the bridge waits and retries.
