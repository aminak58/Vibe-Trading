"""talk_agent — converse with a Vibe-Trading research agent via its local HTTP API.

Location: agent/scripts/talk_agent.py  (Vibe-Trading project)
Run without a console window on Windows:  pythonw talk_agent.py ...
Interactive console:                       python  talk_agent.py ...

Usage:
  talk_agent.py "MESSAGE" [--session ID] [--title T] [--timeout SEC]

Examples:
  # create a new session and ask the agent to run a swarm research on mt5 gold/BTC/Dow
  pythonw talk_agent.py \
    "Investigate WaveTrend 3D MTF scalping on XAUUSD_o/BTCUSD/YM using backtest source=mt5, position_adjustment=hold, then run swarm preset quant_scalp_desk" \
    --title wt3d-mtf

  # reply to an existing session (continue previous conversation)
  pythonw talk_agent.py "continue: now run the swarm report" --session <session_id>

  # long timeout (agent reasoning may take minutes)
  pythonw talk_agent.py "..." --timeout 600

API used (local, read-only research):
  POST /sessions                     -> create session        (returns session_id)
  POST /sessions/<id>/messages       -> send user message     (409 = agent busy, we wait)
  GET  /sessions/<id>/messages?limit -> read messages         (list of MessageResponse)
  POST /sessions/<id>/cancel         -> cancel in-flight loop

Every call is logged to ~/.vibe-trading/talk_agent.jsonl
"""
import argparse, json, os, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8899"
LOG = os.path.join(os.path.expanduser("~"), ".vibe-trading", "talk_agent.jsonl")


def _req(method, path, body=None, timeout=30):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def create_session(title):
    st, j = _req("POST", "/sessions", {"title": title[:50]})
    if st == 200 and isinstance(j, dict):
        sid = j.get("session_id") or j.get("id")
        if sid:
            return sid, j
    return None, j


def send_message(sid, text):
    st, j = _req("POST", f"/sessions/{sid}/messages", {"content": text})
    if st == 200:
        return True, j
    if st == 409:
        return False, {"busy": True}
    return False, j


def get_messages(sid, limit=200):
    st, j = _req("GET", f"/sessions/{sid}/messages?limit={limit}")
    return j if (st == 200 and isinstance(j, list)) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--title", default="talk-agent")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("message", nargs="*")
    a = ap.parse_args()
    text = " ".join(a.message).strip()
    if not text:
        print("usage: talk_agent.py MESSAGE [--session ID] [--title T] [--timeout SEC]")
        return

    sid = a.session
    if not sid:
        sid, j = create_session(a.title)
        if not sid:
            print("create_session failed:", j)
            return
        print("[session] created:", sid)
    else:
        print("[session] reuse:", sid)

    before = get_messages(sid, 200)
    before_asst = sum(1 for m in before if isinstance(m, dict) and m.get("role") == "assistant")

    sent = False
    for _ in range(24):  # up to ~2 min wait if agent busy
        ok, resp = send_message(sid, text)
        if ok:
            sent = True
            break
        if isinstance(resp, dict) and resp.get("busy"):
            print("  (agent busy, waiting 5s...)", flush=True)
            time.sleep(5)
        else:
            print("send failed:", resp)
            return
    if not sent:
        print("could not send (agent stayed busy)")
        return
    print("[sent] ok. polling for reply...", flush=True)

    deadline = time.time() + a.timeout
    reply = None
    while time.time() < deadline:
        msgs = get_messages(sid, 200)
        assistants = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        if len(assistants) > before_asst:
            reply = assistants[-1].get("content", "")
            break
        time.sleep(4)

    print("\n=== agent reply ===")
    print(reply if reply else "(no reply within timeout)")
    print("\n(session_id: %s)" % sid)

    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "session": sid, "sent": text,
                                "reply": reply or None}, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()