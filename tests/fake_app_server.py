"""A fake `codex app-server` for offline tests.

Speaks the real protocol (newline-delimited JSON-RPC over stdio) so the real
openai-codex SDK and conduix/backend.py run unmodified against it:

    Backend(launch_args=(sys.executable, "tests/fake_app_server.py"))

Responses that carry complex protocol types (initialize, account/read,
model/list, thread/start) are replayed from fixtures recorded from a real
app-server (tests/fixtures/app_server_responses.json, email scrubbed).

Turns answer with a dump of the thread's model-visible history
("ctx=user:A;assistant:B;..."). Keywords in the user text script failures:

    TOOL   start a commandExecution item first (chat-only mode was bypassed)
    QUOTA  fail with codexErrorInfo usageLimitExceeded (error notification + failed turn)
    RETRY  send a will-retry error notification, then answer normally
    SLOW   stream deltas slowly until turn/interrupt arrives
    CRASH  exit the process mid-turn
    CALLTOOL  call the thread's first dynamic tool (`item/tool/call` request),
              then wait for turn/interrupt
    APPROVE   ask for a command approval (`item/commandExecution/requestApproval`)
    WEIRD     send a server request the client can't know
    SEARCH    run a webSearch item (only if the thread enabled web_search)
    TWICE     two model calls: two thread/tokenUsage/updated notifications

`fake/stats` returns what the server saw, for assertions. Environment:
FAKE_ACCOUNT=none makes account/read report no login.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "app_server_responses.json").read_text(encoding="utf-8"))

out_lock = threading.Lock()
threads: dict[str, dict] = {}
stats: dict = {
    "thread_start_params": [], "injected": [], "turn_start_params": [], "interrupted": [],
    "unsubscribed": [], "client_replies": [], "api_key_in_env": any(os.environ.get(k) for k in
                                              ("OPENAI_API_KEY", "CODEX_API_KEY")),
}
interrupts: set[str] = set()


def send(msg: dict) -> None:
    with out_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def notify(method: str, params: dict) -> None:
    send({"method": method, "params": params})


def now_ms() -> int:
    return int(time.time() * 1000)


def text_of(content: list[dict]) -> list[str]:
    out = []
    for c in content:
        if c.get("type") in ("text", "input_text", "output_text"):
            out.append(c["text"])
        elif c.get("type") in ("image", "input_image"):
            out.append(f"<image {len(c.get('url') or c.get('image_url') or '')}>")
    return out


def run_turn(thread_id: str, turn_id: str, input_items: list[dict]) -> None:
    th = threads[thread_id]
    base = {"threadId": thread_id, "turnId": turn_id}
    user_texts = text_of(input_items)
    joined = " ".join(user_texts)
    notify("turn/started", {"threadId": thread_id,
                            "turn": {"id": turn_id, "items": [], "status": "inProgress"}})
    user_item = {"type": "userMessage", "id": str(uuid.uuid4()),
                 "content": [{"type": "text", "text": t} for t in user_texts]}
    notify("item/started", {**base, "item": user_item, "startedAtMs": now_ms()})
    notify("item/completed", {**base, "item": user_item, "completedAtMs": now_ms()})
    th["history"] += [("user", t) for t in user_texts]

    def complete(status: str, error: dict | None = None) -> None:
        turn = {"id": turn_id, "items": [], "status": status}
        if error:
            turn["error"] = error
        notify("turn/completed", {"threadId": thread_id, "turn": turn})

    if "CRASH" in joined:
        os._exit(3)
    if "QUOTA" in joined:
        err = {"message": "You've hit your usage limit.", "codexErrorInfo": "usageLimitExceeded"}
        notify("error", {**base, "error": err, "willRetry": False})
        complete("failed", err)
        return
    if "CALLTOOL" in joined and th["tools"]:
        send({"id": f"srv-{turn_id}", "method": "item/tool/call", "params": {
            **base, "callId": "exec-1", "namespace": None, "tool": th["tools"][0]["name"],
            "arguments": {"city": "Kyiv"}}})
        for _ in range(500):
            if turn_id in interrupts:
                complete("interrupted")
                return
            time.sleep(0.01)
        complete("completed")
        return
    if "APPROVE" in joined:
        send({"id": "srv-approve", "method": "item/commandExecution/requestApproval",
              "params": {**base, "itemId": "c1", "command": "rm -rf /"}})
    if "WEIRD" in joined:
        send({"id": "srv-weird", "method": "item/somethingNew/request", "params": base})
    if "SEARCH" in joined and th["web_search"]:
        ws = {"type": "webSearch", "id": "ws_1", "query": "python",
              "action": {"type": "search", "query": "python"}}
        notify("item/started", {**base, "item": ws, "startedAtMs": now_ms()})
        done = {**ws, "action": {"type": "openPage", "url": "https://www.python.org/"}}
        notify("item/completed", {**base, "item": done, "completedAtMs": now_ms()})
    if "RETRY" in joined:
        notify("error", {**base, "error": {"message": "reconnecting..."}, "willRetry": True})
    if "TOOL" in joined:
        item = {"type": "commandExecution", "id": "call_1", "command": "dir", "commandActions": [],
                "cwd": th["cwd"], "status": "inProgress"}
        notify("item/started", {**base, "item": item, "startedAtMs": now_ms()})

    msg_id = f"msg_{uuid.uuid4().hex}"
    reply = "ctx=" + ";".join(f"{r}:{t}" for r, t in th["history"])
    item = {"type": "agentMessage", "id": msg_id, "text": "", "phase": "final_answer"}
    notify("item/started", {**base, "item": item, "startedAtMs": now_ms()})

    if "SLOW" in joined:
        for _ in range(400):
            if turn_id in interrupts:
                complete("interrupted")
                return
            notify("item/agentMessage/delta", {**base, "itemId": msg_id, "delta": "."})
            time.sleep(0.02)
        complete("completed")
        return

    half = len(reply) // 2
    for delta in (reply[:half], reply[half:]):
        notify("item/agentMessage/delta", {**base, "itemId": msg_id, "delta": delta})
    notify("item/completed", {**base, "item": {**item, "text": reply}, "completedAtMs": now_ms()})
    th["history"].append(("assistant", reply))
    for _ in range(2 if "TWICE" in joined else 1):
        last = {"inputTokens": 100, "cachedInputTokens": 80, "outputTokens": 5,
                "reasoningOutputTokens": 0, "totalTokens": 105, "cacheWriteInputTokens": 0}
        th["total"] = {k: th["total"].get(k, 0) + v for k, v in last.items()}
        notify("thread/tokenUsage/updated",
               {**base, "tokenUsage": {"last": last, "total": dict(th["total"])}})
    complete("completed")


def handle(method: str, params: dict) -> dict | None:
    if method == "initialize":
        return FIXTURES["initialize"]
    if method == "account/read":
        if os.environ.get("FAKE_ACCOUNT") == "none":
            return {"account": None, "requiresOpenaiAuth": True}
        return FIXTURES["account/read"]
    if method == "model/list":
        return FIXTURES["model/list"]
    if method == "account/rateLimits/read":
        return {"rateLimits": {"primary": {"usedPercent": 100, "windowDurationMins": 10080,
                                           "resetsAt": 1791131515}}}
    if method == "thread/start":
        stats["thread_start_params"].append(params)
        resp = json.loads(json.dumps(FIXTURES["thread/start"]))
        tid = str(uuid.uuid4())
        resp["thread"]["id"] = resp["thread"]["sessionId"] = tid
        threads[tid] = {"history": [], "cwd": params.get("cwd") or resp["cwd"],
                        "tools": params.get("dynamicTools") or [],
                        "web_search": (params.get("config") or {}).get("web_search") not in
                        (None, "disabled"), "total": {}}
        if params.get("developerInstructions"):
            threads[tid]["history"].append(("developer", params["developerInstructions"]))
        return resp
    if method == "thread/inject_items":
        stats["injected"].append(params["items"])
        for it in params["items"]:
            threads[params["threadId"]]["history"] += [
                (it["role"], t) for t in text_of(it["content"])]
        return {}
    if method == "thread/unsubscribe":
        stats["unsubscribed"].append(params["threadId"])
        return {"status": "unsubscribed"}
    if method == "turn/start":
        stats["turn_start_params"].append({k: v for k, v in params.items() if k != "input"})
        turn_id = str(uuid.uuid4())
        # Notifications must follow the response, as the real server's do.
        threading.Timer(0.01, run_turn, (params["threadId"], turn_id, params["input"])).start()
        return {"turn": {"id": turn_id, "items": [], "status": "inProgress"}}
    if method == "turn/interrupt":
        stats["interrupted"].append(params["turnId"])
        interrupts.add(params["turnId"])
        return {}
    if method == "fake/stats":
        return stats
    raise KeyError(method)


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if "id" not in msg:  # a notification from the client (e.g. `initialized`)
            continue
        if "method" not in msg:  # the client answering one of our requests
            stats["client_replies"].append(msg)
            continue
        try:
            send({"id": msg["id"], "result": handle(msg["method"], msg.get("params") or {})})
        except KeyError as e:
            send({"id": msg["id"], "error": {"code": -32601, "message": f"unknown method {e}"}})


if __name__ == "__main__":
    main()
