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
    IMAGE     run an imageGeneration item (only if the thread enabled
              features.image_generation); the image is also written to disk,
              as Codex does. IMAGELIMIT fails it on a usage limit instead
    TWICE     two model calls: two thread/tokenUsage/updated notifications

Thread realtime (speech, step 14): thread/realtime/start answers a webrtc
offer with protocol v3 (anything else fails like the real server: websocket
needs an API key, v1/v2 are refused, unknown voices are listed). Each
appendText is "spoken" back as a transcript of the text after its "...: "
prefix. Keywords in that text:

    MUMBLE   the voice says "Mhm." instead, every time
    ONCE     says "Mhm." the first time, then the text
    RTERROR  a thread/realtime/error instead of a transcript

`fake/stats` returns what the server saw, for assertions. Environment:
FAKE_ACCOUNT=none makes account/read report no login.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "app_server_responses.json").read_text(encoding="utf-8"))

# A 1x1 PNG, and where this fake "Codex home" keeps generated images.
FAKE_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
GENERATED_IMAGES = Path(tempfile.gettempdir()) / "conduix_fake_codex" / "generated_images"

out_lock = threading.Lock()
threads: dict[str, dict] = {}
stats: dict = {
    "thread_start_params": [], "injected": [], "turn_start_params": [], "interrupted": [],
    "unsubscribed": [], "client_replies": [],
    "realtime_start": [], "realtime_text": [], "realtime_stopped": [], "api_key_in_env": any(os.environ.get(k) for k in
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
    if "IMAGE" in joined and th["image_generation"]:
        ig = {"type": "imageGeneration", "id": f"exec-{turn_id}", "status": "in_progress",
              "result": ""}
        notify("item/started", {**base, "item": ig, "startedAtMs": now_ms()})
        if "IMAGELIMIT" in joined:
            done = {**ig, "status": "failed", "failure": {
                "type": "usageLimitExceeded", "limitId": "image_generation",
                "resetsAt": 1791131515}}
        else:
            saved = GENERATED_IMAGES / thread_id / f"{ig['id']}.png"
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(base64.b64decode(FAKE_IMAGE))
            done = {**ig, "status": "completed", "result": FAKE_IMAGE, "revisedPrompt": "A dot.",
                    "savedPath": str(saved), "transparentBackground": False}
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


V3_VOICES = ["juniper", "maple", "spruce", "ember", "vale", "breeze", "arbor", "sol", "cove",
             "aube", "bubbie", "haetsal", "hanul", "himari", "leher", "miko", "neer", "porto",
             "rindo", "rio", "rivage", "selva", "sonna", "tinta", "viola"]
realtime_seen: dict[str, int] = {}  # thread id → appendText calls


def realtime_start(params: dict) -> None:
    tid = params["threadId"]
    transport = params.get("transport") or {"type": "websocket"}
    voice = params.get("voice")
    if transport.get("type") != "webrtc":
        error = "realtime conversation requires API key auth"
    elif params.get("version") != "v3":
        error = "AVAS realtime calls require realtime v1 or v3"
    elif voice is not None and voice not in V3_VOICES:
        error = f"realtime voice `{voice}` is not supported for v3; supported voices: " +             ", ".join(V3_VOICES)
    else:
        error = None
    if error:
        notify("thread/realtime/error", {"threadId": tid, "message": error})
        return
    notify("thread/realtime/started", {"threadId": tid, "realtimeSessionId": tid,
                                       "version": "v3"})
    notify("thread/realtime/sdp", {"threadId": tid, "sdp": "fake-answer:" + transport["sdp"]})


def realtime_speak(tid: str, text: str) -> None:
    n = realtime_seen[tid] = realtime_seen.get(tid, 0) + 1
    if "RTERROR" in text:
        notify("thread/realtime/error", {"threadId": tid, "message": "voice backend failed"})
        return
    said = text.split(": ", 1)[-1]
    if "MUMBLE" in text or ("ONCE" in text and n == 1):
        said = "Mhm."
    notify("thread/realtime/transcript/done", {"threadId": tid, "role": "assistant",
                                               "text": " " + said})


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
                        (None, "disabled"),
                        "image_generation": (params.get("config") or {}).get(
                            "features.image_generation") is True,
                        "total": {}}
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
    if method == "thread/realtime/listVoices":
        return {"voices": {"v1": V3_VOICES[:9], "v2": ["alloy", "marin"], "v3": V3_VOICES,
                           "defaultV1": "cove", "defaultV2": "marin"}}
    if method == "thread/realtime/start":
        stats["realtime_start"].append(params)
        threading.Timer(0.01, realtime_start, (params,)).start()
        return {}
    if method == "thread/realtime/appendText":
        stats["realtime_text"].append(params["text"])
        threading.Timer(0.01, realtime_speak, (params["threadId"], params["text"])).start()
        return {}
    if method == "thread/realtime/stop":
        stats["realtime_stopped"].append(params["threadId"])
        notify("thread/realtime/closed", {"threadId": params["threadId"], "reason": "requested"})
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
