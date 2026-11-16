"""Voice spike: can the ChatGPT subscription produce speech through Codex's
experimental thread realtime API?

Steps: thread/realtime/listVoices, start a chat-only thread, then
thread/realtime/start with outputModality=audio over the websocket transport,
send one short phrase with thread/realtime/appendSpeech (falling back to
appendText if no audio arrives), and collect thread/realtime/outputAudio/delta
chunks into a WAV.

Run (spends a little plan quota; set PROBE_CODEX_BIN to try another Codex CLI):
    conda run -n conduix --no-capture-output python scripts/probe_realtime_voice.py [voice]

Writes the notification log to scratch/probe_voice_<timestamp>.jsonl and any
audio to scratch/probe_voice_<timestamp>_<phase>.wav.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_sdk import CHAT_ONLY_CONFIG, _dump  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scratch"

PHRASE = "Hello from Conduix. This is a short voice test."
TTS_PROMPT = (
    "You are a text-to-speech engine. When given text, read it aloud exactly as "
    "written, in a natural voice. Do not add, answer, or comment on anything."
)
QUIET_S = 6.0      # stop collecting after this long without an audio chunk
MAX_WAIT_S = 45.0  # hard cap per phase
LONG = 300


class AnyResult(BaseModel):
    model_config = ConfigDict(extra="allow")


def _shorten(obj: Any) -> Any:
    if isinstance(obj, str) and len(obj) > LONG:
        return f"<{len(obj)} chars> {obj[:40]}…"
    if isinstance(obj, list):
        return [_shorten(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _shorten(v) for k, v in obj.items()}
    return obj


def _transport() -> dict:
    """PROBE_TRANSPORT=webrtc sends the SDP offer in PROBE_SDP_FILE (or a stub)."""
    if os.environ.get("PROBE_TRANSPORT") != "webrtc":
        return {"type": "websocket"}
    f = os.environ.get("PROBE_SDP_FILE")
    stub = ["v=0", "o=- 0 0 IN IP4 127.0.0.1", "s=-", "t=0 0", "m=audio 9 UDP/TLS/RTP/SAVPF 111", "c=IN IP4 0.0.0.0", "a=rtpmap:111 opus/48000/2", "a=sendrecv"]
    sdp = Path(f).read_text() if f else "\r\n".join(stub) + "\r\n"
    return {"type": "webrtc", "sdp": sdp}


def _pump(sync: Any, out: queue.Queue, stop: threading.Event) -> None:
    """Move global (non-turn) notifications onto a queue. Runs in a thread."""
    while not stop.is_set():
        try:
            n = sync.next_notification()
        except BaseException as e:  # noqa: BLE001 - client closed
            out.put(e)
            return
        out.put(n)


async def main() -> int:
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):  # bill the ChatGPT plan, as the server does
        if os.environ.pop(key, None) is not None:
            print(f"== removed {key} from the environment")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"probe_voice_{stamp}.jsonl"
    workspace = Path(tempfile.mkdtemp(prefix="conduix_ws_"))
    voice = sys.argv[1] if len(sys.argv) > 1 else None

    # PROBE_CODEX_BIN picks a different Codex CLI than the SDK's bundled one.
    async with AsyncCodex(CodexConfig(codex_bin=os.environ.get("PROBE_CODEX_BIN"))) as codex:
        client = codex._client
        server = codex.metadata.serverInfo
        print("== codex:", server.version if server else None)
        acct = (await codex.account()).account
        info = acct.root if acct is not None else None
        print("== account:", getattr(info, "type", None), getattr(info, "plan_type", None))
        if getattr(info, "type", None) != "chatgpt":
            print("!! not a chatgpt account; refusing to run")
            return 1

        async def call(method: str, params: dict | None) -> dict | None:
            try:
                r = await client.request(method, params, response_model=AnyResult)
                d = r.model_dump(mode="json")
                print(f"== {method} ok: {json.dumps(_shorten(d))[:600]}")
                return d
            except Exception as e:  # noqa: BLE001 - the probe wants to see anything
                print(f"!! {method} failed: {type(e).__name__}: {e}")
                return None

        await call("thread/realtime/listVoices", {})

        thread = await codex.thread_start(
            approval_mode=ApprovalMode.deny_all,
            sandbox=Sandbox.read_only,
            cwd=str(workspace),
            base_instructions=TTS_PROMPT,
            ephemeral=True,
            config=dict(CHAT_ONLY_CONFIG),
        )
        print(f"== thread: {thread.id}")

        notes: queue.Queue = queue.Queue()
        stop = threading.Event()
        threading.Thread(target=_pump, args=(client._sync, notes, stop), daemon=True).start()

        log = log_path.open("w", encoding="utf-8")
        t0 = time.perf_counter()

        async def drain(phase: str, until_audio_quiet: bool, max_wait: float) -> dict:
            """Collect notifications; return audio chunks and counts for the phase."""
            counts: Counter[str] = Counter()
            pcm = bytearray()
            fmt: tuple[int, int] | None = None
            transcript: list[str] = []
            last_audio = None
            started = time.perf_counter()
            while True:
                now = time.perf_counter()
                if now - started > max_wait:
                    break
                if until_audio_quiet and last_audio is not None and now - last_audio > QUIET_S:
                    break
                try:
                    n = await asyncio.to_thread(notes.get, True, 0.5)
                except queue.Empty:
                    if not until_audio_quiet and now - started > max_wait:
                        break
                    continue
                if isinstance(n, BaseException):
                    print(f"!! notification stream ended: {n!r}")
                    break
                payload = _dump(n.payload)
                counts[n.method] += 1
                log.write(json.dumps({"phase": phase, "t": round(time.perf_counter() - t0, 2),
                                      "method": n.method, "payload": _shorten(payload)},
                                     default=str) + "\n")
                if n.method == "thread/realtime/outputAudio/delta":
                    a = payload.get("audio", {})
                    pcm += base64.b64decode(a.get("data", ""))
                    fmt = (a.get("sampleRate") or a.get("sample_rate"),
                           a.get("numChannels") or a.get("num_channels"))
                    last_audio = time.perf_counter()
                    if counts[n.method] <= 2:
                        print(f"  {time.perf_counter() - t0:5.1f}s audio chunk: "
                              f"{json.dumps(_shorten(a))[:200]}")
                    continue
                if "transcript" in n.method and isinstance(payload, dict):
                    if payload.get("delta"):
                        transcript.append(payload["delta"])
                print(f"  {time.perf_counter() - t0:5.1f}s {n.method}: "
                      f"{json.dumps(_shorten(payload))[:300]}")
                if n.method in ("thread/realtime/started",) and not until_audio_quiet:
                    break
                if n.method in ("thread/realtime/error", "thread/realtime/closed"):
                    break
            return {"counts": dict(counts), "pcm": bytes(pcm), "fmt": fmt,
                    "transcript": "".join(transcript)}

        def save(phase: str, res: dict) -> None:
            print(f"  -- [{phase}] counts={res['counts']}")
            if res["transcript"]:
                print(f"  -- [{phase}] transcript: {res['transcript']!r}")
            if not res["pcm"]:
                print(f"  -- [{phase}] no audio")
                return
            rate, ch = res["fmt"] or (24000, 1)
            out = OUT_DIR / f"probe_voice_{stamp}_{phase}.wav"
            with wave.open(str(out), "wb") as w:
                w.setnchannels(ch or 1)
                w.setsampwidth(2)  # assume PCM16 (the realtime API's default)
                w.setframerate(rate or 24000)
                w.writeframes(res["pcm"])
            secs = len(res["pcm"]) / 2 / (ch or 1) / (rate or 24000)
            print(f"  ** [{phase}] {len(res['pcm'])} bytes PCM, {rate} Hz x{ch} "
                  f"≈ {secs:.1f}s → {out}")

        try:
            params: dict[str, Any] = {
                "threadId": thread.id,
                "outputModality": "audio",
                "transport": _transport(),
                "prompt": TTS_PROMPT,
                "includeStartupContext": False,
            }
            if voice:
                params["voice"] = voice
            if await call("thread/realtime/start", params) is None:
                return 2
            res = await drain("start", until_audio_quiet=False, max_wait=20)
            print(f"  -- [start] counts={res['counts']}")

            got_audio = False
            if await call("thread/realtime/appendSpeech",
                          {"threadId": thread.id, "text": PHRASE}) is not None:
                res = await drain("speech", until_audio_quiet=True, max_wait=MAX_WAIT_S)
                save("speech", res)
                got_audio = bool(res["pcm"])

            if not got_audio:
                print("\n== no audio from appendSpeech; trying appendText")
                if await call("thread/realtime/appendText",
                              {"threadId": thread.id, "text": f"Read this aloud: {PHRASE}"}
                              ) is not None:
                    res = await drain("text", until_audio_quiet=True, max_wait=MAX_WAIT_S)
                    save("text", res)
        finally:
            await call("thread/realtime/stop", {"threadId": thread.id})
            res = await drain("stop", until_audio_quiet=False, max_wait=3)
            print(f"  -- [stop] counts={res['counts']}")
            stop.set()
            log.close()

    print(f"\n== full log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
