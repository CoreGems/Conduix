"""Voice spike, part 2: thread realtime over the WebRTC transport.

The websocket transport is refused on a ChatGPT login ("realtime conversation
requires API key auth"), but a webrtc offer gets past that check. This probe
acts as the WebRTC peer itself (aiortc): it sends a real SDP offer with a
silent microphone track, applies the answer from thread/realtime/sdp, sends one
phrase with thread/realtime/appendSpeech, and records the incoming audio track
to a WAV.

Needs aiortc, which is not a Conduix dependency. Install it somewhere apart and
put that folder on PYTHONPATH:
    python -m pip install --target <dir> aiortc
    PYTHONPATH=<dir> PROBE_CODEX_BIN=<codex.exe> \
        conda run -n conduix --no-capture-output python scripts/probe_realtime_webrtc.py [voice]
"""
from __future__ import annotations

import array
import asyncio
import json
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
from aiortc.mediastreams import AudioStreamTrack

from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_realtime_voice import PHRASE, TTS_PROMPT, AnyResult, _pump, _shorten  # noqa: E402
from probe_sdk import CHAT_ONLY_CONFIG, _dump  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scratch"
RECORD_CAP_S = 25.0


def wav_stats(path: Path) -> str:
    with wave.open(str(path), "rb") as w:
        rate, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(frames)
    if width != 2 or not raw:
        return f"{rate} Hz x{ch}, width {width}, {frames} frames"
    s = array.array("h", raw)
    # Loudness per 100 ms window: how much of the file is more than near silence.
    win = max(1, rate * ch // 10)
    loud = sum(1 for i in range(0, len(s), win)
               if max((abs(x) for x in s[i:i + win]), default=0) > 1000)
    total = (len(s) + win - 1) // win
    return (f"{rate} Hz x{ch}, {frames / rate:.1f}s, peak {max(map(abs, s))}, "
            f"{loud}/{total} windows of 100 ms above near silence")


async def main() -> int:
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        if os.environ.pop(key, None) is not None:
            print(f"== removed {key} from the environment")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"probe_webrtc_{stamp}.jsonl"
    wav_path = OUT_DIR / f"probe_webrtc_{stamp}.wav"
    workspace = Path(tempfile.mkdtemp(prefix="conduix_ws_"))
    voice = sys.argv[1] if len(sys.argv) > 1 else None
    log = log_path.open("w", encoding="utf-8")
    t0 = time.perf_counter()

    def note(kind: str, data: Any) -> None:
        log.write(json.dumps({"t": round(time.perf_counter() - t0, 2), "kind": kind,
                              "data": _shorten(data)}, default=str) + "\n")
        log.flush()

    # --- the WebRTC peer -------------------------------------------------
    pc = RTCPeerConnection()
    pc.addTrack(AudioStreamTrack())  # silent microphone
    recorder: MediaRecorder | None = None
    got_track = asyncio.Event()
    dc_events: list[dict] = []
    response_done = asyncio.Event()  # a voice turn ended (v3: turn.done)
    session_started = asyncio.Event()  # data channel: session.started

    @pc.on("track")
    async def on_track(track):  # noqa: ANN001
        nonlocal recorder
        print(f"  {time.perf_counter() - t0:5.1f}s remote track: {track.kind}")
        if track.kind == "audio" and recorder is None:
            recorder = MediaRecorder(str(wav_path))
            recorder.addTrack(track)
            await recorder.start()
            got_track.set()

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"  {time.perf_counter() - t0:5.1f}s peer connection: {pc.connectionState}")

    dc = pc.createDataChannel("oai-events")

    @dc.on("message")
    def on_dc(msg):  # noqa: ANN001
        try:
            ev = json.loads(msg)
        except Exception:  # noqa: BLE001
            ev = {"raw": str(msg)[:200]}
        dc_events.append(ev)
        note("datachannel", ev)
        t = ev.get("type", "?")
        if not t.endswith(".delta"):
            print(f"  {time.perf_counter() - t0:5.1f}s dc: {t}")
        if t in ("response.done", "turn.done"):
            response_done.set()
        if t == "session.started":
            session_started.set()

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)  # aiortc gathers ICE candidates here
    sdp = pc.localDescription.sdp
    print(f"== local offer: {len(sdp)} chars")

    try:
        async with AsyncCodex(CodexConfig(codex_bin=os.environ.get("PROBE_CODEX_BIN"))) as codex:
            client = codex._client
            server = codex.metadata.serverInfo
            print("== codex:", server.version if server else None)
            acct = (await codex.account()).account
            info = acct.root if acct is not None else None
            if getattr(info, "type", None) != "chatgpt":
                print("!! not a chatgpt account; refusing to run")
                return 1
            print("== account:", info.type, info.plan_type)

            async def call(method: str, params: dict) -> dict | None:
                try:
                    r = await client.request(method, params, response_model=AnyResult)
                    print(f"== {method} ok")
                    return r.model_dump(mode="json")
                except Exception as e:  # noqa: BLE001
                    print(f"!! {method} failed: {type(e).__name__}: {e}")
                    return None

            thread = await codex.thread_start(
                approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only,
                cwd=str(workspace), base_instructions=TTS_PROMPT, ephemeral=True,
                config=dict(CHAT_ONLY_CONFIG),
            )
            print(f"== thread: {thread.id}")

            notes: queue.Queue = queue.Queue()
            stop = threading.Event()
            threading.Thread(target=_pump, args=(client._sync, notes, stop), daemon=True).start()

            started = asyncio.Event()
            failed: list[str] = []
            transcript: list[str] = []

            async def watch() -> None:
                while not stop.is_set():
                    try:
                        n = await asyncio.to_thread(notes.get, True, 0.5)
                    except queue.Empty:
                        continue
                    if isinstance(n, BaseException):
                        return
                    payload = _dump(n.payload)
                    note(n.method, payload)
                    m = n.method
                    if m == "thread/realtime/sdp":
                        print(f"  {time.perf_counter() - t0:5.1f}s answer SDP received")
                        await pc.setRemoteDescription(
                            RTCSessionDescription(sdp=payload["sdp"], type="answer"))
                    elif m == "thread/realtime/started":
                        print(f"  {time.perf_counter() - t0:5.1f}s {m}: {payload}")
                        started.set()
                    elif m in ("thread/realtime/error", "thread/realtime/closed"):
                        print(f"  {time.perf_counter() - t0:5.1f}s {m}: {payload}")
                        failed.append(m)
                        started.set()
                    elif "transcript" in m:
                        if payload.get("delta"):
                            transcript.append(payload["delta"])
                        if not m.endswith("delta"):
                            print(f"  {time.perf_counter() - t0:5.1f}s {m}: "
                                  f"{json.dumps(_shorten(payload))[:200]}")
                    elif m.startswith("thread/realtime/"):
                        print(f"  {time.perf_counter() - t0:5.1f}s {m}: "
                              f"{json.dumps(_shorten(payload))[:200]}")

            watcher = asyncio.create_task(watch())
            params: dict[str, Any] = {
                "threadId": thread.id, "outputModality": "audio",
                "transport": {"type": "webrtc", "sdp": sdp},
                "prompt": TTS_PROMPT, "includeStartupContext": False,
            }
            if voice:
                params["voice"] = voice
            if os.environ.get("PROBE_VERSION"):  # realtime protocol: v1, v2 or v3
                params["version"] = os.environ["PROBE_VERSION"]
            try:
                if await call("thread/realtime/start", params) is None:
                    return 2
                try:
                    await asyncio.wait_for(started.wait(), 20)
                except TimeoutError:
                    print("!! no thread/realtime/started within 20s; continuing anyway")
                if failed:
                    return 3
                try:
                    await asyncio.wait_for(got_track.wait(), 10)
                except TimeoutError:
                    print("!! no remote audio track within 10s")

                try:
                    await asyncio.wait_for(session_started.wait(), 10)
                except TimeoutError:
                    print("!! no session.started on the data channel within 10s")
                # Let any unprompted greeting finish before sending the phrase.
                try:
                    await asyncio.wait_for(response_done.wait(), 6)
                    print("  (an unprompted voice turn finished first)")
                except TimeoutError:
                    pass

                attempts = [
                    ("appendSpeech", {"threadId": thread.id, "text": PHRASE}),
                    ("appendText", {"threadId": thread.id,
                                    "text": f"Say exactly this, and nothing else: {PHRASE}"}),
                ]
                for method, args in attempts:
                    response_done.clear()
                    transcript.clear()
                    t_speech = time.perf_counter()
                    print(f"\n== {method} at {t_speech - t0:.1f}s")
                    await call(f"thread/realtime/{method}", args)
                    try:
                        await asyncio.wait_for(response_done.wait(), 15)
                        await asyncio.sleep(1.5)  # let the tail of the audio arrive
                    except TimeoutError:
                        print("  (no voice turn within 15s)")
                    said = "".join(transcript)
                    print(f"== [{method}] transcript: {said!r}")
                    if "Conduix" in said or "voice test" in said:
                        break
            finally:
                await call("thread/realtime/stop", {"threadId": thread.id})
                await asyncio.sleep(1.0)
                stop.set()
                watcher.cancel()
    finally:
        if recorder is not None:
            await recorder.stop()
        await pc.close()
        log.close()

    types: dict[str, int] = {}
    for ev in dc_events:
        types[ev.get("type", "?")] = types.get(ev.get("type", "?"), 0) + 1
    print(f"== data channel events: {types}")
    if wav_path.exists():
        print(f"== audio: {wav_path}\n   {wav_stats(wav_path)}")
    else:
        print("== no audio recorded")
    print(f"== full log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
