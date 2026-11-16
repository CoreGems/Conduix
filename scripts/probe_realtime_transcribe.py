"""Transcription spike: does a Codex realtime voice session (WebRTC, v3)
transcribe audio we send it?

Plays an audio file into the WebRTC microphone track in real time, then a few
seconds of silence, and collects what the session reports it heard
(thread/realtime/transcript/* with role "user", and the data channel's events).

Run (spends a little plan quota; PROBE_INTERRUPT=1 interrupts a delegated Codex turn;
PROBE_LINGER=<s> keeps the session open after the
transcript, to see what the voice and Codex do next):
    conda run -n conduix --no-capture-output python scripts/probe_realtime_transcribe.py <audio file> [text|audio]

The second argument is the session's outputModality (default text, so the
voice can't talk back). Log: scratch/probe_transcribe_<timestamp>.jsonl.
"""
from __future__ import annotations

import asyncio
import fractions
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

from openai_codex import ApprovalMode, AsyncCodex, Sandbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_realtime_voice import AnyResult, _shorten  # noqa: E402
from probe_sdk import CHAT_ONLY_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scratch"
RATE, FRAME = 48_000, 960  # 20 ms frames, as WebRTC sends Opus
TRAILING_SILENCE_S = 6.0
PROMPT = ("You are a transcription service. Listen to the user's audio. Do not reply, answer, "
          "greet or comment. Stay silent.")


class FileTrack(MediaStreamTrack):
    """Silence until `playing` is set, then the file's audio paced in real
    time, then silence forever. (Audio sent before the data channel's
    session.started is lost: found in this spike.)"""
    kind = "audio"

    def __init__(self, path: str) -> None:
        super().__init__()
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        pcm = bytearray()
        with av.open(path) as c:
            for frame in c.decode(audio=0):
                for f in resampler.resample(frame):
                    pcm += bytes(f.planes[0])[: f.samples * 2]
        self.pcm = bytes(pcm)
        self.duration = len(self.pcm) / 2 / RATE
        self._pos = 0
        self._start: float | None = None
        self._ts = 0
        self.finished_at: float | None = None
        self.playing = False

    async def recv(self) -> av.AudioFrame:
        if self._start is None:
            self._start = time.monotonic()
        else:
            self._ts += FRAME
            wait = self._start + self._ts / RATE - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        if self.playing:
            chunk = self.pcm[self._pos: self._pos + FRAME * 2]
            self._pos += FRAME * 2
        else:
            chunk = bytes(FRAME * 2)
        if self.playing and len(chunk) < FRAME * 2:
            if chunk == b"" and self.finished_at is None:
                self.finished_at = time.monotonic()
            chunk = chunk + bytes(FRAME * 2 - len(chunk))
        frame = av.AudioFrame(format="s16", layout="mono", samples=FRAME)
        frame.planes[0].update(chunk)
        frame.sample_rate = RATE
        frame.pts = self._ts
        frame.time_base = fractions.Fraction(1, RATE)
        return frame


async def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        os.environ.pop(key, None)
    src = sys.argv[1]
    modality = sys.argv[2] if len(sys.argv) > 2 else "text"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = (OUT_DIR / f"probe_transcribe_{stamp}.jsonl").open("w", encoding="utf-8")
    t0 = time.perf_counter()

    def note(kind: str, data: Any) -> None:
        log.write(json.dumps({"t": round(time.perf_counter() - t0, 2), "kind": kind,
                              "data": _shorten(data)}, default=str) + "\n")
        log.flush()

    def say(msg: str) -> None:
        print(f"  {time.perf_counter() - t0:5.1f}s {msg}")

    track = FileTrack(src)
    print(f"== source: {src} ({track.duration:.1f}s), outputModality={modality}")
    pc = RTCPeerConnection()
    pc.addTrack(track)
    ready = asyncio.Event()
    heard: list[str] = []
    said: list[str] = []
    channel = pc.createDataChannel("oai-events")

    @channel.on("message")
    def on_dc(msg: Any) -> None:
        try:
            ev = json.loads(msg)
        except (TypeError, ValueError):
            return
        note("dc", ev)
        t = ev.get("type", "?")
        if t == "session.started":
            ready.set()
        if not t.endswith(".delta") and t != "output_transcript.added":
            say(f"dc {t}: {json.dumps(_shorten(ev))[:220]}")

    @pc.on("track")
    def on_track(remote: Any) -> None:
        say(f"remote track: {remote.kind}")

    await pc.setLocalDescription(await pc.createOffer())

    async with AsyncCodex() as codex:
        client, sync = codex._client, codex._client._sync
        acct = (await codex.account()).account
        if getattr(acct.root if acct else None, "type", None) != "chatgpt":
            print("!! not a chatgpt account")
            return 1
        thread = await codex.thread_start(
            approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only,
            cwd=tempfile.mkdtemp(prefix="conduix_ws_"), base_instructions=PROMPT, ephemeral=True,
            config=dict(CHAT_ONLY_CONFIG))

        loop = asyncio.get_running_loop()
        failed = asyncio.Event()

        # Route this thread's realtime notifications here (as Conduix's backend does).
        orig = sync._router.route_notification

        def route(n: Any) -> None:
            method = getattr(n, "method", "")
            pd = n.payload.model_dump(mode="json") if hasattr(n.payload, "model_dump") else {}
            if not method.startswith("thread/realtime/") and (
                    pd.get("thread_id") == thread.id or
                    (pd.get("thread") or {}).get("id") == thread.id):
                loop.call_soon_threadsafe(on_thread, method, pd)
            if method.startswith("thread/realtime/"):
                payload = n.payload.model_dump(mode="json") if hasattr(n.payload, "model_dump") \
                    else getattr(n.payload, "params", {})
                loop.call_soon_threadsafe(on_codex, method, payload)
                return
            orig(n)

        thread_events: list[str] = []

        def on_thread(method: str, p: dict) -> None:
            """Anything else on this thread: a Codex turn would show up here."""
            note(method, p)
            thread_events.append(method)
            if method == "turn/started" and os.environ.get("PROBE_INTERRUPT"):
                # A delegated Codex turn: stop it before the model call.
                turn_id = (p.get("turn") or {}).get("id")
                say(f"interrupting delegated turn {turn_id}")
                loop.create_task(call("turn/interrupt", {"threadId": thread.id, "turnId": turn_id}))
            if "delta" not in method:
                say(f"THREAD {method}: {json.dumps(_shorten(p))[:260]}")

        def on_codex(method: str, p: dict) -> None:
            note(method, p)
            if method == "thread/realtime/sdp":
                loop.create_task(pc.setRemoteDescription(
                    RTCSessionDescription(sdp=p["sdp"], type="answer")))
                say("answer SDP applied")
                return
            if "delta" in method:
                return
            say(f"{method}: {json.dumps(_shorten(p))[:300]}")
            if method == "thread/realtime/transcript/done":
                (heard if p.get("role") == "user" else said).append(p.get("text", ""))
            if method in ("thread/realtime/error", "thread/realtime/closed"):
                failed.set()

        sync._router.route_notification = route

        async def call(method: str, params: dict) -> bool:
            try:
                await client.request(method, params, response_model=AnyResult)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"!! {method}: {type(e).__name__}: {e}")
                return False

        try:
            ok = await call("thread/realtime/start", {
                "threadId": thread.id, "outputModality": modality, "version": "v3",
                "transport": {"type": "webrtc", "sdp": pc.localDescription.sdp},
                "prompt": os.environ.get("PROBE_PROMPT", PROMPT), "includeStartupContext": False,
                # PROBE_START_EXTRA: JSON merged into the start params, to try options.
                **json.loads(os.environ.get("PROBE_START_EXTRA", "{}"))})
            if not ok:
                return 3
            waiter = asyncio.create_task(ready.wait())
            stopper = asyncio.create_task(failed.wait())
            await asyncio.wait({waiter, stopper}, timeout=30,
                               return_when=asyncio.FIRST_COMPLETED)
            if failed.is_set() or not ready.is_set():
                print("!! session did not start")
                return 4
            track.playing = True
            say("session started; the file is playing into the mic")
            # The track started pacing when the connection came up; wait for the
            # file to end, then trailing silence.
            deadline = time.monotonic() + track.duration + 15
            while track.finished_at is None and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            say("file finished; waiting for the transcript")
            end = time.monotonic() + TRAILING_SILENCE_S + 10
            while not heard and not failed.is_set() and time.monotonic() < end:
                await asyncio.sleep(0.1)
            linger = float(os.environ.get("PROBE_LINGER", "0"))
            say(f"transcript in; lingering {linger:.0f}s before stopping")
            await asyncio.sleep(linger)
        finally:
            await call("thread/realtime/stop", {"threadId": thread.id})
            await asyncio.sleep(1)
            sync._router.route_notification = orig
            await pc.close()
            log.close()

    print(f"\n== heard (user transcript): {heard!r}")
    print(f"== said (assistant transcript): {said!r}")
    print(f"== other notifications on the thread: {sorted(set(thread_events)) or 'none'}")
    print(f"== log: {OUT_DIR / f'probe_transcribe_{stamp}.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
