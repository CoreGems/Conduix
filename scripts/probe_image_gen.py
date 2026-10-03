"""Step 13 probe: turn Codex's built-in image generation back on for one
thread and dump every notification, so the `image_generation` tool is built
against real payloads rather than guesses.

Run (spends plan quota: one image per prompt):
    conda run -n conduix --no-capture-output python scripts/probe_image_gen.py [model]

Writes the notification log to scratch/probe_image_<timestamp>.jsonl (long
base64 strings shortened) and each generated image to scratch/.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from openai_codex import ApprovalMode, AsyncCodex, Sandbox
from openai_codex.generated.v2_all import ModelProviderCapabilitiesReadResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_sdk import CHAT_ONLY_CONFIG, _dump, _short  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scratch"

BASE_INSTRUCTIONS = (
    "You are a helpful assistant answering over a chat API. "
    "You have no shell and no filesystem. You can generate images."
)

TURNS = [
    ("image", "Generate an image: a small red circle centered on a plain white background."),
    ("followup", "In one sentence, what did the image you just made show?"),
]

LONG = 400  # strings longer than this are shortened in the log


def _shorten(obj):
    if isinstance(obj, str) and len(obj) > LONG:
        return f"<{len(obj)} chars> {obj[:60]}…"
    if isinstance(obj, list):
        return [_shorten(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _shorten(v) for k, v in obj.items()}
    return obj


async def main() -> int:
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):  # bill the ChatGPT plan, as the server does
        if os.environ.pop(key, None) is not None:
            print(f"== removed {key} from the environment")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"probe_image_{stamp}.jsonl"
    workspace = Path(tempfile.mkdtemp(prefix="conduix_ws_"))
    model = sys.argv[1] if len(sys.argv) > 1 else None

    with log_path.open("w", encoding="utf-8") as log:
        async with AsyncCodex() as codex:
            acct = (await codex.account()).account
            info = acct.root if acct is not None else None
            print("== account:", getattr(info, "type", None), getattr(info, "plan_type", None))

            try:
                caps = await codex._client.request(
                    "modelProvider/capabilities/read", {},
                    response_model=ModelProviderCapabilitiesReadResponse)
                print("== capabilities:", _dump(caps))
                log.write(json.dumps({"kind": "capabilities", "data": _dump(caps)}) + "\n")
            except Exception as e:  # noqa: BLE001 - probe wants to see anything
                print(f"!! modelProvider/capabilities/read failed: {type(e).__name__}: {e}")

            thread = await codex.thread_start(
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.read_only,
                cwd=str(workspace),
                base_instructions=BASE_INSTRUCTIONS,
                ephemeral=True,
                model=model,
                config={**CHAT_ONLY_CONFIG, "features.image_generation": True},
            )
            print(f"== thread: {thread.id}  workspace={workspace}")

            images = 0
            for label, prompt in TURNS:
                print(f"\n=== turn [{label}] {prompt!r}")
                counts: Counter[str] = Counter()
                t0 = time.perf_counter()
                handle = await thread.turn(prompt, effort="low", summary="auto")
                async for n in handle.stream():
                    payload = _dump(n.payload)
                    counts[n.method] += 1
                    item = payload.get("item") if isinstance(payload, dict) else None
                    if (isinstance(item, dict) and item.get("type") == "imageGeneration"
                            and n.method == "item/completed" and item.get("result")):
                        try:
                            data = base64.b64decode(item["result"], validate=True)
                            images += 1
                            out = OUT_DIR / f"probe_image_{stamp}_{images}.bin"
                            out.write_bytes(data)
                            print(f"  ** image: {len(data)} bytes, magic={data[:12]!r} → {out}")
                        except Exception as e:  # noqa: BLE001
                            print(f"  ** result is not base64: {type(e).__name__}: {e}")
                        saved = Path(item.get("saved_path") or "")
                        if saved.is_file():  # Codex's own copy, under ~/.codex/generated_images
                            print(f"  ** Codex saved it to {saved}; removing that copy")
                            saved.unlink()
                            if not any(saved.parent.iterdir()):
                                saved.parent.rmdir()
                    log.write(json.dumps({"turn": label, "t": round(time.perf_counter() - t0, 2),
                                          "method": n.method,
                                          "payload_type": type(n.payload).__name__,
                                          "payload": _shorten(payload)}, default=str) + "\n")
                    if "delta" in n.method.lower() and counts[n.method] > 2:
                        continue
                    print(f"  {time.perf_counter() - t0:5.1f}s {n.method}: "
                          f"{_short(_shorten(payload), 300)}")
                print(f"  -- {time.perf_counter() - t0:.1f}s; counts={dict(counts)}")

    leftovers = [str(p) for p in workspace.rglob("*")]
    print(f"\n== workspace untouched: {not leftovers} {leftovers or ''}")
    print(f"== full log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
