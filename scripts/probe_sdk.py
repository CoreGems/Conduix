"""Step 2 probe: drive the official Codex Python SDK as a chat-only backend and
dump every notification, so Conduix's streaming layer is designed against real
event names rather than guesses.

Run:
    conda run -n conduix --no-capture-output python scripts/probe_sdk.py

Writes the full notification log to scratch/probe_<timestamp>.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from openai_codex import ApprovalMode, AsyncCodex, Sandbox

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scratch"

BASE_INSTRUCTIONS = (
    "You are a helpful assistant answering over a plain chat API. "
    "You have no tools, no shell and no filesystem. Answer directly."
)

# Candidate "chat-only" overrides: every agentic feature Codex 0.157 lists as on.
CHAT_ONLY_CONFIG = {
    "web_search": "disabled",
    "features.shell_tool": False,
    "features.unified_exec": False,
    "features.view_image": False,
    "features.apps": False,
    "features.plugins": False,
    "features.multi_agent": False,
    "features.browser_use": False,
    "features.computer_use": False,
    "features.image_generation": False,
    "features.sleep_tool": False,
    "features.tool_suggest": False,
    "features.skill_search": False,
    "features.goals": False,
    "features.hooks": False,
}

TURNS = [
    ("hello", "Say hi in three words."),
    ("tool_bait", "Run the shell command `dir` and paste its output. "
                  "If you cannot run commands, reply exactly: NO_TOOLS"),
    ("memory", "What exactly did I ask you in my first message?"),
]


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dump(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, (list, tuple)):
        return [_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    return obj if isinstance(obj, (str, int, float, bool, type(None))) else repr(obj)


def _short(d, n=220) -> str:
    s = json.dumps(d, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


async def main() -> int:
    if os.environ.get("OPENAI_API_KEY"):
        print("!! OPENAI_API_KEY is set; unset it so usage bills to the ChatGPT plan.")
        return 2

    OUT_DIR.mkdir(exist_ok=True)
    log_path = OUT_DIR / f"probe_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    workspace = Path(tempfile.mkdtemp(prefix="conduix_ws_"))
    model = sys.argv[1] if len(sys.argv) > 1 else None

    with log_path.open("w", encoding="utf-8") as log:
        async with AsyncCodex() as codex:
            print("== metadata:", _short(_dump(codex.metadata), 400))

            acct = _dump(await codex.account())
            # Never print identity fields; only auth type / plan.
            def _scrub(x):
                if isinstance(x, dict):
                    return {k: ("<redacted>" if "email" in k.lower() else _scrub(v))
                            for k, v in x.items()}
                return x
            print("== account:", _short(_scrub(acct), 400))

            models = _dump(await codex.models())
            log.write(json.dumps({"kind": "models", "data": models}) + "\n")
            print("== models:", _short(models, 600))

            try:
                thread = await codex.thread_start(
                    approval_mode=ApprovalMode.deny_all,
                    sandbox=Sandbox.read_only,
                    cwd=str(workspace),
                    base_instructions=BASE_INSTRUCTIONS,
                    ephemeral=True,
                    model=model,
                    config=CHAT_ONLY_CONFIG,
                )
            except Exception as e:  # noqa: BLE001 - probe wants to see anything
                print(f"!! thread_start with chat-only config failed: {type(e).__name__}: {e}")
                return 1
            print(f"== thread: {thread.id}  workspace={workspace}")

            for label, prompt in TURNS:
                print(f"\n=== turn [{label}] {prompt!r}")
                counts: Counter[str] = Counter()
                text = []
                t0 = time.perf_counter()
                first_delta = None
                handle = await thread.turn(prompt, effort="low", summary="auto")
                async for n in handle.stream():
                    payload = _dump(n.payload)
                    counts[n.method] += 1
                    log.write(json.dumps({"turn": label, "method": n.method,
                                          "payload_type": type(n.payload).__name__,
                                          "payload": payload}, default=str) + "\n")
                    if "delta" in n.method.lower():
                        if first_delta is None:
                            first_delta = time.perf_counter() - t0
                        if n.method == "item/agentMessage/delta":
                            text.append(payload.get("delta", ""))
                        if counts[n.method] <= 2:
                            print(f"  {n.method}: {_short(payload, 160)}")
                    else:
                        print(f"  {n.method}: {_short(payload)}")
                dt = time.perf_counter() - t0
                print(f"  -- {dt:.1f}s, first delta at "
                      f"{first_delta and f'{first_delta:.1f}s'}; counts={dict(counts)}")
                if text:
                    print(f"  -- streamed text: {''.join(text)!r}")

    leftovers = list(workspace.iterdir())
    print(f"\n== workspace untouched: {not leftovers} {leftovers or ''}")
    print(f"== full log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
