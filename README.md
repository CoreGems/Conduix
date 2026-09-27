# Conduix

OpenAI-compatible local API server, powered by the official
[Codex Python SDK](https://github.com/openai/codex/tree/main/sdk/python)
(`openai-codex`).

Conduix exposes the **OpenAI wire format** (`/v1/responses`,
`/v1/chat/completions`) on `127.0.0.1:8766`. Every request is routed through
your **ChatGPT subscription** via the locally logged-in Codex session, not a
metered `OPENAI_API_KEY`. Any client that talks to `https://api.openai.com/v1`,
including the official `openai` Python and TypeScript SDKs, works with only a
`base_url` change.

It is the OpenAI-side twin of **[Conduit](https://github.com/CoreGems/Conduit)**,
which does the same for a Claude Max plan behind an Anthropic-compatible
`/v1/messages` API. The two are designed to run side by side, with Conduit on
`:8765` and Conduix on `:8766`.

> **Status: v1.1.** Everything below is built and tested: offline against a
> fake Codex app-server, and live against a ChatGPT plan through the official
> `openai` SDK. See [`BRIEF.md`](./BRIEF.md) §6 for the build log.

## Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="not-used")

r = client.responses.create(model="gpt-6-astra", input="Say hi in three words.")
print(r.output_text)
```

The [usage guide](./CONDUIX_API_USEAGE_GUIDE.md) covers every feature with
examples.

## Features

| Feature | Notes | Guide |
| ------- | ----- | ----- |
| **Responses API** and **Chat Completions** | Streaming and non-streaming, with OpenAI's exact event and chunk shapes | [§4–5](./CONDUIX_API_USEAGE_GUIDE.md#4-quick-start) |
| **Multi-turn**, three ways | Resend the history, `previous_response_id` (with branching), or the `session_id` extension | [§6](./CONDUIX_API_USEAGE_GUIDE.md#6-multi-turn-conversations) |
| **Models and reasoning effort** | `/v1/models` lists the plan's models and each one's `effort` values; reasoning summaries | [§7](./CONDUIX_API_USEAGE_GUIDE.md#7-models-and-reasoning-effort) |
| **Images** | Base64 data URLs (PNG / JPEG / GIF / WebP), remembered across turns | [§8](./CONDUIX_API_USEAGE_GUIDE.md#8-images) |
| **Structured output** | `json_schema` via `text.format` / `response_format`, and JSON mode (`json_object`) | [§9](./CONDUIX_API_USEAGE_GUIDE.md#9-structured-output-json-schema) |
| **Function tools** | Client-executed function calling on both endpoints, streaming included | [§10](./CONDUIX_API_USEAGE_GUIDE.md#10-function-tools) |
| **Web search** | `tools: [{"type": "web_search"}]`, or `web_search_options` in Chat Completions | [§11](./CONDUIX_API_USEAGE_GUIDE.md#11-web-search) |
| **Usage** | OpenAI-style token counts; `/health` shows how much of the plan's usage window is used | [§12](./CONDUIX_API_USEAGE_GUIDE.md#12-usage-and-token-counts) |
| **Errors** | OpenAI's error envelope. Quota errors are a 429 with the reset time that the SDK doesn't retry. Failures while streaming arrive as events | [§13](./CONDUIX_API_USEAGE_GUIDE.md#13-errors) |

A function-calling loop looks exactly like it does against OpenAI:

```python
import json

tools = [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]

r = client.responses.create(model="gpt-6-astra", input="Weather in Kyiv?", tools=tools)
while calls := [o for o in r.output if o.type == "function_call"]:
    r = client.responses.create(
        model="gpt-6-astra", tools=tools, previous_response_id=r.id,
        input=[{"type": "function_call_output", "call_id": c.call_id,
                "output": get_weather(**json.loads(c.arguments))} for c in calls],
    )
print(r.output_text)
```

## Endpoints

| Method | Path                    | Purpose                                   |
| ------ | ----------------------- | ----------------------------------------- |
| POST   | `/v1/responses`         | OpenAI Responses API (primary)            |
| POST   | `/v1/chat/completions`  | OpenAI Chat Completions                   |
| GET    | `/v1/models`            | Models available on your plan             |
| GET    | `/v1/models/{id}`       | One model, with its `effort` values       |
| GET    | `/v1/sessions`          | List active sessions                      |
| POST   | `/v1/sessions`          | Create session → `{session_id}`           |
| DELETE | `/v1/sessions/{id}`     | Tear down session                         |
| GET    | `/health`               | Liveness, Codex login, plan and usage     |
| GET    | `/docs`                 | Swagger UI                                |

## Prerequisites

- Conda (or any Python 3.11 environment)
- A ChatGPT account logged in to Codex once (`codex login`). The SDK reuses
  that session from `~/.codex`.
- No manual `OPENAI_API_KEY` handling is needed. The server removes it (and
  `CODEX_API_KEY`) from its own environment before starting Codex, and
  refuses to start unless Codex is logged in with a ChatGPT account, so
  usage always bills to the subscription.

## Setup

```powershell
conda create -n conduix python=3.11 -y
conda run -n conduix pip install -e ".[dev]"
```

The SDK pins and installs its own matching Codex CLI binary.

### Run

```powershell
.\start_app.ps1            # kill whatever holds :8766 (and its Codex child), then start
.\start_app.ps1 -Reload    # dev mode
.\start_app.ps1 -Port 9000
```

### Configuration

Environment variables, or a `.env` file in the project root:

| Variable                         | Default |
| -------------------------------- | ------- |
| `CONDUIX_HOST` / `CONDUIX_PORT`  | `127.0.0.1` / `8766` |
| `CONDUIX_DEFAULT_MODEL`          | Codex's default (`gpt-6-astra`) |
| `CONDUIX_DEFAULT_EFFORT`         | The model's default effort |
| `CONDUIX_DEFAULT_INSTRUCTIONS`   | None |
| `CONDUIX_WEB_SEARCH_MODE`        | `live` (or `cached`) |
| `CONDUIX_CODEX_BIN`              | The SDK's bundled Codex binary |
| `CONDUIX_WORKSPACE_DIR`          | `%LOCALAPPDATA%\conduix\workspace` (kept empty; Codex's sandbox root) |
| `CONDUIX_SESSION_IDLE_TIMEOUT_S` | `1800` |
| `CONDUIX_MAX_SESSIONS`           | `100` |

### Tests

```powershell
conda run -n conduix python -m pytest          # offline: no Codex, no quota
```

The offline suite runs the real Codex SDK against a fake app-server
(`tests/fake_app_server.py`) and drives every route with the official
`openai` client. The live tests spend plan quota, so they only run when asked:

```powershell
.\start_app.ps1                                # in another terminal
conda run -n conduix python -m pytest -m integration
```

### Probe the SDK

This sends three short chat-only turns through Codex and logs every event to
`scratch/`:

```powershell
conda run -n conduix --no-capture-output python scripts/probe_sdk.py
```

## Architecture

```
your client (openai SDK / rust / curl)
        │  HTTP + SSE  (OpenAI wire format)
        ▼
FastAPI ──► SessionManager ──► AsyncCodex (openai-codex SDK)
                                     │  JSON-RPC over stdio
                                     ▼
                              codex app-server
                                     │
                                     ▼
                        ChatGPT subscription (codex login)
```

- **Only `conduix/backend.py` touches the Codex SDK.** An SDK upgrade should
  only need changes there.
- **Codex runs in a locked-down mode.** Approvals are declined, the sandbox is
  read-only in an empty workspace, and Codex's own agent tools (shell, file
  edits, MCP) are off. It behaves like a model endpoint, not a coding agent.
  `/health` counts any blocked attempt (`blocked_agent_items`).
- **Earlier turns are real messages.** Conversation history is put into Codex
  threads as user, assistant and tool items, not pasted into a prompt.
- **Function tools use Codex's dynamic tools.**
  1. When the model calls a function, the response ends with the call and
     the Codex turn is stopped.
  2. The client's result comes back in the next request.
  3. Conduix replays the conversation on a fresh thread, and the model
     continues from the result.

## Docs

- [`CONDUIX_API_USEAGE_GUIDE.md`](./CONDUIX_API_USEAGE_GUIDE.md): how clients
  connect and use the API
- [`BRIEF.md`](./BRIEF.md): goals, architecture, feature map, protocol
  findings, build log, open questions
- [Conduit](https://github.com/CoreGems/Conduit): the Claude/Anthropic
  sibling project this one mirrors

## Limitations

- **Local, single-user.** It binds to `127.0.0.1` and has no auth. Don't
  expose it to a network.
- **Counts against your plan limits.** Each turn carries about 4.4k tokens of
  fixed Codex prompt overhead, much of which is cached.
- **In-memory state.** Sessions and `previous_response_id` history are lost
  on restart. Resending the history always works.
- **Function tools:**
  - One call comes back per response.
  - `tool_choice: "required"` behaves like `"auto"`.
  - Tools can't be combined with `session_id`.
- **Ignored parameters:** sampling parameters (`temperature`, `top_p`,
  `max_output_tokens`, ...) are accepted and ignored, because Codex has no
  setting for them.
- **Experimental protocol:** Codex's app-server protocol is marked
  experimental upstream, so pin the SDK version and upgrade deliberately.
