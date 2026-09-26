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

> **Status: design + probe stage.** The server is not built yet. What exists
> today is the design ([`BRIEF.md`](./BRIEF.md)), the client guide for the
> planned API ([`CONDUIX_API_USEAGE_GUIDE.md`](./CONDUIX_API_USEAGE_GUIDE.md)),
> a working SDK probe (`scripts/probe_sdk.py`), and the launcher
> (`start_app.ps1`).

## Planned usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="not-used")

r = client.responses.create(model="gpt-6-astra", input="Say hi in three words.")
print(r.output_text)
```

The planned API covers streaming, multi-turn conversations
(`previous_response_id`, or the Conduix `session_id` extension), images,
reasoning effort, errors, and limits. See the
[usage guide](./CONDUIX_API_USEAGE_GUIDE.md) for all of these.

## Endpoints (v1)

| Method | Path                    | Purpose                                   |
| ------ | ----------------------- | ----------------------------------------- |
| POST   | `/v1/responses`         | OpenAI Responses API (primary)            |
| POST   | `/v1/chat/completions`  | OpenAI Chat Completions                   |
| GET    | `/v1/models`            | Models available on your plan             |
| GET    | `/v1/sessions`          | List active sessions                      |
| POST   | `/v1/sessions`          | Create session → `{session_id}`           |
| DELETE | `/v1/sessions/{id}`     | Tear down session                         |
| GET    | `/health`               | Liveness + Codex login / plan status      |
| GET    | `/docs`                 | Swagger UI                                |

## Prerequisites

- Conda (or any Python 3.11 environment)
- A ChatGPT account logged in to Codex once (`codex login`). The SDK reuses
  that session from `~/.codex`.
- No manual `OPENAI_API_KEY` handling is needed. `start_app.ps1` removes
  it (and `CODEX_API_KEY`) from the server's environment so usage always bills
  to the subscription.

## Setup

```powershell
conda create -n conduix python=3.11 -y
conda run -n conduix pip install "openai-codex==0.157.1"
```

The SDK pins and installs its own matching Codex CLI binary.

### Probe the SDK

This sends three short chat-only turns through Codex and logs every event to
`scratch/`:

```powershell
conda run -n conduix --no-capture-output python scripts/probe_sdk.py
```

### Run (once the server exists)

```powershell
.\start_app.ps1            # kill whatever holds :8766 (and its Codex child), then start
.\start_app.ps1 -Reload    # dev mode
.\start_app.ps1 -Port 9000
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

Codex runs in a locked-down, chat-only mode: approvals are denied, the sandbox
is read-only in an empty workspace, and the built-in tools are disabled. It
behaves like a plain model endpoint, not a coding agent.

## Docs

- [`BRIEF.md`](./BRIEF.md): goals, architecture, feature map, SDK probe
  findings, build sequence, open questions
- [`CONDUIX_API_USEAGE_GUIDE.md`](./CONDUIX_API_USEAGE_GUIDE.md): how clients
  connect and use the API
- [Conduit](https://github.com/CoreGems/Conduit): the Claude/Anthropic
  sibling project this one mirrors

## Limitations

- **Local, single-user.** It binds to `127.0.0.1` and has no auth. Don't
  expose it to a network.
- **Counts against your plan limits.** Each turn carries about 4.4k tokens of
  fixed Codex prompt overhead, which is mostly cached after the first turn.
- **v1 is chat only.** Function tools and web search are planned for v1.1.
- Codex's app-server protocol is marked experimental upstream, so pin the SDK
  version and upgrade deliberately.
