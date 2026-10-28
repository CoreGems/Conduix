# ConduiG — Brief

**Gemini-compatible local API server, powered by your Google AI Pro
subscription via the Antigravity CLI (`agy`).**

> Drafted in the Conduix repo on 2026-10-28. Move it to
> `D:\mygems\ConduiG\BRIEF.md` when that project is created.

ConduiG is the third member of the family:

| Project  | Wire format exposed            | Backend                          | Billed to            | Port   |
| -------- | ------------------------------ | -------------------------------- | -------------------- | ------ |
| Conduit  | Anthropic `/v1/messages`       | `claude-agent-sdk`               | Claude Max plan      | `8765` |
| Conduix  | OpenAI `/v1/responses`, `/v1/chat/completions` | `openai-codex` SDK | ChatGPT plan         | `8766` |
| **ConduiG** | **Gemini `/v1beta/models/{m}:generateContent`** | **`agy` headless, stream-json** | **Google AI Pro plan** | **`8767`** |

It exposes the **Gemini API wire format** and routes every request through
Google's own **Antigravity CLI** running in headless mode, using the login
`agy` caches after a normal interactive sign-in. Any client that talks to
`https://generativelanguage.googleapis.com`, including the official
`google-genai` Python/TS SDKs, should work with only a `base_url` change.

> **Status: design only (2026-10-28).** Nothing is built. Facts marked
> *(docs)* come from Google's published docs as of this date and have
> **not** been checked against a real `agy`. Everything marked
> **PROBE** is unknown until step 2.

---

## 0. Why this is shaped differently from Conduit / Conduix

Read this before anything else. It explains every odd decision below.

1. **There is no subscription-backed SDK.** Conduit and Conduix each sit on
   an official SDK that uses the subscription login. Google's equivalent,
   the Antigravity SDK (`google-antigravity` on PyPI), accepts **only**
   `GEMINI_API_KEY` or Vertex credentials. Google-account login for it is an
   open feature request (`antigravity-sdk-python#20`) with no commitment.
2. **Gemini CLI is no longer an option.** Since 2026-06-18, Gemini CLI's login
   no longer serves the free, AI Pro and Ultra tiers. Only Code Assist
   Standard/Enterprise licences still work. `agy` replaced it and does accept
   AI Pro.
3. **Therefore the backend is a CLI process, not a library.** ConduiG drives
   `agy -p --input-format stream-json --output-format stream-json` over
   stdin/stdout. That is closer to Conduix's `codex app-server` fallback than
   to its SDK path: we own process lifecycle, framing and error mapping.
4. **Terms of service are a first-class risk** (§8). Google states that using
   Gemini CLI login from third-party software to reach its backend violates
   its terms, and has restricted paying subscribers' accounts for routing
   that login through proxies. ConduiG **never touches tokens and never calls
   Google's backend itself**: it only runs Google's unmodified `agy` binary,
   on the user's own machine, for the user's own programs. That is the most
   defensible shape available, but it is **not** a sanctioned use, and it is the
   user's own Google account at stake.

---

## 1. Goals

1. **Wire-format parity with the Gemini API.** Request/response shapes and
   the SSE stream close enough that `google-genai` parses them without
   complaint.
2. **Subscription-backed only.** Use the auth `agy` cached from an
   interactive login. Never fall back to an API key; strip key variables from
   the child environment so `agy` can't silently pick one up.
3. **Never handle credentials.** ConduiG does not read, copy, refresh or
   forward OAuth tokens. It does not call any Google endpoint directly. The
   only thing that talks to Google is `agy`.
4. **Local, single-user.** Bind to `127.0.0.1`, no auth, same as the siblings.
5. **Behave like a plain model.** `agy` is a coding agent. Every built-in tool
   is denied, by more than one independent mechanism (§3.3).
6. **Legible failures.** Quota → a Gemini-shaped `429 RESOURCE_EXHAUSTED` with
   a retry delay when known. Not logged in → a clear `503` telling the user to
   run `agy` once interactively. Never an empty 200.
7. **Swappable backend.** Everything `agy`-specific stays behind one
   `Backend` interface, so an API-key backend (the Antigravity SDK, "path B")
   can be dropped in later without touching routes or streaming.

## Non-goals (v1)

- Being a coding agent (no shell, files, browser, subagents).
- Embeddings, `countTokens`, the Files API, cached content, Live API, image
  or audio generation, tuned models, batch.
- Any direct call to Google APIs, including Code Assist (`cloudcode-pa`).
- Network exposure, multiple users, or running unattended at high volume.
- An OpenAI-compatible surface. Conduix already does that. Clients that want
  OpenAI shape use Conduix.

---

## 2. Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST | `/v1beta/models/{model}:generateContent` | Gemini generateContent (non-streaming) |
| POST | `/v1beta/models/{model}:streamGenerateContent?alt=sse` | Streaming. Only `alt=sse` (what `google-genai` sends); any other form → 400 |
| GET  | `/v1beta/models` | Models the plan can use, from `agy models`, in Gemini `Model` shape |
| GET  | `/v1beta/models/{model}` | One model |
| POST | `/v1beta/models/{model}:countTokens` | **501**, documented as unsupported |
| GET  | `/v1/sessions` | List active sessions (ConduiG extension) |
| POST | `/v1/sessions` | Create empty session → `{session_id}` |
| DELETE | `/v1/sessions/{id}` | Tear down session (kills its `agy` process) |
| GET  | `/health` | Liveness, `agy` version, login status, blocked-tool counter |
| GET  | `/docs` | Swagger UI |

The same `models/...` routes are also mounted under `/v1/` because
`google-genai` can be configured with `api_version="v1"`.

**Sessions extension.** The Gemini API has no session field and no
`previous_response_id`. ConduiG accepts an optional session id via the
**`X-ConduiG-Session` header** (settable through `HttpOptions(headers=...)`)
or a top-level `session_id` body field. Without it, requests are stateless
like the real API. Multi-turn for ordinary clients goes through history
replay with a resume cache (§3.4).

Incoming `x-goog-api-key` / `key=` values are ignored and never logged.

---

## 3. Architecture

```
your client (google-genai / rust / curl)
        │  HTTP + SSE  (Gemini wire format)
        ▼
FastAPI ──► SessionManager ──► AgyPool ──► agy process (one per active conversation)
                                   │        `agy -p --input-format stream-json
                                   │              --output-format stream-json ...`
                                   │        cwd = empty workspace, isolated settings,
                                   │        deny-all permissions + deny-all hook
                                   ▼
                          NDJSON events on stdout  (init / step_update / result)
                                   │
                                   ▼
                     Google AI Pro subscription (agy's own cached login)
```

### 3.1 What `agy` headless gives us *(docs)*

- `-p/--prompt`, `--output-format text|json|stream-json`,
  `--input-format text|stream-json`, `--model`, `--effort low|medium|high`,
  `--agent`, `--continue`, `--conversation <id>`, `--json-schema`,
  `--print-timeout` (default `5m`), `--sandbox`, `--mode accept-edits|auto|plan`,
  `--add-dir`, `--dangerously-skip-permissions` (**never used**).
- **Multi-turn in one process:** with both formats set to `stream-json`,
  user turns are written to stdin as
  `{"event":"user","message":{"content":"..." | [{"type":"text","text":"..."}]}}`.
  **Only `text` content blocks are supported** (others → exit code 1).
- **Output events (NDJSON):**
  - `init`: `conversation_id`, `init.{cwd, tools[], permission_mode, model, agent, json_schema}`
  - `step_update`: `{conversation_id, step_index, state: ACTIVE|DONE,
    step_type: user_input|agent_response|tool|checkpoint, tool_name,
    text_delta, duration_seconds, usage{...}, tool_info{name, parameters,
    output, error}, subagent_info}`
  - `result`: `{conversation_id, status: SUCCESS|ERROR|CANCELED|INTERRUPTED|INVALID|WAITING|RUNNING,
    response, error, duration_seconds, num_turns, structured_output,
    json_schema, usage{input_tokens, output_tokens, thinking_tokens,
    cache_read_tokens, total_tokens}}`
- **Exit codes:** 0 success, 1 error (bad JSON, unsupported content), 2
  critical (unknown slash/control request).
- **Auth:** headless uses cached credentials. With none, it exits with
  `authentication required` instead of hanging.
- **Unknown model** → non-zero exit, no silent fallback.
- **Permissions:** `~/.gemini/antigravity-cli/settings.json` →
  `permissions.{allow, ask, deny}` with `action(target)` rules. Actions:
  `read_file`, `write_file`, `read_url`, `execute_url`, `command`, `mcp`,
  `unsandboxed`. Precedence deny > ask > allow. Default: file ops inside the
  workspace **auto-approve**; commands soft-deny (continue with a stderr
  notice).
- **Hooks:** `PreToolUse`, `PostToolUse`, `PreInvocation`, `PostInvocation`,
  `Stop`. A `PreToolUse` hook can answer `deny`.

**Not documented, so a PROBE item:** a system-prompt flag, a way to disable
specific tools, a settings-dir / home override, image attachment in headless
mode, an interrupt/cancel control message, a login-status command, and the
custom-agent definition format.

### 3.2 Module mapping

| Conduix (Codex) | ConduiG (agy) |
| --------------- | -------------- |
| `openai-codex` SDK, one `AsyncCodex` | `AgyPool`: `asyncio.create_subprocess_exec` per conversation, our own NDJSON reader |
| `AsyncThread` per session | one long-lived `agy` process per session, fed turns over stdin |
| ephemeral thread for stateless requests | a short-lived process, **or** reuse via the resume cache (§3.4) |
| `thread_resume` / `inject_items` | `--conversation <id>` resume; no known item injection, so history is replayed as text (PROBE) |
| `streaming.py` synthesizes OpenAI SSE | `streaming.py` synthesizes Gemini SSE from `step_update.text_delta` |
| dynamic tools + interrupt | v1.1: a ConduiG-hosted MCP server + kill/resume (§7) |
| `codexErrorInfo` → 429 | `result.status/error` + stderr + exit code → 429/503/500. Probably **string matching**, unlike Conduix (PROBE) |
| `reasoning.effort` | `generationConfig.thinkingConfig.thinkingLevel` → `--effort` |
| `output_schema` | `--json-schema` (a **per-process** flag, see §4) |
| `blocked_agent_items` | same counter: every `step_type: tool` step and every hook denial |

### 3.3 Locking `agy` down to a chat model

Defence in depth. **Any one layer failing must not be enough** to touch disk
or run a command.

1. **Isolated config.** ConduiG must not edit the user's own
   `~/.gemini/antigravity-cli/settings.json` (that would also lock down their
   interactive `agy`). Preferred: run the child with its own config root
   (a home or settings override, `%LOCALAPPDATA%\conduig\agy-home`). PROBE
   whether `agy` honours `USERPROFILE`/`HOME`/an env var for this.
   - If it does, the user does a **one-time interactive `agy` login inside
     that isolated home** (`scripts/login.ps1`). ConduiG never copies
     credentials between homes; that would edge toward the "credential
     harvesting" the terms prohibit.
   - If it doesn't, fall back to a clearly marked block **in the user's
     settings** scoped by workspace path (if rules support path scoping), or
     to the hook layer alone. Decide after the probe.
2. **Permissions: deny everything.** `deny` contains a wildcard for every
   action: `command`, `read_file`, `write_file`, `read_url`, `execute_url`,
   `mcp`, `unsandboxed`. The exact wildcard syntax is a PROBE item. Nothing in
   `allow` or `ask`.
3. **`PreToolUse` hook that always answers `deny`.** A tiny script
   (`conduig/hooks/deny_all.py`) that also appends to a counter file, so
   `/health` can report `blocked_agent_items`. This is the hard backstop if
   the permission syntax turns out to be wrong.
4. **Empty workspace, sandbox on.** `cwd` = `%LOCALAPPDATA%\conduig\workspace`
   (empty, never the server's cwd), no `--add-dir`, `--sandbox` on, never
   `--dangerously-skip-permissions`. PROBE whether `--mode plan` also stops
   edits without changing answer quality.
5. **Stream filter.** Any `step_update` with `step_type: tool` (or
   `subagent_info`) is dropped from the client stream and counted. Tests
   assert the count stays zero for normal prompts and that the workspace stays
   empty after tool-bait prompts ("run `dir`", "write hello.txt", "open
   https://…").

### 3.4 Conversation state and the resume cache

`agy` is stateful per conversation. The Gemini API is stateless: clients
resend the whole `contents` every turn. A cold process per request with
the whole history flattened into text would be slow and would bill the
history every time.

- **Session mode (`X-ConduiG-Session`).** One long-lived process per
  session. Each request sends **only the last user turn** to stdin. A
  per-session `asyncio.Lock` serializes turns. Idle eviction kills the
  process but keeps `conversation_id`, so a later request restarts it
  with `--conversation <id>` (cheap eviction, PROBE that resume restores
  context).
- **Stateless mode with a resume cache.** After each turn, store
  `hash(contents including the model's reply) → conversation_id`. On the
  next request, if `hash(contents[:-1])` matches, resume that conversation and
  send only `contents[-1]`. This is how `google-genai`'s `chats` (which
  resends history) gets real continuity. Cache in memory, LRU-bounded.
- **Cache miss with history.** Start a fresh conversation and send one user
  turn containing a flattened transcript (`systemInstruction` + prior turns,
  clearly delimited) followed by the new message. This is lossy (the model
  sees its own past replies as quoted text, and thoughts are gone), so it's
  documented. PROBE whether there is any real history-injection mechanism.
- **Warm pool (optimization, after v1 works).** If process start-up latency is
  large (measured in step 2), keep N idle pre-spawned processes for the
  default model/effort/schema combination.

---

## 4. Request / response mapping

### Request (`GenerateContentRequest`)

| Gemini field | ConduiG handling |
| ------------ | ----------------- |
| `contents[]` | Last item must be `role: user` with text parts (else 400). Earlier items → resume cache or flattened transcript (§3.4) |
| `contents[].parts[].text` | Passed through |
| `parts[].inlineData` (images) | **PROBE.** stdin takes text blocks only. Candidate: write the image into a per-request inbox dir and allow `read_file` for that dir only. That loosens the lockdown, so v1 returns **400 `INVALID_ARGUMENT`** until it's proven safe |
| `parts[].fileData` (URIs) | 400 |
| `parts[].functionCall/functionResponse` | 400 in v1; v1.1 (§7) |
| `systemInstruction` | **PROBE.** No flag. Options, best first: (a) a generated custom agent (`--agent`) whose definition carries the prompt, if the format allows it; (b) a preamble in the first user turn. Changing it mid-session forces a new conversation |
| `generationConfig.thinkingConfig.thinkingLevel` | `LOW/MEDIUM/HIGH` → `--effort low/medium/high`; others → 400. A per-process flag: a different effort on an existing session restarts the process with `--conversation` |
| `thinkingConfig.thinkingBudget` | Bucketed to an effort level (`0`/small → low, …; thresholds set after the probe). `-1` (dynamic) → default |
| `thinkingConfig.includeThoughts` | Thought parts (`thought: true`) only if `agy` emits thought text (PROBE). Otherwise accepted and ignored; `thoughtsTokenCount` is still reported |
| `responseMimeType: application/json` + `responseSchema` / `responseJsonSchema` | `--json-schema`; reply text = `json.dumps(result.structured_output)`. Per-process flag, so stateless only in v1 (400 with a session) |
| `responseMimeType: application/json`, no schema | JSON mode: an instruction on that turn + code-fence stripping while streaming (reuse Conduix's `json_safe_prefix`). Not guaranteed |
| `temperature`, `topP`, `topK`, `maxOutputTokens`, `stopSequences`, `seed`, `presencePenalty`, `frequencyPenalty` | Accepted, **ignored**, listed in the README. `stopSequences` could be applied client-side by truncating the stream (v1.1) |
| `candidateCount > 1` | 400 |
| `safetySettings` | Accepted and ignored |
| `tools: [{googleSearch: {}}]` | v1.1 (§7). 400 in v1 |
| `tools: [{functionDeclarations}]`, `toolConfig` | v1.1 (§7). 400 in v1 |
| `cachedContent` | 400 |
| `{model}` path segment | Validated against `agy models`; unknown → 404 `NOT_FOUND`. `models/` prefix tolerated |

### Response (`GenerateContentResponse`)

```json
{
  "candidates": [{
    "content": {"role": "model", "parts": [{"text": "..."}]},
    "finishReason": "STOP",
    "index": 0
  }],
  "usageMetadata": {
    "promptTokenCount": 0, "candidatesTokenCount": 0,
    "thoughtsTokenCount": 0, "cachedContentTokenCount": 0,
    "totalTokenCount": 0
  },
  "modelVersion": "<agy model slug>",
  "responseId": "<conversation_id>:<step_index>"
}
```

- **Usage mapping:** `promptTokenCount` ← `input_tokens` (PROBE whether it
  already includes `cache_read_tokens`; Gemini's count does include cached
  tokens), `candidatesTokenCount` ← `output_tokens`, `thoughtsTokenCount` ←
  `thinking_tokens`, `cachedContentTokenCount` ← `cache_read_tokens`,
  `totalTokenCount` ← `total_tokens`. PROBE whether `result.usage` is
  **per turn or cumulative** for a multi-turn process. Conduix had exactly
  this bug: if it's cumulative, compute the delta.
- **finishReason:** `SUCCESS` → `STOP`; `--print-timeout` hit → `OTHER` with a
  note in the error log; `CANCELED`/`INTERRUPTED` → `OTHER`. `MAX_TOKENS` is
  never produced (no output cap is forwarded).

### Streaming (`:streamGenerateContent?alt=sse`)

- One `data: <GenerateContentResponse>` chunk per
  `step_update{step_type: agent_response, text_delta}`, each carrying only the
  new text in `candidates[0].content.parts[0].text`.
- Final chunk: empty or last text, `finishReason`, `usageMetadata` (from
  `result`), `modelVersion`.
- PROBE: are `text_delta`s token-level or step-sized? If coarse, streaming
  still works, it just arrives in larger chunks. Document it, don't fake it.
- **Errors mid-stream:** sent as a final `data: {"error": {...}}` chunk in the
  Google error shape, never as a dropped connection (Conduit `a7ce494`
  lesson). Step 10 must confirm `google-genai` raises a proper `APIError`
  from that chunk. If not, find the shape it does recognize.

### Errors (Google envelope)

```json
{"error": {"code": 429, "message": "...", "status": "RESOURCE_EXHAUSTED",
           "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3600s"}]}}
```

| Condition | HTTP / `status` |
| --------- | --------------- |
| Plan quota / rate limit | 429 `RESOURCE_EXHAUSTED`, `RetryInfo` if a reset time is known |
| `authentication required` / not logged in | 503 `UNAVAILABLE`, message: "run `scripts/login.ps1` (interactive `agy` login) once" |
| Unknown model | 404 `NOT_FOUND` |
| Unsupported field / content | 400 `INVALID_ARGUMENT`, naming the field |
| Turn timeout | 504 `DEADLINE_EXCEEDED` |
| `agy` crash / exit 2 / unparseable output | 500 `INTERNAL`, stderr tail in the server log only |
| Too many live processes | 429 `RESOURCE_EXHAUSTED` with `retryDelay: "1s"` (local back-pressure, distinct message) |

Quota classification will probably have to match on `result.error` / stderr
text, because no structured error enum is documented. Keep every pattern in
one table in `errors.py`, and record the real strings in step 2 / step 8.

---

## 5. Stack, layout, config

Same stack as the siblings: Python 3.11, FastAPI, uvicorn, `sse-starlette`,
pydantic-settings. **`google-genai`** is the source of truth for wire types
(`google.genai.types`), the equivalent of Conduix re-exporting `openai`
types. Conda env `conduig`. `agy` is an external prerequisite (not pip
installable), version recorded on `/health` and pinned in the README.

```
conduig/
  app.py            FastAPI app; lifespan checks agy, login, settings; starts/stops AgyPool
  config.py         CONDUIG_* settings
  schema.py         re-export google.genai types + session extension
  agy.py            AgyProcess: spawn (clean env, cwd, flags), NDJSON reader, stdin writer, kill/resume
  backend.py        Backend interface + AgyBackend (pool, per-turn event stream → internal events)
  lockdown.py       writes the isolated settings.json + hook config; verifies them at start-up
  hooks/deny_all.py PreToolUse hook: always deny, bump counter
  sessions.py       SessionManager, locks, idle eviction, resume cache
  streaming.py      internal events → Gemini SSE chunks / collected response
  errors.py         agy failures → Google error envelope (pattern table lives here)
  routes/generate.py
  routes/models.py
  routes/sessions.py
tests/              offline (fake agy) + integration (-m integration, through google-genai)
  fake_agy.py       speaks agy's stream-json, replaying NDJSON recorded in the probe
scripts/            probe_agy.py (step 2), login.ps1, demos
scratch/            probe logs (gitignored)
start_app.ps1       kill the whole process tree (incl. agy children) and restart on port
```

### Config (`CONDUIG_*`, env or `.env`)

| Variable | Default |
| -------- | ------- |
| `CONDUIG_HOST` / `CONDUIG_PORT` | `127.0.0.1` / `8767` |
| `CONDUIG_AGY_BIN` | `agy` on `PATH` |
| `CONDUIG_AGY_HOME` | `%LOCALAPPDATA%\conduig\agy-home` (isolated config root, if supported) |
| `CONDUIG_WORKSPACE_DIR` | `%LOCALAPPDATA%\conduig\workspace` |
| `CONDUIG_DEFAULT_MODEL` | *(agy's default)* |
| `CONDUIG_DEFAULT_EFFORT` | *(none)* |
| `CONDUIG_DEFAULT_SYSTEM_INSTRUCTION` | *(none)* |
| `CONDUIG_TURN_TIMEOUT_S` | `600` (passed as `--print-timeout`, plus our own watchdog) |
| `CONDUIG_MAX_PROCESSES` | `6` (live `agy` processes, all kinds) |
| `CONDUIG_MAX_SESSIONS` | `20` |
| `CONDUIG_SESSION_IDLE_TIMEOUT_S` | `900` (process killed; conversation resumable) |
| `CONDUIG_RESUME_CACHE_SIZE` | `200` |
| `CONDUIG_MAX_REQUESTS_PER_HOUR` | *(unset)*. Optional self-imposed cap (§8) |

### Billing / credential guard

- Build the child environment explicitly: copy `os.environ`, then
  **remove** `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI`,
  `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`,
  `GOOGLE_CLOUD_LOCATION`. Unlike Codex, a subprocess env can be set exactly,
  so this happens per spawn.
- At start-up, run a trivial headless turn (or a login-status command if one
  exists, PROBE) and refuse to serve if it reports API-key auth or no auth.
  Show the account/plan on `/health` if `agy` exposes it.
- ConduiG never opens, parses or logs anything under the `agy` credential
  store.

---

## 6. Build sequence

Each step is independently verifiable. Don't start step N+1 until step N
passes.

1. **Bootstrap.** Install `agy` (not installed on this machine as of
   2026-10-28), log in interactively with the AI Pro account, confirm
   `agy -p "hi"` works. pyproject, conda env, `/health`.
2. **Probe** (`scripts/probe_agy.py`, logs in `scratch/`). This is the most
   important step: several design choices are open until it's done. Answer and
   record in a new §3.5:
   - [ ] Isolated config root: does `HOME`/`USERPROFILE`/an env var relocate
         settings **and** login? Does login work inside it?
   - [ ] Deny-all wildcard syntax per action; does a tool-bait prompt touch
         nothing? Does the deny-all hook fire and block?
   - [ ] What `init.tools[]` lists. Can tools be removed, not just denied?
   - [ ] Hidden prompt overhead per turn (input tokens for "hi").
   - [ ] `text_delta` granularity and time to first delta; process start-up time.
   - [ ] Multi-turn over stdin: one `result` per turn? Is usage per turn or cumulative?
   - [ ] `--conversation` resume restores context after the process is killed.
   - [ ] System prompt: custom agent format, or preamble only?
   - [ ] Images: any headless path at all?
   - [ ] Thought text in events? `--effort` values per model; `agy models` output shape.
   - [ ] `--json-schema` behaviour, incl. an invalid schema and `{}`-only schemas.
   - [ ] Interrupt: is there a stdin control message, or is kill the only way?
   - [ ] Auth-missing and (if it can be seen without burning quota) quota-error shapes:
         exit code, `result.status`, `result.error`, stderr.
   - [ ] Where `agy` writes conversation state, and whether it grows unbounded.
3. **`agy.py` + `backend.py`:** spawn, read, write, kill, resume; internal
   event type; clean env.
4. **`lockdown.py` + hook**, verified at start-up (refuse to serve if the
   lockdown can't be confirmed).
5. **Sessions:** process per session, locks, eviction, resume cache.
6. **Streaming + collector** for generateContent.
7. **Routes:** `generateContent`, `streamGenerateContent`, `models`, sessions.
8. **Errors:** pattern table, mid-stream error chunks.
9. **JSON mode + `--json-schema`.**
10. **Tests:** `fake_agy.py` replaying probe NDJSON; offline unit tests assert
    the exact flags/env/settings each spawn gets. Integration tests through
    `google-genai`, including tool bait with an empty-workspace assertion.
11. **v1.1:** function tools, Google Search, images if the probe allows (§7).

### Acceptance for v1

```python
from google import genai
from google.genai import types

c = genai.Client(api_key="not-used",
                 http_options=types.HttpOptions(base_url="http://127.0.0.1:8767"))
MODEL = "<a slug from `agy models`>"

print(c.models.generate_content(model=MODEL, contents="Say hi in three words.").text)

for ch in c.models.generate_content_stream(model=MODEL, contents="Count to five."):
    print(ch.text or "", end="")

chat = c.chats.create(model=MODEL)          # resends history → resume cache
chat.send_message("My name is Ada.")
print(chat.send_message("What is my name?").text)   # must say Ada
```

All three work, streaming and non-streaming. The third hits the resume cache
(visible in the server log) rather than a flattened transcript. Tool-bait
prompts leave the workspace empty and `blocked_agent_items` explains any
attempt.

---

## 7. Open questions / risks

- **Everything marked PROBE.** In particular, if neither an isolated config
  root nor path-scoped rules exist, the lockdown rests on the hook alone.
  Decide then whether that is acceptable.
- **CLI churn.** `agy` is months old and replaced its predecessor abruptly.
  Pin the version in the README, check it at start-up (warn on mismatch), and
  keep every `agy` detail in `agy.py` / `backend.py`.
- **Process cost.** A process per conversation is heavier than an SDK thread.
  `CONDUIG_MAX_PROCESSES` keeps memory bounded. Measure in step 2.
- **History fidelity.** Without item injection, a cache miss means a
  flattened transcript. Acceptable for chat; poor for long agent loops.
- **Hidden overhead and quota.** Like Codex (~4.4k tokens/turn), `agy` likely
  sends a large agent preamble. It counts against plan limits.
- **Function tools (v1.1).** Design sketch: ConduiG hosts a local MCP server
  exposing the request's `functionDeclarations`, registered in the isolated
  config, with **only** `mcp(conduig/*)` allowed. When the model calls one,
  ConduiG emits a `functionCall` part, ends the response, and kills or
  interrupts the process; there is no extra model call. The client's
  `functionResponse` comes back in resent `contents`. It is resumed via the resume
  cache (`--conversation`) with the result delivered as the MCP tool's answer
  if the process is still parked, otherwise as text. Open: whether a parked
  MCP call can be held across HTTP requests safely, versus Conduix's
  "always interrupt" approach.
- **Google Search (v1.1).** `tools: [{googleSearch: {}}]` → allow `agy`'s
  search/`read_url` for that turn only. Return results as text; full
  `groundingMetadata` only if `agy` exposes sources.
- **Images (v1.1).** Only if a headless path exists that doesn't require
  loosening file access beyond a per-request inbox.
- **Quota signal** is likely string-based and has to be captured live without
  burning the plan.

---

## 8. Terms of service and account safety

This section is part of the design. It is not a disclaimer.

- **What Google says:** using third-party software to piggyback on Gemini CLI
  / Code Assist login to reach its backend is a violation, and accounts
  (including paying AI Pro/Ultra subscribers) have been restricted for
  routing that login through proxies. `agy` succeeded Gemini CLI, and its
  terms should be assumed to be at least as strict.
- **What ConduiG does to stay on the defensible side:** it runs Google's
  own unmodified `agy`; it never reads, copies or replays tokens; it never
  calls Google endpoints itself; it binds to localhost; it serves one person's
  own programs.
- **What remains true:** it is still automation of a consumer subscription
  outside the product it was sold for. Google can treat it as a violation, and
  the account at risk is the user's main Google account.
- **Mitigations:**
  - Re-read the current Antigravity terms before step 1 and before each
    `agy` upgrade. Record the date and the clause in this file.
  - Keep volume human-scale. `CONDUIG_MAX_REQUESTS_PER_HOUR` exists for
    this. No 24/7 unattended loops, no fan-out of parallel agents.
  - Consider a **separate Google account** with its own AI Pro subscription
    for ConduiG, so a restriction doesn't hit Gmail, Drive, etc.
  - Keep the path-B backend (Antigravity SDK + `GEMINI_API_KEY`) as the
    fallback. Because of goal 7, switching is a config change, not a rewrite.

---

## References (checked 2026-10-28)

- Antigravity CLI headless mode: https://antigravity.google/docs/cli/headless/
- Antigravity CLI reference: https://antigravity.google/docs/cli/reference/
- Antigravity permissions: https://antigravity.google/docs/permissions/
- Antigravity SDK overview (API key / Vertex only): https://antigravity.google/docs/sdk/overview/
- SDK Google-account login request: https://github.com/google-antigravity/antigravity-sdk-python/issues/20
- Headless AI Pro login request: https://github.com/google-antigravity/antigravity-cli/issues/1194
- Gemini CLI service update (third-party login enforcement): https://github.com/google-gemini/gemini-cli/discussions/22970
- Gemini CLI terms: https://www.geminicli.com/docs/resources/tos-privacy
- Consumer-tier shutdown, 2026-06-18: https://github.com/NousResearch/hermes-agent/issues/29294
- Account restriction report: https://discuss.ai.google.dev/t/account-restricted-without-warning-google-ai-ultra-oauth-via-openclaw/122778/22
- Permissions/hooks write-up (third party): https://agenticcontrolplane.com/controls/antigravity
