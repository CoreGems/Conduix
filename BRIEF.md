# Conduix — Brief

**OpenAI-compatible local API server, powered by your ChatGPT subscription via OpenAI Codex.**

Conduix is Conduit's twin: Conduit
(`C:\Users\alex_\mygems\Conduit`) puts an Anthropic `/v1/messages` surface in
front of the Claude Agent SDK so a Claude Max plan can be used from any
program. Conduix does the same for the **ChatGPT plan that Codex is logged
into**. (The step-2 probe on 2026-09-26 found `plan_type: "plus"` on the
current `codex login`. If Pro is the plan you meant to use, re-login first.) It exposes the
**OpenAI wire format** (`/v1/responses`, `/v1/chat/completions`) and routes
every request through the official **Codex Python SDK** (`openai-codex`),
which reuses the `codex login` ChatGPT session. Usage is billed to the
subscription instead of a metered `OPENAI_API_KEY`.

Any client that talks to `https://api.openai.com/v1` should work with only a
`base_url` change. That includes the official `openai` Python/TS SDKs, and
Rust via `reqwest` + SSE.

---

## 1. Goals

1. **Wire-format parity with OpenAI.** Match the request and response shapes and
   the SSE event names closely enough that the official `openai` SDK parses
   them without complaint.
2. **Subscription-backed only.** Use the auth that `codex login` stores
   (ChatGPT account) and never fall back to an API key without saying so.
3. **Local, single-user.** Bind to `127.0.0.1` only, with no auth, same as
   Conduit.
4. **Server-side sessions** are an optional Conduix extension (`session_id`).
   When the field is omitted, requests are stateless like the real API.
5. **Legible failures.** When the plan quota runs out, return a clean 429 with
   the reset time, as Conduit does. Never return an empty 200.
6. **Image generation** (added in v1.2). Generate images on the subscription
   through Codex's built-in image tool, in OpenAI's shapes: the
   `image_generation` tool on `/v1/responses`, and `/v1/images/generations`.
   It is off unless a request asks for it, like web search.

## Non-goals (v1)

- Being a general coding agent. Codex's own shell, file-edit, and patch tools
  are **disabled**, so Conduix behaves like a plain model endpoint.
- Embeddings, audio, fine-tuning, the Assistants API, and the other Images
  endpoints (`/v1/images/edits`, `/v1/images/variations`). *Exception:*
  speech output (`/v1/audio/speech`) was added in step 14 (§3.3), and
  transcription (`/v1/audio/transcriptions`) in step 15 (§3.4).
  Translation and audio input to chat turns stay non-goals.
- Exposing it on a network or supporting more than one user.
- Automatic Claude↔Codex failover. That idea lives in Conduit's
  `CODEX_INTEGRATION_HOWTO.md`. Conduix is the standalone half it would build on.

---

## 2. Endpoints

| Method | Path                    | Purpose                                              |
| ------ | ----------------------- | ---------------------------------------------------- |
| POST   | `/v1/responses`         | OpenAI Responses API (primary; closest to Codex's own model). Optional `session_id`. |
| POST   | `/v1/chat/completions`  | OpenAI Chat Completions (compat layer over the same core). Optional `session_id`. |
| POST   | `/v1/images/generations` | OpenAI Images API (a Responses turn with the image tool on, §3.2) |
| POST   | `/v1/audio/speech`      | OpenAI text-to-speech over Codex thread realtime (§3.3). Experimental, off unless `CONDUIX_SPEECH=1` |
| POST   | `/v1/audio/transcriptions` | OpenAI speech-to-text over the same realtime session (§3.4). Experimental, off unless `CONDUIX_TRANSCRIPTION=1` |
| GET    | `/v1/models`            | Models the logged-in plan can use (from the Codex model list). |
| GET    | `/v1/sessions`          | List active sessions                                 |
| POST   | `/v1/sessions`          | Create empty session → `{session_id}`                |
| DELETE | `/v1/sessions/{id}`     | Tear down session                                    |
| GET    | `/health`               | Liveness, plus Codex login and version status        |
| GET    | `/docs`                 | Swagger UI                                           |

The Responses API comes first because Codex already speaks it upstream:
reasoning items, `previous_response_id`-style continuity, and function
calls. Chat Completions is a thin translation layer on top of it, because most
third-party tools still target that endpoint.

`previous_response_id` (the native Responses API way to continue a
conversation) maps directly onto a Conduix session or Codex thread. That gives
real OpenAI clients multi-turn support without using the `session_id`
extension.

---

## 3. Architecture

```
your client (openai SDK / rust / curl)
        │  HTTP + SSE  (OpenAI wire format)
        ▼
FastAPI ──► SessionManager ──► AsyncCodex (one per server, many AsyncThreads)
                                  │   openai-codex Python SDK (official)
                                  ▼   JSON-RPC over stdio (managed by the SDK)
                          `codex app-server`  (bundled via openai-codex-cli-bin)
                                  │
                                  ▼
                     ChatGPT Pro subscription (codex login OAuth)
```

**Backend: the official `openai-codex` Python SDK** (`pip install openai-codex`,
source in `openai/codex` under `sdk/python`). It is the Codex counterpart of
`claude-agent-sdk`, so Conduix lines up with Conduit module for module. It
drives `codex app-server` internally, which is the transport Conduit's
`CODEX_INTEGRATION_HOWTO.md` §3 recommended. The SDK also takes care of the
JSON-RPC framing, process lifecycle, typed notifications, and retries.

What the SDK gives us directly (checked against 0.157.1):

- `AsyncCodex` / `AsyncThread` / `AsyncTurnHandle`. **Streams are routed by
  turn ID**, so a single client can run several sessions' turns at once.
- `thread_start(model, base_instructions, developer_instructions, sandbox,
  approval_mode, cwd, config, ephemeral)`. `ephemeral=True` suits stateless
  requests. `base_instructions` replaces the coding-agent persona.
- `thread.turn(input, effort, summary, output_schema, model, sandbox)`, which
  returns a handle with `.stream()`, `.interrupt()`, and `.steer()`.
  `thread.run(...)` returns a `TurnResult`: `final_response`, `items`, `usage`,
  `error`.
- `thread_resume` / `thread_fork`. These give `previous_response_id` without us
  keeping any history of our own.
- `ImageInput(url="data:image/...;base64,...")`. Remote HTTP URLs are
  **deprecated** upstream, which settles that open question: reject them.
- `output_schema` maps directly to Responses API `text.format` json_schema.
- `retry_on_overload`, `is_retryable_error`, `ServerBusyError`, and
  `TurnError` for error mapping.
- Login helpers (`login_chatgpt`, `login_chatgpt_device_code`) and
  account info, which feed the billing guard.

**Pinning:** the SDK depends on an exact `openai-codex-cli-bin==<same version>`,
so the Codex binary is pinned with the pip install. No manual version pinning
or schema vendoring is needed. Auth is still shared through `~/.codex`.

> ⚠️ Use **`openai-codex`**, not `openai-codex-sdk` on PyPI. The latter is a
> different package: v0.1.x, with no repo link, pinned to an old 0.88 alpha
> binary, and it wraps `codex exec`.

Fallbacks, in order: talk to raw `codex app-server` JSON-RPC ourselves if the
SDK hides something we need, then `codex exec --json` for debugging only.

Mapping to Conduit's modules:

| Conduit (Claude)          | Conduix (Codex)                                    |
| ------------------------- | -------------------------------------------------- |
| `claude-agent-sdk`        | `openai-codex` (official Python SDK)               |
| `ClaudeSDKClient` per session | one `AsyncThread` per session on a shared `AsyncCodex`; ephemeral thread for stateless requests |
| `streaming.py` *forwards* Anthropic SSE | `streaming.py` *synthesizes* OpenAI SSE from Codex events. This is the main engineering cost. |
| `tool_bridge.py` (SDK MCP bridge) | Custom function tools come through a Conduix-hosted MCP server, or through Codex dynamic tools if the protocol supports them (see §7) |
| `_QUOTA_MARKERS` → 429    | Detect Codex/ChatGPT usage-limit events → OpenAI-shaped 429 `rate_limit_exceeded` |
| `effort` extension        | Native `reasoning.effort`, passed through as-is. The allowed values depend on the model: `low/medium/high/xhigh/max`, plus `ultra` on some (§3.1) |
| `include_thinking`        | Native: reasoning **summaries** come back as `reasoning` items with the Responses API, and are dropped for Chat Completions |

### Codex must be told it is a chat model

Every thread starts with the following settings pinned. They are passed as
`thread_start` arguments or `config={...}` and are never read from the user's
`~/.codex/config.toml`:

- `approval_mode=ApprovalMode.deny_all`, so nothing ever waits on a human.
  The only other value is `auto_review`, which is the SDK default.
- `sandbox=Sandbox.read_only` and `cwd=` a **dedicated empty workspace dir**
  (e.g. `%LOCALAPPDATA%\conduix\workspace`), never the server's cwd.
- Built-in tools **off** through `config`. The probe confirmed that this set
  is accepted by 0.157.1:
  `web_search="disabled"`, and `features.<x>=False` for `shell_tool`,
  `unified_exec`, `view_image`, `apps`, `plugins`, `multi_agent`,
  `browser_use`, `computer_use`, `image_generation`, `sleep_tool`,
  `tool_suggest`, `skill_search`, `goals`, `hooks`. The canonical copy lives in
  `scripts/probe_sdk.py` (`CHAT_ONLY_CONFIG`). Web search gets turned back on
  per request in v1.1, and image generation in v1.2.
- `ephemeral=True` for stateless requests, so the throwaway threads don't pile
  up in Codex's thread store.
- `base_instructions=` a minimal assistant prompt, so there is no coding-agent
  persona. `instructions` / the `system` message from the request goes in as
  `developer_instructions`.
- The model comes from the request, then `CONDUIX_DEFAULT_MODEL`, then Codex's
  own default (`gpt-6-astra`, with `is_default: true` in the model list).

### 3.1 Probe findings (step 2, 2026-09-26, SDK 0.157.1)

Source: `scripts/probe_sdk.py`. The full log is in `scratch/probe_*.jsonl`,
which is gitignored. There were 3 turns in one thread: hello, tool bait, and
memory.

**Account / billing.** `codex.account()` returned `type: "chatgpt"` and
`plan_type: "plus"`, with `requires_openai_auth: true`. That is subscription
auth, not API-key auth. The shell also had `OPENAI_API_KEY` set, which the
probe's guard caught. The server must **remove it from the Codex process's
environment** and not rely on the user unsetting it. (`CodexConfig.env` is
merged *over* `os.environ`, so it can't remove a key: the server deletes the
keys from its own environment before spawning Codex. See `scrub_api_keys()`.)

**Models for this plan** (`codex.models()`):

| id | default | default effort | efforts | input |
| -- | -- | -- | -- | -- |
| `gpt-6-astra`   | ✅ | low    | low…max, ultra | text, image |
| `gpt-6-sol`     |    | medium | low…max, ultra | text, image |
| `gpt-6-luna`    |    | medium | low…max        | text, image |
| `gpt-5.6-sol`   |    | low    | low…max, ultra | text, image |
| `gpt-5.6-terra` |    | medium | low…max, ultra | text, image |
| `gpt-5.6-luna`  |    | medium | low…max        | text, image |
| `gpt-5.5`       |    | medium | low…xhigh      | text, image |

`/v1/models` can be served straight from this list. Validate `effort` against
the chosen model's list and return a 400 on a mismatch.

**Notification sequence per turn** (`method`: payload highlights):

1. `turn/started`: `turn.id`, `status: "inProgress"`
2. `item/started` + `item/completed` for `type: "userMessage"` (the echoed
   input, which Conduix ignores)
3. *(optional)* `item/started` + `item/completed` for `type: "reasoning"`:
   `summary: []`, `content: []`
4. `item/started` for `type: "agentMessage"` with `phase: "final_answer"`
5. `item/agentMessage/delta` × N: `{delta, item_id, turn_id}`. These are
   real token-level deltas.
6. `item/completed` for `agentMessage`: the full `text`
7. `thread/tokenUsage/updated`: `token_usage.last` and `.total`, each with
   `input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`,
   `output_tokens`, `reasoning_output_tokens`, `total_tokens`, plus
   `model_context_window`
8. `turn/completed`: `status`, `duration_ms`, `items`

Mapping to the Responses API: step 4 → `response.output_item.added`,
step 5 → `response.output_text.delta`, step 6 → `response.output_text.done` +
`response.output_item.done`, and step 8 → `response.completed`. The `usage`
comes from `token_usage.last`: `input_tokens`,
`input_tokens_details.cached_tokens` ← `cached_input_tokens`,
`output_tokens`, and `output_tokens_details.reasoning_tokens` ←
`reasoning_output_tokens`. Use the `item.id` values (`msg_…`, `rs_…`) as
the output item ids.

**Chat-only lockdown.** The config above was accepted with no errors. The tool
bait ("run `dir`…") got back `NO_TOOLS`, and the workspace dir stayed empty.
That is behavioural evidence, not proof. The step 11 tests should assert that
no `commandExecution`, `fileChange`, or `mcpToolCall` items ever appear.
*Done (step 11):* such items are counted in `Backend.blocked_items` (shown
as `blocked_agent_items` on `/health`) and dropped; the live tool-bait tests
assert the count stays zero and the workspace stays empty, and the offline
fake-app-server tests assert the exact lockdown parameters reach
`thread/start` (`approvalPolicy: never`, `sandbox: read-only`, the full
`CHAT_ONLY_CONFIG`).

**Continuity.** A second turn in the same thread correctly quoted the first.
Thread state is enough for sessions.

**Latency** at `effort="low"`: about 3.2–3.8 s to the first delta and about
3.5 s per turn in total.

**Hidden prompt overhead.** Even "hi" costs about **4.4k input tokens** despite
the custom `base_instructions`. From the second turn on, about 4.2k of that is
cached. Report it honestly in `usage` and investigate in §7.

**Reasoning summaries arrive only when the model actually reasons.** With
`summary="auto"` at `effort="low"` the `reasoning` item came back with
`summary: []`, and an easy question at medium/high used 0 reasoning tokens
and produced no reasoning item at all. A hard problem at `xhigh` with
`summary="detailed"` (2026-09-27) produced 66 reasoning items, 108
`item/reasoning/summaryTextDelta` events and ~34k reasoning tokens in one
turn: summaries work, but many small reasoning items per turn is normal.

**History injection.** `thread/inject_items` (raw request; the SDK has no
wrapper) appends raw Responses API items (`message` with user / assistant /
developer roles) to a thread's model-visible history. Checked live: an
injected assistant turn was recalled. Stateless replay and
`previous_response_id` rebuilds use it instead of flattening history to text.

### 3.2 Image generation findings (step 13, 2026-10-03, SDK 0.157.1)

Source: `scripts/probe_image_gen.py` (log in `scratch/probe_image_*.jsonl`),
then live runs through the server.

- **It works on the Plus plan.** `modelProvider/capabilities/read` reports
  `image_generation: true`. A thread started with
  `features.image_generation=True` (everything else still locked down)
  generates images; the read-only sandbox doesn't get in the way.
- **One item, no partial images.** `item/started` for `type:
  "imageGeneration"` (`status: "in_progress"`, `result: ""`), then
  `item/completed` about 20 s later with `result` (the whole image as base64
  PNG, about 0.7 MB for a simple one), `revised_prompt`,
  `transparent_background` and `saved_path`. After it the model makes a
  second call and closes with an **empty** `agentMessage`.
- **Codex writes its own copy to disk**, outside the workspace:
  `~/.codex/generated_images/<thread>/<item>.png`. Conduix deletes it as soon
  as the item completes. Checked live: the thread carries on fine without it.
- **The image is fed back to the model** (the second model call has about
  2.2k more input tokens), so a follow-up on the same thread sees it.
- **Replaying an image on a new thread:** `thread/inject_items` accepts an
  `image_generation_call` item, but the model then **doesn't see the
  image** (found live, with the feature on or off). A user message with a
  short note and the image as `input_image` does work, so that is how a
  generated image is replayed.
- **Edits work.** Given an earlier image and "make the square red", the tool
  ran as an edit of the provided image (its `revised_prompt` said so), and
  the result kept the layout.
- **Size can be asked for.** A size in the prompt was honoured
  (`1536x1024` came back as exactly that); with no size the image was
  1254x1254. The model writes the tool call itself, so size, quality and
  background can't be enforced.
- **Failure shape (not seen live):** the item carries `failure: {type:
  "usageLimitExceeded", limitId, resetsAt}`. That is the SDK's type, and the
  Codex binary serializes the same thing (`ImageGenerationFailure::
  UsageLimitExceeded`, an internally tagged enum with `limitId` and
  `resetsAt`).
- **Which limit it is.** `account/rateLimits/read` reports a single bucket
  for this account, `codex` (the weekly window on `/health`), and no image
  bucket. Its `used_percent` went from 2 to 4 over this step's 8 images and
  about 16 text turns, so images draw on the plan's ordinary usage limit.
  Reaching it on purpose would use up the week's quota, so the
  failure has only been tested offline (`tests/test_image_generation.py`,
  through the real SDK against the fake app-server).

### 3.3 Voice output findings (step 14 spike, 2026-11-16, Codex CLI 0.163.0)

Source: `scripts/probe_realtime_voice.py` (websocket transport) and
`scripts/probe_realtime_webrtc.py` (WebRTC, with `aiortc` as the peer); logs
and WAVs in `scratch/probe_voice_*` and `scratch/probe_webrtc_*`.

- **Codex has an experimental thread realtime API.** It isn't in the SDK's
  generated types; `codex app-server generate-json-schema --experimental`
  lists it: `thread/realtime/start | stop | appendText | appendSpeech |
  appendAudio | listVoices`, and notifications `thread/realtime/started | sdp
  | outputAudio/delta | transcript/delta | transcript/done | error | closed`.
  The SDK already initializes with `experimentalApi: true`, so raw
  `client.request(...)` calls reach it. Its notifications carry `threadId`
  but no turn, so the SDK would put them on its global queue; Conduix's
  reader loop hands them to the speech session instead.
- **Websocket transport is refused on a ChatGPT login:** `thread/realtime/error`
  "realtime conversation requires API key auth" (bundled 0.157.1 and 0.163.0).
- **WebRTC transport passes the auth gate** (it goes through ChatGPT's
  `backend-api/.../realtime/calls`). Conduix must then be the WebRTC peer:
  send a real SDP offer in `transport: {type: "webrtc", sdp}`, apply the answer
  from `thread/realtime/sdp`. A stub offer gets "Invalid SDP offer".
- **Only realtime protocol `version: "v3"` works there.** v1 (the default) →
  "AVAS requires OpenAI-Alpha: quicksilver=v2"; v2 → "AVAS realtime calls
  require realtime v1 or v3". WebRTC + v3 works on the SDK's bundled
  0.157.1 as well as 0.163.0, so no other Codex binary is needed.
- **Audio comes on the WebRTC media track** (Opus; recorded as 48 kHz
  stereo), not as `outputAudio/delta` notifications. The `oai-events` data
  channel carries `session.started`, `turn.created | delta | done`,
  `output_transcript.added` and `session.usage.updated` (`audio_duration_ms`).
  The thread also gets `thread/realtime/transcript/*`.
- **It is a conversational voice model, not TTS.** Text sent before the data
  channel's `session.started` was dropped, and the voice greeted on its own
  ("Hi there."). After `session.started`, `appendSpeech` text was taken as
  context ("Mhm."), while `appendText` "Say exactly this, and nothing else:
  <text>" was spoken verbatim (transcript matched exactly).
- **Timing:** about 6 s from `thread/realtime/start` to `session.started`;
  the 9-word phrase took about 5 s to speak.
- **Voices** (`thread/realtime/listVoices`): v3 lists `juniper, maple,
  spruce, ember, vale, breeze, arbor, sol, cove, aube, bubbie, haetsal,
  hanul, himari, leher, miko, neer, porto, rindo, rio, rivage, selva, sonna,
  tinta, viola`. OpenAI's TTS names (`alloy`, `marin`, ...) are v2-only.
  The `voice` start parameter is honoured and validated: `marin` gets
  "realtime voice `marin` is not supported for v3; supported voices: ...",
  `juniper` works.
- **Built (step 14), live 2026-11-16 through the `openai` SDK:** a 9-word
  phrase as WAV (3.1 s of audio, 13 s per request), as MP3 with `juniper`,
  and a 330-character input as Opus in two chunks (20.7 s of audio, 35 s).
  All transcripts matched on the first try.

### 3.4 Transcription findings (step 15 spike, 2026-11-16, Codex 0.157.1 bundled)

Source: `scripts/probe_realtime_transcribe.py` (logs in
`scratch/probe_transcribe_*.jsonl`). The clips were step 14's own output:
a 4.5 s MP3 (`juniper`) and the 20.8 s two-chunk Opus (`cove`).

- **The realtime session transcribes what it hears.** Audio played into the
  WebRTC microphone track comes back as `thread/realtime/transcript/done` with
  `role: "user"`, one per user turn. The data channel also streams it as it
  goes: `input_transcript.added` per word or two, with `start_ms` / `end_ms`
  on the session's timeline in 200 ms steps, and `turn.created` / `turn.done`
  with `role: "user"`.
- **Accuracy:** the 4.5 s clip was exact every time ("Hi, this is Juniper
  speaking through Conduix on your ChatGPT plan"; only the final period
  dropped). Over four runs of the 20.8 s clip, everything was exact except
  the coined name "Conduix" ("Can Dewix", "Candux", "Can do it"),
  "plan." → "planned" twice, and "WebRTC" → "Web RTC" once. The whole clip
  was one user turn each time (its 0.25 s gap didn't split it).
- **Real time.** The audio has to be played at its own speed: ~7 s setup,
  then the audio's length, then ~2–3 s to the final transcript (4.5 s →
  ~14 s, 20.8 s → ~31 s).
- **Audio sent before the data channel's `session.started` is lost** (one run
  lost its first sentence). Send silence until then.
- **Output must be audio.** `outputModality: "text"` → "text realtime output
  modality requires realtime v2", and v2 is refused (§3.3). The voice replies
  out loud once the speaker stops ("Hey Juniper, I'm here and ready…"); the
  replies are simply not recorded.
- **The voice can delegate to Codex, which runs a real turn.** On the longer
  clip the v3 model raised `delegation.created` (data channel) /
  `thread/realtime/itemAdded` `handoff_request`, and Codex started a turn
  on the thread *before* the user transcript arrived: a `userMessage`
  wrapping `<realtime_delegation>`, one model call (~4.75k input tokens), an
  empty answer. It is the voice model's choice: the short clip never
  delegated. A "never delegate" prompt didn't stop it, and neither did
  `clientManagedHandoffs: true`.
- **Interrupting it works.** `turn/interrupt` sent as soon as `turn/started`
  appears on the thread ended the turn in 108 ms with status `interrupted`,
  before any model call (no `thread/tokenUsage/updated`, no items).
- Not seen: a language field, confidence scores, or punctuation control.

---

## 4. Feature map (v1 target)

| Feature                                   | v1 | Notes |
| ----------------------------------------- | -- | ----- |
| Text chat, streaming + non-streaming      | ✅ | SSE synthesized: `response.created`, `response.output_text.delta`, `response.completed`, etc.; for chat completions, `chat.completion.chunk` + `[DONE]` |
| Stateless history replay                  | ✅ | Same as Conduit Pattern A: replay `input`/`messages` into a fresh thread |
| Stateful sessions (`session_id`, `previous_response_id`) | ✅ | One Codex thread per session, with a per-session `asyncio.Lock` |
| Image input (`input_image`, `image_url`)  | ✅ | Data URLs go through as `ImageInput`. Remote URLs get a 400 (deprecated upstream). All listed models take image input. |
| `reasoning.effort`                        | ✅ | Passed through and validated against the model's supported efforts (§3.1) |
| Reasoning summaries                       | ✅ | As `reasoning` output items (Responses API). Only when the model reasons; empty at low effort (§3.1) |
| Usage (`input_tokens`, `output_tokens`, `reasoning_tokens`, cached) | ✅ | Taken from `thread/tokenUsage/updated` → `token_usage.last`. Every OpenAI usage field has a direct source (§3.1). |
| Custom function tools (client-executed)   | ✅ v1.1 | Codex dynamic tools; on a call the response ends with `function_call` and the turn is interrupted; `function_call_output` resumes on a rebuilt thread (§7, verified live 2026-09-27) |
| Hosted `web_search`                       | ✅ v1.1 | `tools: [{"type": "web_search"}]` / chat `web_search_options` turn Codex's web search on for that thread (`CONDUIX_WEB_SEARCH_MODE`, default `live`); `web_search_call` output items |
| Image generation (`image_generation` tool, `/v1/images/generations`) | ✅ v1.2 | `tools: [{"type": "image_generation"}]` turns Codex's image tool on for that thread; `image_generation_call` output items with the PNG in `result`. Earlier images are replayed as a user image message (§3.2). `/v1/images/generations` runs the same turn and stops it once the image is in. Not on Chat Completions (no image output shape) or with `session_id`. Tool options are ignored; the images endpoint passes `size` / `background` as a hint |
| Transcription (`/v1/audio/transcriptions`) | ✅ v1.4 | Experimental, `CONDUIX_TRANSCRIPTION=1` and `conduix[voice]`. The upload is played into a Codex realtime v3 session over WebRTC; the session's user transcript is the result (§3.4). `json`, `text`, `verbose_json` (segments, word timings), `srt`, `vtt`; no streaming. Delegated Codex turns are interrupted |
| Speech output (`/v1/audio/speech`)        | ✅ v1.3 | Experimental, `CONDUIX_SPEECH=1` and `conduix[voice]`. A Codex realtime v3 session over WebRTC reads the input chunk by chunk; each transcript is checked against its chunk (§3.3). All six `response_format`s; no streaming. OpenAI voice names map to the default voice, `speed` and `instructions` are hints |
| `temperature`, `top_p`, `stop`, `max_output_tokens` | ⚠️ | Accepted. Applied where Codex supports them, otherwise ignored, and the docs list which |
| Structured outputs (`text.format` json_schema) | ✅ | Passed as the turn's `output_schema`; verified live 2026-09-27. Upstream only accepts strict schemas, and a schema allowing only `{}` never finishes (found live), so it gets a 400 |
| JSON mode (`json_object`)                 | ✅ | Can't be a schema (no strict "any object"), so it's an instruction on that turn only, plus code-fence stripping done while streaming (`json_safe_prefix`: only an undecided opening fence and a trailing run of backticks/whitespace are held back). Verified live on both endpoints; not guaranteed like json_schema |

---

## 5. Stack & layout

Same stack as Conduit so the two projects stay easy to switch between:
Python 3.11, FastAPI, uvicorn, `sse-starlette`, pydantic-settings, and
**`openai-codex`** as the backend. The official `openai` package is the source
of truth for wire types (the equivalent of Conduit re-exporting
`anthropic.types`). Conda env is `conduix`.

```
conduix/
  app.py            FastAPI app, lifespan opens/closes the shared AsyncCodex
  config.py         CONDUIX_* settings
  schema.py         re-export openai types + session_id extension
  backend.py        owns the AsyncCodex instance, chat-only thread defaults, Codex-to-internal event mapping
  sessions.py       SessionManager → Codex threads, idle eviction, locks
  streaming.py      Codex events → OpenAI SSE (responses + chat.completion.chunk)
  errors.py         quota / auth / upstream → OpenAI error envelope
  routes/responses.py
  routes/chat.py
  routes/images.py
  routes/audio.py   POST /v1/audio/speech, /v1/audio/transcriptions
  voice.py          the aiortc WebRTC peer; speech (chunking, transcript check, encoding)
                    and transcription (decoding, listening, timings, output formats)
  routes/models.py
  routes/sessions.py
tests/              unit (offline, fake backend) + integration (-m integration)
scripts/            probe_sdk.py (step 2), probe_image_gen.py (step 13), probe_realtime_voice.py + probe_realtime_webrtc.py (step 14 spike), probe_realtime_transcribe.py (step 15 spike) + demos: inference, multiturn, image, effort
scratch/            local probe logs (gitignored)
plan/               step-by-step build plan, same style as Conduit's plan/
start_app.ps1       kill (whole process tree) and restart on port
```

**Default port `8766`**, so Conduit on `8765` and Conduix can run side by side.

### Config (`CONDUIX_*`, env or `.env`)

| Variable                         | Default         |
| -------------------------------- | --------------- |
| `CONDUIX_HOST` / `CONDUIX_PORT`  | `127.0.0.1` / `8766` |
| `CONDUIX_DEFAULT_MODEL`          | *(Codex default)* |
| `CONDUIX_DEFAULT_EFFORT`         | *(none)*        |
| `CONDUIX_DEFAULT_INSTRUCTIONS`   | *(none)*        |
| `CONDUIX_CODEX_BIN`              | *(unset → SDK's bundled binary; set to override via `CodexConfig.codex_bin`)* |
| `CONDUIX_WORKSPACE_DIR`          | `%LOCALAPPDATA%\conduix\workspace` |
| `CONDUIX_SESSION_IDLE_TIMEOUT_S` | `1800`          |
| `CONDUIX_MAX_SESSIONS`           | `100`           |
| `CONDUIX_WEB_SEARCH_MODE`        | `live` (or `cached`), used when a request enables web search |
| `CONDUIX_SPEECH`                 | `false`; `1` turns on `/v1/audio/speech` (experimental) |
| `CONDUIX_SPEECH_VOICE`           | `cove`, the realtime v3 voice for OpenAI voice names |
| `CONDUIX_SPEECH_MIN_SIMILARITY`  | `0.8`, how closely a transcript must match its chunk |
| `CONDUIX_TRANSCRIPTION`          | `false`; `1` turns on `/v1/audio/transcriptions` (experimental) |
| `CONDUIX_TRANSCRIPTION_MAX_SECONDS` | `600`, the longest upload (it is played in real time) |

**Billing guard:** at startup, call `codex.account()` and refuse to serve
unless `account.type == "chatgpt"`. Log `plan_type` and show it on
`/health`. Always start Codex with an environment that has `OPENAI_API_KEY`
(and `CODEX_API_KEY`) **removed** from the server's own `os.environ`, which
the Codex child inherits (`CodexConfig.env` can only add or override). The key really is set
in the user's shell, so the server strips it instead of failing. This mirrors
Conduit's "leave `ANTHROPIC_API_KEY` unset" rule, but here it is enforced in
code instead of only documented.

---

## 6. Build sequence

Each step is independently verifiable. Do not start step N+1 until step N
passes.

1. ✅ **Bootstrap** (done 2026-09-27): pyproject, conda env, `/health`.
2. ✅ **Probe the SDK** (done 2026-09-26): `scripts/probe_sdk.py`, with results
   in §3.1. Still to do: probe reasoning summaries at higher effort, the error
   and quota notification shapes, and one image turn.
3. ✅ **`backend.py`** (done 2026-09-27; `config.py` pulled forward from step 4): a single `AsyncCodex` started and closed in the FastAPI
   lifespan, the chat-only thread defaults, and Codex notifications mapped
   to a small internal event type.
4. ✅ **Schema + config** (done 2026-09-27, folded into other steps): `config.py` came with
   step 3, `schema.py` (request parsing over the `openai` types) with step 7.
5. ✅ **Sessions** (done 2026-09-27; `/v1/sessions` routes and a minimal `errors.py` included): thread per session, locks, eviction.
6. ✅ **Streaming, Responses API** (done 2026-09-27): the SSE synthesizer and its non-streaming
   collector.
7. ✅ **`/v1/responses` route** (done 2026-09-27): stateless replay, `session_id`, and
   `previous_response_id`.
8. ✅ **Errors** (done 2026-09-27; `/v1/models` added too): quota/auth/upstream mapping. Streaming failures are sent as SSE
   `error` events, never as a dropped connection (Conduit commit `a7ce494`).
9. ✅ **`/v1/chat/completions`** (done 2026-09-27) as a translation layer over 6–7. v1 acceptance (§6) passes live.
10. ✅ **Images** (done 2026-09-27; data URL passthrough). Verified live: image as turn input on both
    endpoints, and re-injected via `thread/inject_items` (`input_image` items) in resent history
    and in a `previous_response_id` rebuild. The response store is capped at 256 MB as well as
    1000 records, since stored history now carries base64 images.
11. ✅ **Tests** (done 2026-09-27): offline unit tests against a fake app-server; integration tests
    through the `openai` SDK (`client = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="x")`).
    `tests/fake_app_server.py` speaks the real JSON-RPC protocol (replaying responses recorded
    from a real app-server), so backend.py and the real SDK run offline. Found: `Backend.stop()`
    raised if the app-server had died (SDK close() on Windows); fixed.
12. ✅ **v1.1** (done 2026-09-27): custom function tools, web search. Verified live on both
    endpoints, incl. a two-round agent loop. Design in §7 ("Custom tools").
13. ✅ **v1.2: image generation** (done 2026-10-03; goal 6, findings in §3.2). The
    `image_generation` tool on `/v1/responses` (streaming included) and `/v1/images/generations`.
    Verified live: an image through each, a follow-up on the same thread and on a rebuilt one,
    and an edit of an earlier image named by its id.
14. ✅ **v1.3: speech output** (done 2026-11-16, findings in §3.3).
    `POST /v1/audio/speech` in OpenAI's shape (`model`, `input`, `voice`,
    `response_format`, `instructions`, `speed`), off unless `CONDUIX_SPEECH=1`.
    Built as planned below, except: no other Codex binary is needed (the
    bundled 0.157.1 works); OpenAI voice names map to `CONDUIX_SPEECH_VOICE`
    and an unknown name is a 400 listing the v3 voices; turned off is a 501
    `speech_disabled`. Tests: `tests/test_speech.py` (22, offline).
    Verified live: WAV, MP3 and a two-chunk Opus through
    `client.audio.speech.create`, and a text turn on the same server.
    - **Peer:** a new `conduix/voice.py` runs one `aiortc` peer per request
      (silent mic track, `oai-events` data channel), starts realtime on a
      fresh chat-only thread with `outputModality: "audio"`, `version: "v3"`,
      `includeStartupContext: false` and a read-aloud prompt, applies the
      `thread/realtime/sdp` answer, and records the remote track.
    - **Speaking:** after `session.started`, send `appendText` "Say exactly
      this, and nothing else: …"; drop anything spoken before that; stop at the
      matching `turn.done`, then `thread/realtime/stop` and unsubscribe the
      thread. Long input is split at sentence boundaries, one turn per chunk.
    - **Fidelity:** compare each turn's transcript with its chunk
      (normalized); retry once on a mismatch, then 502 `speech_mismatch`.
      Trim leading and trailing silence.
    - **Output:** `wav` and `pcm` (24 kHz mono PCM16, resampled) first; `mp3`,
      `opus`, `aac`, `flac` through PyAV, which `aiortc` already pulls in. No
      streaming in the first cut (`stream_format: "sse"` is a 400).
    - **Voices:** map OpenAI names to v3 voices (or pass a v3 name through);
      check whether the `voice` start parameter is honoured. `speed` and
      `instructions` go into the prompt as hints.
    - **Plumbing:** `aiortc` as an optional extra (`pip install
      conduix[voice]`); the endpoint answers 501 when it is missing.
      Realtime errors map to the error envelope
      (usage limit → 429 with reset time). One speech request at a time
      (a lock), so voice sessions don't pile up.
    - **Tests:** offline, extend `tests/fake_app_server.py` with the realtime
      methods and fake the peer; live, one short phrase per output format
      through `client.audio.speech.create`, and the transcript check.
15. ✅ **v1.4: transcription** (done 2026-11-16, findings in §3.4).
    Built as planned below, except:
    - **Timings are calibrated on the audio.** The data channel stamps words
      when the recognizer emits them, ~0.7 s after they are spoken (found
      live). Every timing is shifted so the first word starts where speech
      starts in the upload (energy threshold, 10 ms windows), capped at 2 s;
      the last segment ends where speech ends, not after the trailing
      silence.
    - **Finishing** waits for playback to end, `listen_quiet` (2 s) with no
      open user turn, and Codex's transcript for every data-channel turn;
      it gives up `listen_extra` (30 s) after the audio's length.
    - `python-multipart` joined the `voice` extra; the route parses the form
      itself, so the server starts without it (501
      `transcription_unavailable`). A non-multipart body is a 400 on `file`.
    - Failures reuse speech's codes (`speech_failed`, `speech_timeout`).
    - Tests: `tests/test_transcription.py` (25, offline).
    Verified live through `client.audio.transcriptions.create`: the step-14
    samples as `json`, `verbose_json` with words, and `srt`/`text`; a
    delegated turn on the 20.8 s clip was interrupted (`delegatedTurn` on
    `/health`). One early request timed out (504) and didn't reproduce in
    five more; the give-up path now logs its state.
    `POST /v1/audio/transcriptions` in OpenAI's shape: multipart `file`
    (required, ≤ 25 MB), `model`, `language`, `prompt`, `response_format`,
    `temperature`, `timestamp_granularities[]`. Off unless
    `CONDUIX_TRANSCRIPTION=1`; needs the same `conduix[voice]` extra.
    - **Input:** decode any format PyAV reads (mp3, mp4/m4a, wav, webm,
      ogg/opus, flac) to 48 kHz mono PCM16. Reject a file it can't decode
      (400 `invalid_audio`) or longer than `CONDUIX_TRANSCRIPTION_MAX_SECONDS`
      (default 600: every second costs a second of wall time).
    - **Peer:** extend `voice.py`'s `AiortcPeer` with an input track that
      sends silence until told to play, then the PCM paced in real time,
      then silence. Speech keeps its silent microphone.
    - **Session:** `open_realtime` with a listen-only prompt (`language` and
      `prompt` added as hints); after `session.started`, play the audio; collect
      every `role: "user"` transcript; finish once playback has ended plus
      ~2 s of silence and no user turn is still open (data channel
      `turn.created` without its `turn.done`), with a timeout. Voice replies
      are ignored. Then stop and unsubscribe, as speech does.
    - **Delegation guard (backend):** while a realtime session is open on a
      thread, the reader loop watches that thread for `turn/started` and
      sends `turn/interrupt` at once, counting it in `blocked_items`
      (`delegatedTurn`). Applies to speech sessions too.
    - **Output:** `json` `{"text"}` (user turns joined with a space,
      trimmed); `text`; `verbose_json` with `duration`, `segments` (one
      per user turn) and, with `timestamp_granularities[]=word`, `words`
      from `input_transcript.added` (times shifted by when playback started;
      200 ms resolution); `srt` and `vtt` from the segments. `language` in
      `verbose_json` is the request's value or `"unknown"`. No streaming in
      the first cut (`stream: true` is a 400); `input_transcript.added`
      could feed `transcript.text.delta` events later. `usage` as
      `{"type": "duration", "seconds"}`.
    - **Concurrency:** shares speech's lock: one voice session at a time.
    - **Errors:** as speech, with `transcription_disabled`; a session that
      hears nothing returns `{"text": ""}`, not an error.
    - **Tests:** offline, the fake app-server gets a test-only method the fake
      peer calls when it "plays" audio, which makes the server emit a user
      transcript (and, by keyword, a delegated `turn/started` to check the
      interrupt); live, the step-14 samples through
      `client.audio.transcriptions.create` in `json` and `verbose_json`.

### Acceptance for v1

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="not-used")

r = c.responses.create(model="gpt-6-astra", input="Say hi in three words.")
print(r.output_text)

with c.chat.completions.create(model="gpt-6-astra", stream=True,
        messages=[{"role": "user", "content": "Count to five."}]) as s:
    for ch in s: print(ch.choices[0].delta.content or "", end="")
```

Both calls work, streaming and non-streaming. A second turn with
`previous_response_id` remembers the first. Running the Codex shell/file
tools through a prompt has no effect on disk.

---

## 7. Open questions / risks

- **SDK churn.** The SDK is new and follows Codex's release cadence. Pin
  `openai-codex==X.Y.Z` in pyproject and upgrade on purpose. Keep all SDK
  usage behind `backend.py` so a breaking release touches one file.
- **Turning off agent behaviour.** *Mostly settled* (§3.1): `deny_all` plus
  the `CHAT_ONLY_CONFIG` set. Still open: whether any tool schemas are still
  sent to the model. That may explain the second question.
- **About 4.4k tokens of hidden input per turn.** This remains even with
  custom `base_instructions`. Find out what is being sent (leftover tool
  definitions, environment context, AGENTS.md discovery from `cwd`?) and
  whether a config key trims it. Most of it is cached, but it still counts
  toward plan limits.
- **Plan tier.** The probe saw `plan_type: "plus"`. Confirm that is the
  intended account, since Plus and Pro have very different Codex limits.
- **Custom tools.** *Settled (step 12).* The SDK's public API has no
  client-defined function-tool hook, but app-server has **dynamic tools**
  (found in the Codex binary, verified live): `thread/start` takes
  `dynamicTools: [{type: "function", name, description, inputSchema}]`
  (missing from the SDK's ThreadStartParams, so sent raw), and a call arrives
  as the server *request* `item/tool/call` `{threadId, turnId, callId, tool,
  arguments}`, answered with `{contentItems: [{type: "inputText", text}],
  success}`. No MCP server needed.
  Design: Conduix never answers the request. The response ends with a
  `function_call` item and the turn is interrupted (no extra model call);
  the client's `function_call_output` comes back via `previous_response_id`
  or resent history, and is replayed onto a fresh thread with
  `thread/inject_items` (function_call + function_call_output items are
  accepted) followed by an **empty-input turn**, which continues from the
  tool result (verified live). This avoids parking Codex turns across HTTP
  requests entirely, works the same for Responses and Chat Completions, and
  survives restarts. Cost: the model's hidden reasoning isn't carried across
  a tool call (as with OpenAI's stateless function calling without reasoning
  items), and one call surfaces per response (parallel calls arrive over
  successive round trips).
  Needed a replacement for the SDK's reader loop: it answers every server
  request immediately from its single reader thread, and its default handler
  **accepts** command/file-change approvals. Conduix's loop re-routes
  `item/tool/call` into the turn's notification stream, **declines** every
  approval (counted in `blocked_agent_items`), and answers unknown requests
  with a JSON-RPC error.
  Also found: `BASE_INSTRUCTIONS` ("You have no tools") made the model refuse
  declared functions; threads with tools or web search get their own wording.
  And a turn with several model calls sends several `thread/tokenUsage/updated`
  notifications; per-turn usage is now `total` at the end minus `total`
  before the turn (it under-reported before).
- **`max_output_tokens` / `temperature`.** They may not be forwarded to the
  backend at all. If so, document them as advisory.
- **Remote image URLs.** Codex has deprecated them upstream, so return a 400
  in v1 and accept data URLs only.
- **Image generation** (step 13). Still open:
  - A real usage-limit failure during image generation has not been
    captured (§3.2). The mapping is built from the protocol types: on
    `/v1/responses` the item comes back with `status: "failed"` and the turn
    carries on; `/v1/images/generations` returns a 429 with the reset time.
    Unconfirmed: the item's `status` and `result` when it fails, whether the
    turn really continues or fails as a whole (then the ordinary quota path
    answers with a 429), and whether `resetsAt` is always set.
  - The tool's options (`size`, `quality`, `background`, `model`, ...) can't
    be enforced, because the model writes the image request. They are
    ignored on `/v1/responses`; `/v1/images/generations` passes `size` and
    `background` to the model as a hint.
  - After an image the model's closing message is empty, so the response
    ends with a `message` item whose text is `""`. It is kept in the output
    and left out of the history (stored or resent).
- **Speech output** (step 14, §3.3). Still open:
  - Thread realtime is an experimental protocol with internal version gates
    (AVAS, `quicksilver`). A Codex update can break it, so the endpoint stays
    opt-in and fails with a clear 501/502, never silently.
  - The voice is a conversational model asked to read, so it may still
    paraphrase, add a greeting or refuse; the transcript check catches this,
    at the cost of a retry.
  - Usage: the session reports `audio_duration_ms` but no plan bucket of its
    own was seen. How fast speech draws on the weekly limit, and what a
    usage-limit failure looks like, is unknown. For now a realtime error is
    a 429 when `account/rateLimits/read` shows a full window (or the message
    says "usage limit"), otherwise a 502 `speech_failed`.
  - `speed` is only a hint in the prompt; the audio isn't time-stretched.
  - Each request costs about 6 s of setup before audio. Reusing one warm
    session across requests could cut that but mixes context between
    requests; decide after the first cut.
- **Transcription** (step 15, §3.4). Still open:
  - It is a byproduct of a conversational voice model: no confidence
    scores and no language detection. It mishears names and coined words, and
    may differ from run to run. `language` and `prompt` are untested hints.
  - Real-time cost: an hour of audio would hold the only voice slot for an
    hour. Hence the duration cap, and why chunking or parallel sessions
    would need their own spike.
  - Delegated turns are interrupted, not prevented. If a Codex update moves
    the model call earlier, they would cost quota again; `blocked_items`
    on `/health` shows how often it happens.
  - Long pauses may split the audio into several user turns, and the voice's
    own replies might overlap later speech; only tested with continuous
    clips so far.
  - The one live 504 (first request after a server start) is unexplained.
  - Word timings are only as good as the onset calibration: audio with
    loud non-speech before the first word (music, noise) will shift them.
- **Quota signal.** *Mostly settled.* Classification uses Codex's own
  `codexErrorInfo` enum from the app-server protocol schema
  (`usageLimitExceeded`, `rateLimitExceeded`, ...), never message text, and the
  reset time comes from `account/rateLimits/read` (`primary`/`secondary`
  windows with `resets_at`; live 2026-09-27: one weekly 10080-min window).
  Still open: a real usage-limit turn has not been captured, so whether
  Codex sets `codexErrorInfo` on `turn/completed`, on the `error`
  notification, or only as a JSON-RPC error is unconfirmed; all three paths
  are handled. That same RPC family can *consume* the account's free
  rate-limit-reset credits: never call it.
- **Upstream API errors** (found live 2026-09-27): when the OpenAI API behind
  Codex rejects a request (e.g. invalid json_schema), the turn error's
  message is the raw OpenAI error envelope as JSON text. `errors.py` unwraps
  it and passes status/code/param through.
- **ToS.** Codex is built for programmatic use of the subscription, but check
  the terms before using it heavily or unattended.
