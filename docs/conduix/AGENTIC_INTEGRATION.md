# AGENTIC_INTEGRATION

Everything an AI coding agent (or human dev) needs to integrate a client
against a running Conduix server. Self-contained: you should not need the
rest of the docs to build a working integration.

Conduix is the OpenAI-side twin of Conduit. If you have already integrated
Conduit, read §13 first: it lists what is different.

---

## 0. TL;DR card

| Fact | Value |
|---|---|
| Base URL (default) | `http://127.0.0.1:8766/v1` as the SDK `base_url`. Raw paths are `/v1/responses`, `/v1/chat/completions`, ... Health is `GET /health` (no `/v1`). |
| Auth | **None.** Bound to loopback, single-user. The official `openai` SDKs refuse an empty key, so send any non-empty placeholder such as `"not-used"`. Never a real key. |
| Wire format | OpenAI **Responses API** and **Chat Completions**: request fields, response objects and the SSE event order as on `api.openai.com`. Response bodies are validated against the `openai` package's own types in the test suite. |
| What answers | OpenAI models on the operator's **ChatGPT plan**, through the Codex app-server. No API key is ever billed: the server strips `OPENAI_API_KEY` / `CODEX_API_KEY` from its environment and refuses to start unless Codex is logged in with a ChatGPT account. |
| Streaming | SSE. Responses: `event:` + `data:` frames, every event carries `sequence_number`. Chat: `data:` frames ending in `data: [DONE]`. |
| Conduix extensions | `session_id` request field (both endpoints, echoed in the body), `POST` / `GET` / `DELETE /v1/sessions`, `x_document` content parts with `x_document_citation` annotations (`RAG_INTEGRATION.md` §2.0), extra fields on `/v1/models` (`is_default`, `default_effort`, `efforts`, `input_modalities`), extra detail on `/health`, the `x-conduix-voice` header on speech. Nothing else is non-standard. |
| Drop-in OK | Yes. `OpenAI(base_url=..., api_key="not-used")` works for Responses, Chat, streaming, function tools, structured output, images, `previous_response_id` and (when turned on) `audio.speech`. Only `session_id` and `x_document` need `extra_body` / raw dicts. |
| Multi-turn | Three ways: resend the history (stateless), `previous_response_id` (server-side, branching works), `session_id`. See §4. |
| Function tools | Standard OpenAI: the response ends with a `function_call` item, you answer with a `function_call_output` item (via `previous_response_id` or resent history). **One function call per response.** See §5. |
| Hosted tools | `web_search` (also `web_search_preview`; chat `web_search_options`) and `image_generation`. Nothing else: file search, code interpreter, MCP and computer use are a 400. |
| Reasoning | `reasoning.effort` **per request** (`reasoning_effort` on chat). Valid values differ per model; read them from `/v1/models`. A value the model doesn't support is a 400. `reasoning.summary` returns `reasoning` items (Responses only). |
| Structured output | `text.format` `json_schema` (strict schemas) and `json_object` (instructed, best effort). `client.responses.parse(text_format=Model)` works. See §2.1, §8.6 and `RAG_INTEGRATION.md`. |
| Documents and citations | No `input_file` parts. Send passages as `x_document` parts for verified citations (`x_document_citation` annotations with character offsets), or as `input_text` with a citation convention. See `RAG_INTEGRATION.md`. |
| Speech | `POST /v1/audio/speech`, **experimental and off by default** (`CONDUIX_SPEECH=1` on the server). A realtime voice reads the input; each chunk's transcript is checked. ~13 s for a short phrase, one request at a time. See §2.6. |
| Validation errors | **400** with OpenAI's error envelope and a `param` naming the field. Not 422. |
| Latency | About 3–4 s to the first token at low effort; minutes are possible at high effort. Expect ~4.4k input tokens of fixed Codex overhead per call, mostly cached. |

---

## 1. Server contract

### 1.1 Where it runs

- Bound to `127.0.0.1` only (LAN-invisible) by default. `CONDUIX_HOST` / `CONDUIX_PORT` on the server side; default port `8766`.
- Single-user, single-host. No auth, no rate limiting.
- One Codex app-server child serves every request. Stateless requests run in parallel; a session runs one turn at a time (§4.7).
- The model has **no access** to files, shell or network beyond the web search you enable. Codex runs with approvals declined, a read-only sandbox in an empty workspace, and its agent tools off. `/health` counts anything it tried anyway under `blocked_agent_items`; it should stay empty.

### 1.2 Health check

```
GET /health   →   200
{
  "status": "ok",                 // "degraded" if the Codex process stops answering
  "version": "0.1.0",
  "codex": "ok",
  "codex_version": "0.157.1",
  "logged_in": true,
  "account_type": "chatgpt",
  "plan_type": "plus",
  "usage": {"primary": {"used_percent": 45, "window_duration_mins": 10080, "resets_at": 1793798739}},
  "blocked_agent_items": {}
}
```

`usage` is how much of the plan's usage window is used and when it resets
(Unix seconds). A 200 here means Codex answers; a real request can still fail
with a 429 once the window is exhausted (§7.1).

### 1.3 Auto OpenAPI / Swagger UI

`GET /docs` serves the interactive spec; `GET /openapi.json` the raw schema.

### 1.4 Restarts

`start_app.ps1` kills whatever holds the port (and its Codex child) before
starting. A restart loses **every session, every stored response**
(`previous_response_id`) and every generated image addressable by id. Check
`/health` before restarting; clients should treat a `404 not_found` on a
continuation as "resend the history" (§4.2).

---

## 2. Endpoint reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/responses` | OpenAI Responses API. **Primary.** |
| POST | `/v1/chat/completions` | OpenAI Chat Completions, translated onto the Responses path |
| POST | `/v1/images/generations` | OpenAI Images API (generate only) |
| POST | `/v1/audio/speech` | OpenAI text-to-speech. Experimental, off unless the server sets `CONDUIX_SPEECH=1` (§2.6) |
| GET | `/v1/models` | Models the plan can use, with their `efforts` |
| GET | `/v1/models/{id}` | One model |
| POST | `/v1/sessions` | Create a server-side conversation (Conduix extension) |
| GET | `/v1/sessions` | List explicit sessions |
| DELETE | `/v1/sessions/{id}` | End a session |
| GET | `/health` | Liveness, login, plan usage |
| GET | `/docs` | Swagger UI |

### 2.1 `POST /v1/responses`

**Request body (JSON):**

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | string | ❌ | One of the plan's model ids (`GET /v1/models`). Omitted → `CONDUIX_DEFAULT_MODEL`, else Codex's default (`gpt-6-astra` on the plan this was built on). Unknown → `400 model_not_found`. **Per request**, also on a session. |
| `input` | string \| item[] | ✅ | A string is one user message. Items: see §3.1. Must end with a user message or a `function_call_output`, else 400. |
| `instructions` | string | ❌ | System prompt. On a fresh thread it is the thread's developer instructions. Changing it between `previous_response_id` turns is honoured (the history is rebuilt on a new thread). On a `session_id` the thread's instructions are fixed; a different value is added to the conversation as a developer message. |
| `stream` | bool | ❌ default `false` | `true` → SSE (§3.4). |
| `previous_response_id` | string | ❌ | Continue a stored response (§4.2). `404 not_found` if evicted or the server restarted. Not together with `session_id` (400). |
| **`session_id`** | string | ❌ | **Conduix extension.** Run the turn on that session's thread (§4.3). Via `extra_body` in the SDKs. Not together with `tools` (400). |
| `reasoning` | `{"effort", "summary"}` | ❌ | `effort`: one of the model's `efforts` (`low`, `medium`, `high`, `xhigh`, `max`, `ultra` depending on the model); wrong one → `400 unsupported_value`. `summary`: `auto` \| `concise` \| `detailed` \| `none` → `reasoning` items in `output`. Other members ignored. |
| `text` | `{"format": {...}}` | ❌ | `{"type": "text"}` (default), `{"type": "json_schema", "name", "schema", "strict"}` or `{"type": "json_object"}`. `name` and `strict` are accepted and ignored: every schema is treated as strict. See §8.6 and `RAG_INTEGRATION.md`. |
| `store` | bool | ❌ default `true` | `false` keeps the response out of the store, so it can't be continued with `previous_response_id`. |
| `metadata` | `{str: str}` | ❌ | Echoed back. Non-string values → 400. |
| `tools` | tool[] | ❌ | `function`, `web_search` / `web_search_preview`, `image_generation`. Anything else → 400 with `param: tools[i].type`. See §3.2. |
| `tool_choice` | any | ❌ | `"none"` drops every tool. `"auto"`, `"required"` and a named function all behave as `"auto"`. |
| anything else | | | `temperature`, `top_p`, `max_output_tokens`, `truncation`, `user`, `parallel_tool_calls`, `include`, `service_tier`, ... are **accepted and ignored**. Codex has no setting for them. |

**Response, `stream=false`:** the standard `Response` object.

```json
{
  "id": "resp_6c1f...",
  "object": "response",
  "created_at": 1793450000,
  "completed_at": 1793450004,
  "status": "completed",
  "background": false,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "metadata": {},
  "model": "gpt-6-astra",
  "output": [
    {"id": "<item id>", "type": "message", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "Hi there, friend!", "annotations": [], "logprobs": []}]}
  ],
  "parallel_tool_calls": false,
  "previous_response_id": null,
  "reasoning": {"effort": null, "summary": null},
  "temperature": null,
  "text": {"format": {"type": "text"}},
  "tool_choice": "auto",
  "tools": [],
  "top_p": null,
  "usage": {
    "input_tokens": 4481,
    "input_tokens_details": {"cached_tokens": 4224, "cache_write_tokens": 0},
    "output_tokens": 47,
    "output_tokens_details": {"reasoning_tokens": 34},
    "total_tokens": 4528
  }
}
```

Things to know about this object:

- `reasoning.effort` echoes what you **sent** (`null` when you sent nothing), not the model's default that was applied.
- `tools` is always `[]`, `tool_choice` always `"auto"`, `parallel_tool_calls` always `false`, `temperature` / `top_p` always `null`. Don't read your configuration back from the response.
- `session_id` appears as an extra top-level field when the request used one.
- `output` may hold **more than one** `message` item (Codex sometimes emits a short commentary message before the answer). Use the SDK's `output_text` helper, or concatenate every `output_text` part yourself.
- A `message` item may carry a `phase` field (Codex's own, e.g. `"final_answer"`). Ignore it.
- `status` is `completed` for a normal turn **and** for a turn that ended on a function call. `incomplete` means the turn was interrupted upstream. `failed` is only seen inside a stream (§7.2); non-streaming failures are plain HTTP errors.

**Response, `stream=true`:** SSE, see §3.4.

### 2.2 `POST /v1/chat/completions`

A translation layer: the request becomes a Responses request and runs through
the same code, then the events are rendered as `chat.completion` /
`chat.completion.chunk`.

| Field | Notes |
|---|---|
| `model` | As above. |
| `messages` | Roles `system` / `developer` / `user` / `assistant` / `tool`. Content is a string or parts (`text`, `image_url` with a data URL). `system` becomes a developer message in the conversation. An assistant message's `tool_calls` and `tool` messages are the function-call history (§5.4). Legacy `function_call` / `role: "function"` → 400. |
| `stream`, `stream_options.include_usage` | SSE chunks; with `include_usage` a final chunk carries `usage`. |
| `reasoning_effort` | Same values and 400 as `reasoning.effort`. |
| `response_format` | `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`, `{"type": "json_object"}` or `{"type": "text"}`. |
| `tools`, `tool_choice` | Function tools only (`{"type": "function", "function": {...}}`); any other tool type → 400. |
| `web_search_options` | Any object (even `{}`) enables web search. Its members are ignored. |
| **`session_id`** | Conduix extension, as on Responses. |
| `n` | Only `1`; anything else → 400. |
| anything else | Accepted and ignored (`temperature`, `max_tokens`, `top_p`, `stop`, `seed`, `logprobs`, `functions`, ...). |

Chat never stores responses, so there is no `previous_response_id`; `messages`
carries the history, or use `session_id`. Chat has no `reasoning` items, no
image generation (400) and no `annotations` (always `[]`).

```json
{
  "id": "chatcmpl-9a3e...",
  "object": "chat.completion",
  "created": 1793450000,
  "model": "gpt-6-astra",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hi there, friend!", "refusal": null, "annotations": []},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {
    "prompt_tokens": 4481, "completion_tokens": 47, "total_tokens": 4528,
    "prompt_tokens_details": {"cached_tokens": 4224, "cache_write_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 34}
  },
  "system_fingerprint": null
}
```

Several message items are joined with a blank line into one `content`. When
the model called a function, `content` is `null`, `tool_calls` is set and
`finish_reason` is `"tool_calls"`.

### 2.3 `POST /v1/images/generations`

The Images API shape over Codex's built-in image tool: `{"prompt", "model"?,
"size"?, "background"?}` → `{"created", "data": [{"b64_json", "revised_prompt"?}],
"output_format": "png", "size"?, "background"?}`. One image per request
(`n` > 1 → 400), base64 only (`response_format: "url"` → 400), no streaming
(400), about 20–30 s. `model` can't pick an image model; it is accepted, and if
it names one of the plan's chat models that model drives the image tool. No
`usage`. When the model answers without producing an image: `502
no_image_generated` with what it said, `x-should-retry: false`. Plan limit
reached while generating: `429 usage_limit_exceeded`.

On the Responses API, `tools: [{"type": "image_generation"}]` produces an
`image_generation_call` item with the image in `result`; follow-up edits work
through `previous_response_id`. Full detail: usage guide §8.

### 2.4 `GET /v1/models` and `GET /v1/models/{id}`

```json
{"object": "list", "data": [
  {"id": "gpt-6-astra", "object": "model", "created": 1793400000, "owned_by": "openai",
   "is_default": true, "default_effort": "low",
   "efforts": ["low", "medium", "high", "xhigh", "max", "ultra"],
   "input_modalities": ["text", "image"]}
]}
```

The list is the plan's, read once at startup; it is the source of truth for
`model` and `reasoning.effort`. Build pickers from it. Unknown id → `404
not_found`.

### 2.5 Sessions (Conduix extension)

```
POST /v1/sessions      {"model"?: str, "effort"?: str, "instructions"?: str}   (or {} / no body)
  → 200 {"session_id": "sess_...", "object": "session", "model": null, "effort": null,
         "created_at": 1793450000, "last_used_at": 1793450000, "turn_count": 0, "busy": false}

GET  /v1/sessions      → 200 {"object": "list", "data": [SessionInfo, ...]}     (explicit sessions only)

DELETE /v1/sessions/{id}
  → 200 {"id": "sess_...", "object": "session.deleted", "deleted": true}
  → 404 {"error": {"message": "session 'sess_...' not found", "type": "invalid_request_error",
                   "param": "session_id", "code": "not_found"}}
```

`model` and `effort` are **defaults** for the session; a request's own values
win. `instructions` is fixed once the thread starts. Idle sessions expire after
`CONDUIX_SESSION_IDLE_TIMEOUT_S` (default 1800 s). Cap `CONDUIX_MAX_SESSIONS`
(default 100) shared with the implicit sessions of §4.7.

### 2.6 `POST /v1/audio/speech` (experimental)

OpenAI's text-to-speech shape. Codex has no TTS model: a realtime voice
session (WebRTC, protocol v3) is opened per request on the plan, the voice is
asked to read the text word for word, and the recording is returned. Off
unless the server runs with `CONDUIX_SPEECH=1` and has the `voice` extra
installed; otherwise every call is a `501`.

**Request body (JSON):**

| Field | Notes |
|---|---|
| `input` | Required. 1–4096 characters. Split at sentence ends into chunks of ~250 characters, one voice turn each. |
| `voice` | A realtime voice: `juniper`, `maple`, `spruce`, `ember`, `vale`, `breeze`, `arbor`, `sol`, `cove`, `aube`, `bubbie`, `haetsal`, `hanul`, `himari`, `leher`, `miko`, `neer`, `porto`, `rindo`, `rio`, `rivage`, `selva`, `sonna`, `tinta`, `viola`. OpenAI's names (`alloy`, `marin`, ...) and a missing voice use the server default (`cove`, `CONDUIX_SPEECH_VOICE`). Anything else → `400`, `param: "voice"`. `{"id": "..."}` is accepted too. |
| `response_format` | `mp3` (default), `opus` (Ogg), `aac` (ADTS), `flac`, `wav`, `pcm` (raw 16-bit little-endian). Always 24 kHz mono. |
| `instructions`, `speed` | Hints in the voice's prompt ("speak slowly"). `speed` must be 0.25–4.0; the audio isn't time-stretched. |
| `stream_format` | `"audio"` or absent. `"sse"` → `400`: the audio comes back whole. |
| `model` | Accepted and ignored. |

**Response:** `200`, the audio bytes with `Content-Type` `audio/mpeg`,
`audio/ogg`, `audio/aac`, `audio/flac`, `audio/wav` or `audio/pcm`, and
`x-conduix-voice` naming the voice that spoke.

**Verbatim check.** The voice is a conversational model, so each chunk's
transcript is compared with the chunk (normalized similarity ≥
`CONDUIX_SPEECH_MIN_SIMILARITY`, default 0.8). One retry, then `502
speech_mismatch` instead of audio that says something else.

**Timing and concurrency.** About 7 s of session setup, then real-time speech:
~13 s for a 9-word phrase, ~35 s for 330 characters. Requests are served
**one at a time**; others wait. Set the client timeout accordingly (the SDK's
default 600 s is fine).

**Errors** (all with `x-should-retry: false`): `501 speech_disabled`, `501
speech_unavailable` (voice extra missing), `502 speech_mismatch`, `502
speech_failed` (the realtime session failed; message passed through), `504
speech_timeout`, `429 usage_limit_exceeded` when the plan's usage window is
full. Validation errors are `400` with `param` as usual.

It relies on an experimental Codex protocol; a Codex update can break it, and
then it fails with one of the errors above. Feature-detect with one short call
rather than assuming it is there.

---

## 3. Wire format details

### 3.1 `input` items and content parts

`input` is a string (one user message) or a list of items:

| Item | Shape | Notes |
|---|---|---|
| Easy message | `{"role": "user" \| "assistant" \| "system" \| "developer", "content": str \| part[]}` | `type` defaults to `"message"`. `system` is folded into `developer`. |
| `message` | same, with `"type": "message"` | As copied from an earlier `output`. |
| `function_call` | `{"type": "function_call", "call_id", "name", "arguments"}` | History: a call the model made earlier. `arguments` is a JSON string (an object is re-serialised). |
| `function_call_output` | `{"type": "function_call_output", "call_id", "output": str \| part[]}` | Your result. `call_id` must match a `function_call` in this input or in the continued response's history, else 400. Parts may include `input_image`. |
| `image_generation_call` | `{"type": "image_generation_call", "result": <base64>}` or `{"type": "image_generation_call", "id": "<earlier id>"}` | Puts an earlier generated image back in front of the model. By `id` only while the earlier response is still stored (`400 image_not_found` otherwise). |
| `reasoning`, `web_search_call` | as copied from `output` | **Skipped.** They can't be replayed. Appending `r.output` wholesale is fine. |
| anything else | | `400 input item type '...' is not supported`. |

Content parts:

| `type` | Fields | Notes |
|---|---|---|
| `input_text`, `output_text`, `text` | `text` | Any of the three spellings. |
| `refusal` | `refusal` | Treated as text. |
| `input_image` | `image_url`: `data:image/png;base64,...` | PNG, JPEG, GIF, WebP; ≤ 20 MB. **User messages only.** `https://` URLs → `400 invalid_image_url`; `file_id` → 400. A model without image input → `400 unsupported_value`. |
| `image_url` | `{"url": "data:..."}` | The Chat Completions spelling; accepted on both endpoints. |
| `x_document` | `text`, optional `id`, `title`, `context`, `citations` | Conduix extension: a document the answer can cite with verified offsets. Newest user message only. `RAG_INTEGRATION.md` §2.0. |
| `input_file`, `input_audio`, ... | | `400 content part type '...' is not supported`. Audio input isn't supported (speech is output only, §2.6). Documents: `x_document` or text. |

The **last** item must be a user message (the new turn) or a
`function_call_output` (the model continues from your tool results). Several
trailing user messages form one turn. Everything before is injected into the
thread as real conversation items, not pasted into the prompt.

### 3.2 Tool declarations

```json
// Function tool (Responses spelling; chat wraps it in {"type": "function", "function": {...}})
{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
 "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}

// Hosted
{"type": "web_search"}            // or "web_search_preview"; extra members ignored
{"type": "image_generation"}      // options (size, quality, ...) ignored
```

Function names must match `[A-Za-z0-9_-]{1,64}` and be unique in the request
(400 otherwise). `parameters` defaults to an empty object schema.

### 3.3 Output items

| `type` | Fields | When |
|---|---|---|
| `message` | `id`, `role: "assistant"`, `status`, `content: [{"type": "output_text", "text", "annotations": [], "logprobs": []}]`, optional `phase` | Every text answer. Possibly more than one per response. |
| `reasoning` | `id`, `summary: [{"type": "summary_text", "text"}]` | Only with `reasoning.summary` set, and only when the model actually reasoned. Responses only. |
| `function_call` | `id` (`fc_...`), `status: "completed"`, `call_id`, `name`, `arguments` (JSON string) | The model called your function. At most one per response. |
| `web_search_call` | `id`, `status`, `action: {"type": "search", "query"} \| {"type": "open_page", "url"} \| {"type": "find_in_page", "url", "pattern"}` | With the `web_search` tool. |
| `image_generation_call` | `id`, `status`, `result` (base64 PNG), `revised_prompt`, `output_format`, `size`, `background` | With the `image_generation` tool. `status: "failed"` when the plan limit was hit mid-turn. |

### 3.4 SSE event sequence (Responses)

Every frame has an `event:` line equal to the JSON's `type`, and every event
carries a monotonically increasing `sequence_number`. The stream ends with
exactly one of `response.completed`, `response.incomplete` or
`response.failed`, then the connection closes.

**Plain text turn:**

```
event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_...","status":"in_progress",...}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{...}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,
       "item":{"id":"<item id>","type":"message","role":"assistant","status":"in_progress","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"item_id":"<item id>","output_index":0,
       "content_index":0,"part":{"type":"output_text","text":"","annotations":[],"logprobs":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"<item id>","output_index":0,
       "content_index":0,"delta":"Hi","logprobs":[]}

# ... more response.output_text.delta ...

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":9,...,"text":"Hi there, friend!","logprobs":[]}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":10,...,"part":{"type":"output_text","text":"Hi there, friend!",...}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":11,"output_index":0,"item":{...,"status":"completed",...}}

event: response.completed
data: {"type":"response.completed","sequence_number":12,"response":{...,"status":"completed","usage":{...}}}
```

**Reasoning summary** (with `reasoning.summary`), before the message:
`response.output_item.added` (a `reasoning` item), then per summary part
`response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta`
× N, `response.reasoning_summary_text.done`,
`response.reasoning_summary_part.done`, then `response.output_item.done`.

**Function call** (the turn ends here; `response.completed` still follows):

```
event: response.output_item.added
data: {...,"output_index":1,"item":{"id":"fc_...","type":"function_call","status":"in_progress",
       "call_id":"<call id>","name":"get_weather","arguments":""}}

event: response.function_call_arguments.delta
data: {...,"item_id":"fc_...","output_index":1,"delta":"{\"city\":\"Kyiv\"}"}

event: response.function_call_arguments.done
data: {...,"item_id":"fc_...","output_index":1,"arguments":"{\"city\":\"Kyiv\"}"}

event: response.output_item.done
data: {...,"output_index":1,"item":{...,"status":"completed","arguments":"{\"city\":\"Kyiv\"}"}}

event: response.completed
data: {...}
```

The arguments arrive in **one** delta (Codex hands over the complete call),
so there is nothing to reassemble; concatenating deltas still works.

**Web search:** `response.output_item.added` (`web_search_call`),
`response.web_search_call.in_progress`, `.searching`, `.completed`,
`response.output_item.done`, then the message events as usual.

**Image generation:** `response.output_item.added` (`image_generation_call`),
`response.image_generation_call.in_progress`, `.generating`, `.completed`,
`response.output_item.done` with the image in `item.result`. There are no
partial images.

**Failure while streaming:** an `error` event, then `response.failed` whose
`response.error` carries `{"code", "message"}`. The HTTP status is already
200 by then. See §7.2.

Output indices are 0-based and dense, per response. Don't carry them across
responses.

### 3.5 Chat Completions chunk sequence

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1793450000,"model":"gpt-6-astra",
       "system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":"","refusal":null},
       "finish_reason":null,"logprobs":null}]}

data: {...,"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null,"logprobs":null}]}

# ... more content chunks; a "\n\n" chunk separates two message items ...

data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]}

data: {...,"choices":[],"usage":{"prompt_tokens":4481,...}}        # only with stream_options.include_usage

data: [DONE]
```

A function call arrives as `delta.tool_calls` chunks: first
`{"index": 0, "id": "<call id>", "type": "function", "function": {"name": "get_weather", "arguments": ""}}`,
then `{"index": 0, "function": {"arguments": "{\"city\":\"Kyiv\"}"}}`, and the
last chunk has `finish_reason: "tool_calls"`. A failure is a
`data: {"error": {"message", "type", "param", "code"}}` chunk followed by
`[DONE]`; the `openai` SDK raises it. With `include_usage`, every chunk but the
last carries `"usage": null`, as upstream does.

### 3.6 Usage

| Responses | Chat | Meaning |
|---|---|---|
| `usage.input_tokens` | `prompt_tokens` | All model calls of the turn (a tool or search turn makes several). |
| `usage.input_tokens_details.cached_tokens` | `prompt_tokens_details.cached_tokens` | Prompt-cache hits. Codex's own ~4.4k-token prompt is most of it. |
| `usage.input_tokens_details.cache_write_tokens` | `prompt_tokens_details.cache_write_tokens` | Non-standard extra; may be `0`. |
| `usage.output_tokens` | `completion_tokens` | |
| `usage.output_tokens_details.reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | |
| `usage.total_tokens` | `total_tokens` | |

Counts are real but bill the plan's usage window, not money. `GET /health`
shows the window.

---

## 4. Modes of operation

### 4.1 Stateless: resend the history

Omit `previous_response_id` and `session_id`; send every earlier turn in
`input` (or chat `messages`). Each request gets a fresh thread; earlier items
are injected as real messages; the trailing user message(s) run as the turn.
Survives server restarts. You may append `r.output` directly to the history:
`reasoning` and `web_search_call` items are skipped, `function_call` and
`image_generation_call` items are replayed.

Cost: the whole history is sent and processed each time, and the prompt cache
is cold for your own content (Codex's fixed prompt still caches).

### 4.2 `previous_response_id`

Send only the new input and point at the previous response. Behaviour:

- Every completed response is stored unless `store: false`. The store keeps the
  last **1000** responses, at most **256 MB** (images make records large).
  Older ones return `404 not_found`: resend the history instead.
- **Branching works.** Two requests can both continue `r1`; each sees `r1`'s
  history and nothing after it.
- When the continued response is still the latest turn on its thread and
  `instructions` and tools are unchanged, the same thread continues (prompt
  cache warm). Otherwise the history is rebuilt from the store onto a fresh
  thread. Both are transparent; the second costs a cache miss.
- `instructions`, `model`, `reasoning.effort` and `tools` may change between
  turns.
- A response that ended on a `function_call` can only be continued with a
  `function_call_output` for that call (§5.2).
- Chat Completions never stores, so its responses can't be continued this way.

### 4.3 `session_id` (Conduix extension)

Create a session (`POST /v1/sessions`), then send only the new turn with
`session_id` on each request. Works on both endpoints. The session's thread
keeps the conversation, including images and documents from earlier turns.

- **One turn at a time per session.** A second request waits for the first.
- **No tools on a session** (`tools`, web search, image generation → 400).
  Use §4.1 or §4.2 for tool loops.
- `model` and `reasoning.effort` on the request override the session's
  defaults for that turn. A different `instructions` is added to the
  conversation as a developer message.
- The response body carries `"session_id": "sess_..."`.
- In memory only; a restart loses it (`404 not_found`).

Python: `client.responses.create(..., extra_body={"session_id": sid})`.
TypeScript: add `session_id` to the params object (cast past the type checker);
the SDK sends unknown fields through.

### 4.4 Model, effort and instructions are per request

Unlike Conduit, nothing is bound at session creation:

| | Where set | Scope |
|---|---|---|
| Model | `model` | This request. On a session the request's value wins over the session default. |
| Effort | `reasoning.effort` / `reasoning_effort` | This request. Omitted → the session default, else `CONDUIX_DEFAULT_EFFORT`, else the model's own default (`default_effort` in `/v1/models`). |
| Instructions | `instructions` / system message | A fresh thread's developer instructions; on an existing thread a change becomes a developer message (session) or rebuilds the thread (`previous_response_id`). |
| Structured output | `text.format` / `response_format` | This request. |
| Tools | `tools` | This request (a continuation with different tools rebuilds the thread). |

### 4.5 Reasoning summaries

`reasoning: {"effort": "high", "summary": "auto"}` (or `concise` / `detailed`)
adds `reasoning` items to `output`, streamed as in §3.4. They only appear when
the model actually reasons: at low effort or on an easy question there may be
none; a hard problem at high effort can produce dozens of short items. Chat
Completions never returns them. The raw chain of thought is never returned.

### 4.6 Prompt caching and latency

The Codex prompt overhead (~4.4k tokens) is cached across requests;
`cached_tokens` shows it. Your own history caches when the same thread
continues (§4.2 fast path, sessions) and is re-sent otherwise. First token at
low effort: ~3–4 s. High efforts can take minutes on hard tasks; stream, and
give the client a generous timeout (the `openai` SDK's default of 10 minutes
is fine). If the client disconnects mid-stream, Conduix interrupts the Codex
turn so it stops spending plan quota.

### 4.7 Concurrency and the session cap

Every stateless or `previous_response_id` request also runs on a session,
an **implicit** one that is hidden from `GET /v1/sessions` but counts toward
`CONDUIX_MAX_SESSIONS` (100) and is swept after 30 idle minutes. With
`store: false`, or after a function call, the implicit session is dropped at
once. Creating a session when all 100 are busy is a `503 session_limit`; an
idle one is evicted (least recently used) to make room. In practice: up to 100
turns in flight at once, and parallel requests should not share a session.

---

## 5. Function tools

### 5.1 Conceptual model

OpenAI's protocol is stateless: the model emits a `function_call` item, the
response completes, you run the function and send a `function_call_output`
item in a **new** request. Conduix keeps exactly that protocol. Underneath,
the Codex turn is interrupted when the call appears and your output is
replayed, with the history, onto a fresh thread, which is why the hidden
reasoning does not carry across the call (same as upstream's stateless
function calling) and why **at most one function call comes back per
response**: a model that wants several makes them over successive round
trips. `parallel_tool_calls` is always `false`.

### 5.2 Round trip with `previous_response_id`

**Step 1:**

```json
POST /v1/responses
{"model": "gpt-6-astra", "input": "Weather in Kyiv?",
 "tools": [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]}
```

**Step 2, the response** (`status: "completed"`, `output_text` usually empty):

```json
{"id": "resp_A", "status": "completed", "output": [
  {"id": "fc_...", "type": "function_call", "status": "completed",
   "call_id": "<call id>", "name": "get_weather", "arguments": "{\"city\":\"Kyiv\"}"}
], ...}
```

**Step 3:** run the function. **Step 4:** continue with the output:

```json
POST /v1/responses
{"model": "gpt-6-astra", "previous_response_id": "resp_A",
 "tools": [/* same list */],
 "input": [{"type": "function_call_output", "call_id": "<call id>",
            "output": "{\"city\": \"Kyiv\", \"temp_c\": 17, \"sky\": \"cloudy\"}"}]}
```

**Step 5:** the response holds the final `message` (or another
`function_call`: loop). `output` may be a string or a list of `input_text` /
`input_image` parts (a screenshot, say).

### 5.3 Round trip by resending the history

Same thing without the store: put the `function_call` item (copied from the
earlier `output`) and your `function_call_output` into `input`, after the
original user message. The `call_id` must match a `function_call` earlier in
the same `input` (or in the continued response's history), else
`400 no tool call found for function call output with call_id '...'`.

```json
{"model": "gpt-6-astra", "tools": [...], "input": [
  {"role": "user", "content": "Weather in Kyiv?"},
  {"type": "function_call", "call_id": "<call id>", "name": "get_weather", "arguments": "{\"city\":\"Kyiv\"}"},
  {"type": "function_call_output", "call_id": "<call id>", "output": "17C, cloudy"}
]}
```

### 5.4 Chat Completions

Standard: `finish_reason: "tool_calls"`, `message.tool_calls[]`, then append
that assistant message and one `{"role": "tool", "tool_call_id", "content"}`
message per call and send `messages` again. Recipe in §8.2.

### 5.5 Multi-cycle

Loop while the response contains a `function_call`. Each cycle is one HTTP
round trip; `previous_response_id` chains them. No cap on cycles. Send the
same `tools` every time.

### 5.6 Streaming a tool call

As in §3.4: `response.output_item.added` (`function_call`), one
`response.function_call_arguments.delta`, `.done`, `response.output_item.done`,
`response.completed`. Chat: `delta.tool_calls` chunks and `finish_reason:
"tool_calls"`. Continue exactly as in §5.2.

### 5.7 Hosted tools

**Web search:** `tools: [{"type": "web_search"}]` (Responses) or
`web_search_options: {}` (chat). One request, no pause. `output` gains
`web_search_call` items; the answer cites sources as **markdown links in the
text**, never as `url_citation` annotations. Combine freely with function
tools. Live by default; `CONDUIX_WEB_SEARCH_MODE=cached` on the server uses
Codex's search cache. Without the tool the model has no web access.

**Image generation:** `tools: [{"type": "image_generation"}]`. The model
decides when to draw; the image comes back as an `image_generation_call` item
(§3.3). Not on chat, not with `session_id`.

### 5.8 Rules and errors

| Situation | Result |
|---|---|
| `tools` + `session_id` | `400 tools can't be combined with session_id` |
| `function_call_output` with an unknown `call_id` | `400 no tool call found for function call output with call_id '...'` |
| `function_call_output` without `call_id` | `400 function_call_output needs a call_id` |
| Tool type other than `function` / `web_search*` / `image_generation` | `400`, `param: tools[i].type` |
| Bad or duplicate function name | `400`, `param: tools[i].name` |
| `tool_choice: "required"` or a named function | Accepted, behaves like `"auto"` |
| Legacy chat `function_call` / `role: "function"` | `400` |

---

## 6. Limitations

These are implementation limits, not wire incompatibilities. Clients need
awareness, not workarounds.

1. **One function call per response.** Several wanted calls arrive over
   successive round trips. `parallel_tool_calls` is always `false`.
2. **Hidden reasoning doesn't survive a function call** (fresh thread on
   continuation). Same as upstream stateless calling.
3. **No tools on sessions.** Use `previous_response_id` or resent history.
4. **`tool_choice` is advisory beyond `"none"`.**
5. **Sampling parameters are ignored** (`temperature`, `top_p`, `max_output_tokens`, `stop`, `seed`, ...). No error, no effect.
6. **`json_object` is instructed, not guaranteed.** `json_schema` is guaranteed but schemas must be strict (`additionalProperties: false` on every object); `name` and `strict` are ignored.
7. **No `input_file`.** Documents go in as `x_document` parts (verified `x_document_citation` annotations) or as text with a citation convention (`RAG_INTEGRATION.md`). Web search cites with markdown links, not `url_citation` annotations.
8. **Images: base64 data URLs only**, user messages only, one of the plan's image-capable models.
9. **Image generation:** one image per request, PNG base64, options are hints at best, not on chat or sessions.
10. **Other hosted tools** (file search, code interpreter, MCP, computer use), embeddings, transcription and audio input, image edits/variations, files, Assistants and batch: **not available**.
11. **State is in memory.** Sessions, stored responses and generated-image ids vanish on restart. Store: 1000 responses / 256 MB. Sessions: 100, 30 min idle.
12. **One Codex process.** Up to 100 concurrent turns; a session serialises its own turns.
13. **Model list is read at startup.** A plan change needs a restart.
14. **Speech is experimental** (§2.6): off by default, one request at a time, ~7 s setup per request, no streaming, realtime voices only (OpenAI voice names map to the default).

---

## 7. Errors

### 7.1 Envelope and codes

OpenAI's envelope, always:

```json
{"error": {"message": "effort 'max' is not supported by gpt-5.5 (supported: low, medium, high, xhigh)",
           "type": "invalid_request_error", "param": "reasoning.effort", "code": "unsupported_value"}}
```

| Status | `code` | Meaning | What to do |
|---|---|---|---|
| 400 | `null` | Request validation: missing / mistyped field, unsupported item or part type, bad role, `n` > 1, legacy function messages. `param` names the field. **This is where Conduit would say 422.** | Fix the request |
| 400 | `model_not_found` | `model` is not on the plan | Pick from `/v1/models` |
| 400 | `unsupported_value` | `reasoning.effort` not in the model's `efforts`; image sent to a text-only model | Pick from `/v1/models` |
| 400 | `invalid_json_schema` | Upstream rejected the schema (message passed through), or the schema only allows `{}` | Make the schema strict / give it properties |
| 400 | `invalid_image_url`, `unsupported_image_media_type`, `invalid_base64_image`, `image_too_large`, `empty_image_file` | Bad image | Send a valid base64 data URL |
| 400 | `image_not_found` | `image_generation_call` by `id` not in the store | Send the item with its `result` |
| 400 | `context_length_exceeded`, `session_budget_exceeded` | Conversation too long | Trim the history |
| 400 | `content_policy_violation`, `misalignment_policy_violation`, `invalid_request` | Upstream policy / bad request | Don't retry as is |
| 401 | `codex_unauthorized` | Codex login no longer valid | Operator runs `codex login` |
| 404 | `not_found` | `previous_response_id` / `session_id` / model id unknown (expired, evicted, restarted) | Resend the history or start over |
| 429 | `usage_limit_exceeded` | **Plan usage window exhausted.** Message carries the reset time; `retry-after` set; `x-should-retry: false` stops the `openai` SDK's retries | Wait for the reset; retrying won't help |
| 429 | `rate_limit_exceeded` | Transient upstream rate limit | Back off and retry (the SDK does) |
| 500 | `server_error` | Unclassified Codex failure | Retry once; check server logs |
| 502 | `upstream_error` | Codex couldn't reach the model / rejected the request | Retry once |
| 501 | `speech_disabled`, `speech_unavailable` | `/v1/audio/speech` is off on this server, or its `voice` extra isn't installed | Operator sets `CONDUIX_SPEECH=1` / installs `conduix[voice]`; otherwise don't offer speech |
| 502 | `speech_mismatch` | The voice didn't read a chunk verbatim (twice) | Retry later, or rephrase unusual text (numbers, symbols, very long sentences) |
| 502 | `speech_failed` | The realtime voice session failed (message passed through) | Retry once; persistent failures usually mean a Codex update broke it |
| 504 | `speech_timeout` | The voice session didn't start or finish in time | Retry once |
| 502 | `no_image_generated` | Images API: the model answered without an image (message says what it said); `x-should-retry: false` | Change the prompt |
| 503 | `server_overloaded` | Upstream overloaded | Retry with backoff |
| 503 | `codex_unavailable` | The Codex app-server process went away | Restart Conduix |
| 503 | `session_limit` | 100 sessions, all busy | Retry later; reduce concurrency |

Chat Completions reports the same errors with chat-side `param` names
(`messages`, `reasoning_effort`, `response_format.json_schema.schema`, ...).

### 7.2 Failures while streaming

The SSE head is sent before the turn runs, so a failure can't change the HTTP
status. Conduix never drops the connection silently:

- **Responses:** `event: error` with `{"code", "message", "param"}`, then
  `event: response.failed` whose `response.error` is `{"code", "message"}` and
  `response.status` is `"failed"`. A stream that ends this way did **not**
  complete, regardless of the 200.
- **Chat:** a `data: {"error": {...}}` chunk, then `data: [DONE]`. The `openai`
  SDK raises it as an exception.

A plan-quota failure mid-stream looks the same (`code: usage_limit_exceeded`
in the `error` event).

---

## 8. Code recipes

### 8.1 Python, official `openai` SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="not-used")

# Non-streaming
r = client.responses.create(model="gpt-6-astra", input="Say hi in three words.",
                            reasoning={"effort": "low"})
print(r.output_text)

# Streaming
with client.responses.stream(model="gpt-6-astra", input="Count to five.") as s:
    for event in s:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    final = s.get_final_response()          # usage lives here
print("\n", final.usage.input_tokens, final.usage.output_tokens)

# previous_response_id
r1 = client.responses.create(model="gpt-6-astra", input="Remember the number 42.")
r2 = client.responses.create(model="gpt-6-astra", input="What number did I tell you?",
                             previous_response_id=r1.id)

# Chat Completions
c = client.chat.completions.create(model="gpt-6-astra", reasoning_effort="low",
    messages=[{"role": "system", "content": "You are terse."},
              {"role": "user", "content": "Say hi in three words."}])
print(c.choices[0].message.content)
```

### 8.2 Python, function-tool loops

```python
import json

tools = [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]

def get_weather(city: str) -> str:
    return json.dumps({"city": city, "temp_c": 17, "sky": "cloudy"})

# Responses API, previous_response_id
r = client.responses.create(model="gpt-6-astra", input="Weather in Kyiv?", tools=tools)
while calls := [o for o in r.output if o.type == "function_call"]:
    r = client.responses.create(
        model="gpt-6-astra", tools=tools, previous_response_id=r.id,
        input=[{"type": "function_call_output", "call_id": c.call_id,
                "output": get_weather(**json.loads(c.arguments))} for c in calls])
print(r.output_text)

# Chat Completions
chat_tools = [{"type": "function", "function": tools[0]}]
messages = [{"role": "user", "content": "Weather in Oslo?"}]
while True:
    choice = client.chat.completions.create(model="gpt-6-astra", tools=chat_tools,
                                            messages=messages).choices[0]
    if choice.finish_reason != "tool_calls":
        break
    messages.append(choice.message.model_dump(exclude_none=True))
    for tc in choice.message.tool_calls:
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": get_weather(**json.loads(tc.function.arguments))})
print(choice.message.content)
```

### 8.3 Python, raw `httpx` with manual SSE parsing

```python
import httpx, json

def stream_text(prompt: str) -> None:
    with httpx.stream("POST", "http://127.0.0.1:8766/v1/responses", json={
        "model": "gpt-6-astra", "input": prompt, "stream": True,
        "reasoning": {"effort": "low"},
    }, timeout=600) as resp:
        resp.raise_for_status()
        event = None
        for line in resp.iter_lines():
            if not line:
                event = None
                continue
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "response.output_text.delta":
                    print(data["delta"], end="", flush=True)
                elif event == "error":
                    raise RuntimeError(f"{data['code']}: {data['message']}")
                elif event == "response.completed":
                    print("\n", data["response"]["usage"])

stream_text("Tell me a one-sentence joke.")
```

### 8.4 TypeScript, official `openai` SDK

```ts
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:8766/v1", apiKey: "not-used" });

// Non-streaming
const r = await client.responses.create({
  model: "gpt-6-astra", input: "Say hi in three words.", reasoning: { effort: "low" },
});
console.log(r.output_text);

// Streaming
const stream = client.responses.stream({ model: "gpt-6-astra", input: "Count to five." });
for await (const event of stream) {
  if (event.type === "response.output_text.delta") process.stdout.write(event.delta);
}
const final = await stream.finalResponse();

// Function-tool loop
const tools: OpenAI.Responses.Tool[] = [{
  type: "function", name: "get_weather", description: "Current weather for a city.", strict: false,
  parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
}];
let res = await client.responses.create({ model: "gpt-6-astra", input: "Weather in Kyiv?", tools });
for (;;) {
  const calls = res.output.filter(
    (o): o is OpenAI.Responses.ResponseFunctionToolCall => o.type === "function_call");
  if (calls.length === 0) break;
  res = await client.responses.create({
    model: "gpt-6-astra", tools, previous_response_id: res.id,
    input: calls.map((c) => ({
      type: "function_call_output" as const, call_id: c.call_id,
      output: JSON.stringify(getWeather(JSON.parse(c.arguments))),
    })),
  });
}
console.log(res.output_text);

// session_id is a Conduix extension: the SDK passes unknown fields through
const sid = (await fetch("http://127.0.0.1:8766/v1/sessions", { method: "POST" })
  .then((x) => x.json())).session_id as string;
const s1 = await client.responses.create({ model: "gpt-6-astra", input: "Remember 42.", session_id: sid } as any);
```

### 8.5 Rust, `reqwest` + `eventsource-stream`

```rust
use eventsource_stream::Eventsource;
use futures::StreamExt;

let body = serde_json::json!({"model": "gpt-6-astra", "input": "Hello", "stream": true});
let mut events = reqwest::Client::new()
    .post("http://127.0.0.1:8766/v1/responses")
    .json(&body).send().await?.bytes_stream().eventsource();

while let Some(ev) = events.next().await {
    let ev = ev?;
    let data: serde_json::Value = serde_json::from_str(&ev.data)?;
    match ev.event.as_str() {
        "response.output_text.delta" => print!("{}", data["delta"].as_str().unwrap_or("")),
        "error" => return Err(format!("{}: {}", data["code"], data["message"]).into()),
        "response.completed" | "response.incomplete" | "response.failed" => break,
        _ => {}
    }
}
```

Non-streaming is a plain POST; the function-call loop is the same JSON as
§5.2 driven by `serde_json`.

### 8.6 Structured output

```python
import json
from pydantic import BaseModel

schema = {"type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}

# Raw schema
r = client.responses.create(model="gpt-6-astra", input="Extract: 'Alice is 31.'",
                            text={"format": {"type": "json_schema", "name": "person",
                                             "schema": schema, "strict": True}})
person = json.loads(r.output_text)

# Let the SDK build the schema and parse the result
class Person(BaseModel):
    name: str
    age: int

parsed = client.responses.parse(model="gpt-6-astra", input="Extract: 'Alice is 31.'",
                                text_format=Person)
print(parsed.output_parsed)        # Person(name='Alice', age=31)

# Chat Completions
c = client.chat.completions.create(model="gpt-6-astra",
    messages=[{"role": "user", "content": "Extract: 'Alice is 31.'"}],
    response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}})
```

TypeScript: `client.responses.parse({ ..., text: { format: zodTextFormat(PersonSchema, "person") } })`
with `zodTextFormat` from `openai/helpers/zod`; read `output_parsed`. Full
rules (strictness, JSON mode, response shape): `RAG_INTEGRATION.md` §4.

### 8.7 Sessions

```python
import httpx

BASE = "http://127.0.0.1:8766"
sid = httpx.post(f"{BASE}/v1/sessions", json={"effort": "low", "instructions": "Be brief."}).json()["session_id"]
for q in ["Remember the number 42.", "What number did I tell you?"]:
    r = client.responses.create(model="gpt-6-astra", input=q, extra_body={"session_id": sid})
    print(r.output_text, r.model_extra["session_id"])
httpx.delete(f"{BASE}/v1/sessions/{sid}")
```

### 8.8 curl

```bash
# Non-streaming
curl -sS http://127.0.0.1:8766/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"gpt-6-astra","input":"ping","reasoning":{"effort":"low"}}'

# Streaming (-N disables curl's buffering)
curl -N -sS http://127.0.0.1:8766/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"gpt-6-astra","input":"Count to 5","stream":true}'

# Chat Completions
curl -sS http://127.0.0.1:8766/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gpt-6-astra","messages":[{"role":"user","content":"ping"}]}'

# Models and their efforts
curl -sS http://127.0.0.1:8766/v1/models

# Speech (when the server has CONDUIX_SPEECH=1)
curl -sS http://127.0.0.1:8766/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"juniper","input":"Hello there.","response_format":"mp3"}' -o hello.mp3
```

### 8.9 Speech

```python
import openai

try:
    speech = client.audio.speech.create(model="tts-1", voice="juniper",
                                        input="Your report is ready.", response_format="mp3")
    speech.write_to_file("ready.mp3")
except openai.APIStatusError as e:
    if e.status_code == 501:          # speech is off on this server: hide the feature
        ...
    elif e.body and e.body.get("code") == "speech_mismatch":
        ...                           # the voice paraphrased twice; retry later
    else:
        raise
```

`with_raw_response` exposes the `x-conduix-voice` header. Long texts are fine
up to 4096 characters, but each request holds the server's only voice slot for
its whole duration: split very long narration into several requests only if
you need partial results sooner.

---

## 9. Pitfalls (read before you debug)

1. **Placeholder key, and no fallback.** Send `api_key="not-used"`. Make sure the app can't fall back to a real `OPENAI_API_KEY` or to `api.openai.com` when Conduix is down; that would bill real money.
2. **`input` must end with a user message or a `function_call_output`.** An input that ends with an assistant message is a 400.
3. **Validation errors are 400, not 422.** Read `error.param` and `error.message`.
4. **Images are base64 data URLs only.** `https://` URLs and `file_id` are 400s. Only user messages may carry images. Check `input_modalities` in `/v1/models`.
5. **`input_file` is a 400.** Documents go in as `x_document` parts (verified citations, Responses API only) or as `input_text` (see `RAG_INTEGRATION.md`).
6. **Effort is per model.** `max` on `gpt-5.5` is a 400. Build the picker from `/v1/models`; don't hardcode.
7. **Effort and model are per request, not per session.** The opposite of Conduit. A session's `model` / `effort` are only defaults.
8. **One function call per response.** Handle every `function_call` in `output`, but expect one; loop until none. Send the same `tools` on every request of the loop.
9. **`call_id`, not the `fc_` item id,** goes in `function_call_output`.
10. **`previous_response_id` can 404** (store eviction, `store: false`, restart). Keep the history client-side so you can fall back to resending it.
11. **No tools with `session_id`** (400). Tool loops use `previous_response_id` or resent history.
12. **`json_object` is best effort; `json_schema` must be strict.** Every object needs `additionalProperties: false`; `strict` and `name` are ignored; a schema that only allows `{}` is rejected up front. Parse defensively in JSON mode.
13. **Don't read config back from the response.** `tools` is always `[]`, `tool_choice` `"auto"`, `temperature` `null`, `reasoning.effort` what you sent (or `null`).
14. **`output` may hold several `message` items.** Use `output_text` (SDK) or join every `output_text` part. A `message` may carry a Codex `phase` field; ignore it.
15. **Reasoning items are optional and Responses-only.** Their absence at low effort is normal. `reasoning` and `web_search_call` items in resent history are skipped silently.
16. **Watch for `error` on a 200 stream.** Responses: `error` then `response.failed`. Chat: a `{"error": ...}` chunk before `[DONE]`.
17. **A `429 usage_limit_exceeded` won't clear by retrying.** It carries `x-should-retry: false` and `retry-after`; surface the reset time to the user. `/health` shows the window.
18. **Give it time.** 3–4 s to first token at low effort; high efforts can run minutes. Don't put a 30 s timeout on a high-effort call.
19. **Parallel requests go across sessions, never within one.** A session queues; 100 busy sessions (implicit ones included) is a `503 session_limit`.
20. **Restart loses everything in memory:** sessions, stored responses, generated-image ids. The API itself is stateless-capable, so clients that keep their own history survive it.
21. **Chat Completions has no `previous_response_id`, no reasoning items, no image generation.** Prefer the Responses API for new code.
22. **`metadata` values must be strings.** Anything else is a 400.
23. **Web search cites in prose.** Expect markdown links in the text and `web_search_call` items; `annotations` stays empty. Declaring the tool is advisory; the model decides whether to search.
24. **Generated images: read the `image_generation_call` item**, not `output_text` (often empty afterwards). `image_generation_call` by `id` only works while the first response is stored.
25. **`instructions` on a session is fixed.** A different value per request is appended as a developer message, not swapped.
26. **Speech is slow and serial.** `/v1/audio/speech` takes ~13 s even for a short phrase and serves one request at a time, so don't put it on a latency-critical path or fire many in parallel. A `501` means the server has it off, not that the request was wrong.

---

## 10. Debugging recipes

### 10.1 `400 input must end with a user message or function_call_output`
Your history ends with an assistant message, or the only user content is an `image_generation_call` item. Append the new user turn last.

### 10.2 `400 no tool call found for function call output with call_id '...'`
The `call_id` is not in the continued response's history (wrong `previous_response_id`, or you continued a different branch), or you resent the history without the `function_call` item. Check you copied `call_id` (not the `fc_...` item id).

### 10.3 `404 not_found` on `previous_response_id` or `session_id`
Evicted (1000 responses / 256 MB), `store: false`, idle 30 min, or a restart. Resend the history (§4.1) or start a new session.

### 10.4 `400 unsupported_value` on effort
Not in that model's `efforts`. `GET /v1/models/{id}` lists them. Chat spells the param `reasoning_effort`.

### 10.5 `429 usage_limit_exceeded`
The plan's window is used up. `GET /health` → `usage.primary.resets_at`. Nothing to fix client-side; show the reset time.

### 10.6 `503 codex_unavailable`, or `/health` says `degraded`
The Codex app-server process died. Restart Conduix (`start_app.ps1`).

### 10.7 `503 session_limit`
100 sessions and every one is mid-turn. Lower concurrency, or raise `CONDUIX_MAX_SESSIONS` on the server.

### 10.8 Empty `output_text`
The response ended on a `function_call` (look in `output`), or on an `image_generation_call` (the image is in `result`), or the turn failed inside a stream (`response.failed`). An empty text on a normal turn is not something Conduix produces silently: check `status` and `error`.

### 10.9 Stream starts late or seems stuck
3–4 s before the first token is normal at low effort; high effort can reason for minutes before any `output_text.delta`. Ask for `reasoning.summary` to see progress as `reasoning_summary_text.delta` events.

### 10.10 Structured output errors
`400 invalid_json_schema` carries the upstream message: usually a non-strict schema (`additionalProperties` missing somewhere) or a schema that only allows `{}`. `json_object` output that isn't an object: that mode is instructed, not enforced. See `RAG_INTEGRATION.md` §4.

### 10.11 `400 model '...' does not accept image input`
That model's `input_modalities` lacks `image`. Pick another from `/v1/models`.

### 10.12 Confirm wire-format parity
Run the official `openai` Python or TypeScript SDK against the base URL. If it works there, your own client's shapes are right. The server's own tests validate every response body against the `openai` package's types.

### 10.13 See what the server sees
Server logs go to stdout at INFO: thread rebuilds (`rebuilding history of resp_... onto a new thread`), interrupted turns, declined approvals, blocked agent items. `/health` → `blocked_agent_items` should be `{}`.

---

## 11. Versioning expectations

Conduix is pre-1.0 (`version` `0.1.0` on `/health`; feature set v1.3) and
tracks the OpenAI API loosely:

- Request fields it doesn't know are accepted and ignored on purpose, so
  newer SDKs keep working.
- Response shapes are validated against the pinned `openai` package's types;
  the Codex SDK is pinned (`openai-codex==0.157.1`).
- Model ids and `efforts` are whatever the plan serves and may change
  without a Conduix release. Always read `/v1/models`.
- Speech (`/v1/audio/speech`) is experimental: it rides an experimental Codex
  protocol and may stop working after a Codex update without a Conduix
  release.
- Not planned for v1: parallel function calls, tools on sessions, `input_file`,
  `url_citation` annotations, sampling parameters, streaming speech,
  transcription, persistence across restarts.

---

## 12. Minimum viable integration checklist

- [ ] `GET /health` returns 200 with `"logged_in": true`.
- [ ] `POST /v1/responses` with `{"model": "<from /v1/models>", "input": "ping"}` returns `status: "completed"` and non-empty `output_text`.
- [ ] The same with `"stream": true` yields `response.output_text.delta` events and ends with `response.completed`.
- [ ] `reasoning.effort` taken from the model's `efforts`; a bad value gives `400 unsupported_value`.
- [ ] Multi-turn: either resend the history, or `previous_response_id` with a fallback to resending on `404`.
- [ ] Function tools: handle a `function_call` in `output`, answer with `function_call_output` carrying its `call_id`, loop until none.
- [ ] Structured output: a strict `json_schema` returns parseable `output_text`; a non-strict schema gives `400 invalid_json_schema`.
- [ ] Errors: `error.message` / `error.code` are surfaced; `429 usage_limit_exceeded` is not retried; `error` events on streams are handled.
- [ ] Speech (if used): a `501` hides the feature; `audio.speech.create` returns playable audio; the client timeout allows ~1 min.
- [ ] The client cannot fall back to `api.openai.com` or a real key.

---

## 13. Coming from Conduit

| | Conduit (`:8765`) | Conduix (`:8766`) |
|---|---|---|
| Wire format | Anthropic `/v1/messages` | OpenAI `/v1/responses`, `/v1/chat/completions` |
| Backend / plan | Claude Agent SDK, Claude Max | Codex app-server, ChatGPT plan |
| Validation errors | 422 (pydantic list) | **400**, OpenAI envelope with `param` |
| Error envelope | `{"detail": {"type", "message"}}` | `{"error": {"message", "type", "param", "code"}}` |
| Model / effort / structured output | Bound at session creation | **Per request** |
| Effort spelling | `effort` extension or `output_config.effort` | `reasoning.effort` (standard) |
| Effort values | Fixed five | Per model, from `/v1/models` |
| Function tools | Pause/resume on a server session, `x-conduit-session-id` header, parallel calls | Stateless `function_call` → `function_call_output`, `previous_response_id`, one call per response |
| Tools + session | Required (the session holds the pause) | Forbidden (400) |
| Server-side history | `session_id` only | `previous_response_id` (branching, 1000-entry store) and `session_id` |
| Thinking | `include_thinking` → `thinking` blocks | `reasoning.summary` → `reasoning` items |
| Documents | `document` blocks, verbatim | `x_document` parts (Conduix extension) or `input_text`; `input_file` is a 400 |
| Citations | Native `citations` / `citations_delta` | `x_document` parts → `x_document_citation` annotations (Conduix extension, `RAG_INTEGRATION.md` §2.0) |
| Structured output | `output_config.format` → one `text` block | `text.format` / `response_format` → one `output_text` |
| Hosted tools | `web_search`, `web_fetch` | `web_search`, `image_generation` |
| Images | Input only | Input and generation |
| Speech | None | `/v1/audio/speech`, experimental, off by default |
| Quota error | `429 rate_limit_error` + reset in message | `429 usage_limit_exceeded` + `retry-after` + `x-should-retry: false` |
| Usage | `input_tokens`, cache fields | OpenAI usage, plus ~4.4k fixed overhead |

Both run side by side on one machine; FounderDesk-style apps can point an
Anthropic-shaped provider at Conduit and an OpenAI-shaped provider at Conduix.
