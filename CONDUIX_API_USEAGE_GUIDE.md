# Conduix API Usage Guide

How to connect a client to Conduix and use it.

> **Status: v1.1, built and tested.** Everything below works against a live
> ChatGPT plan (`openai-codex` 0.157.1): Responses and Chat Completions,
> streaming, three multi-turn styles, images, structured output, function
> tools and web search. Model names and limits depend on the plan that Codex
> is logged in to.

---

## 1. What Conduix is

Conduix is a **local server that speaks the OpenAI API**. Your requests are
answered by OpenAI models through your **ChatGPT subscription** (via Codex),
not a metered API key.

If your code already talks to `https://api.openai.com/v1`, point it at
Conduix instead. Nothing else changes.

| | |
|---|---|
| Base URL | `http://127.0.0.1:8766/v1` |
| Auth | None. Send any non-empty API key; it is ignored. |
| Wire format | OpenAI Responses API and Chat Completions (JSON + SSE) |
| Reachable from | This machine only (bound to `127.0.0.1`) |

---

## 2. Before you connect

The person running the server needs to have:

1. Logged in to Codex once with a ChatGPT account (`codex login`).
2. Started Conduix:
   ```powershell
   .\start_app.ps1    # or: conda run -n conduix python -m uvicorn conduix.app:app --port 8766
   ```
3. Checked that it is up:
   ```powershell
   curl http://127.0.0.1:8766/health
   ```
   Example output:
   ```json
   {
     "status": "ok",
     "version": "0.1.0",
     "codex": "ok",
     "codex_version": "0.157.1",
     "logged_in": true,
     "account_type": "chatgpt",
     "plan_type": "plus",
     "usage": {"primary": {"used_percent": 4, "window_duration_mins": 10080, "resets_at": 1791131515}},
     "blocked_agent_items": {}
   }
   ```

The server **refuses to start** unless Codex is logged in with a ChatGPT
account, and it removes `OPENAI_API_KEY` / `CODEX_API_KEY` from its own
environment, so nothing is ever billed to an API key.

- `usage` is how much of the plan's usage window is used, and when it resets
  (Unix seconds).
- `status` turns `"degraded"` if the Codex process stops responding.
- `blocked_agent_items` counts shell / file / approval attempts Conduix
  refused. It should stay empty.

A Swagger console for trying requests by hand is at
<http://127.0.0.1:8766/docs>.

---

## 3. Endpoints

| Method | Path                    | Use it for |
| ------ | ----------------------- | ---------- |
| POST   | `/v1/responses`         | **Recommended.** OpenAI Responses API |
| POST   | `/v1/chat/completions`  | OpenAI Chat Completions, for older clients and tools |
| GET    | `/v1/models`            | Models your plan can use |
| GET    | `/v1/models/{id}`       | One model, with its supported `effort` values |
| POST   | `/v1/sessions`          | Create a server-side conversation → `{session_id}` |
| GET    | `/v1/sessions`          | List active sessions |
| DELETE | `/v1/sessions/{id}`     | End a session |
| GET    | `/health`               | Liveness, login status and plan usage |
| GET    | `/docs`                 | Swagger UI |

---

## 4. Quick start

### Python (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="not-used")

r = client.responses.create(
    model="gpt-6-astra",
    input="Say hi in three words.",
)
print(r.output_text)
```

### TypeScript (`openai` SDK)

```ts
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:8766/v1", apiKey: "not-used" });

const r = await client.responses.create({
  model: "gpt-6-astra",
  input: "Say hi in three words.",
});
console.log(r.output_text);
```

### curl

```bash
curl http://127.0.0.1:8766/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-6-astra", "input": "Say hi in three words."}'
```

### Chat Completions (for tools that only support this endpoint)

```python
c = client.chat.completions.create(
    model="gpt-6-astra",
    messages=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hi in three words."},
    ],
)
print(c.choices[0].message.content)
```

---

## 5. Streaming

Set `stream=True`. Conduix sends real token-by-token deltas. Expect roughly
3–4 s before the first token at low effort; after that, tokens arrive quickly.
If the client disconnects mid-stream, Conduix stops the Codex turn so it
doesn't keep using your plan quota.

### Responses API

```python
with client.responses.stream(model="gpt-6-astra", input="Count to five.") as s:
    for event in s:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    final = s.get_final_response()
```

The events follow api.openai.com's order:

| Event | Meaning |
| ----- | ------- |
| `response.created`, `response.in_progress` | Request accepted |
| `response.output_item.added` | An item started: `message`, `reasoning`, `function_call` or `web_search_call` |
| `response.content_part.added` | A message's text part started |
| `response.output_text.delta` | Next chunk of text |
| `response.output_text.done`, `response.content_part.done` | Full text of the part |
| `response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta` / `.done`, `response.reasoning_summary_part.done` | Reasoning summary (§7) |
| `response.function_call_arguments.delta` / `.done` | A function call's arguments (§10) |
| `response.web_search_call.in_progress` / `.searching` / `.completed` | A web search (§11) |
| `response.output_item.done` | Item finished |
| `response.completed` | Done. Carries the final `response`, including `usage` |
| `error`, then `response.failed` | The turn failed (see §13) |

Every stream ends with exactly one of `response.completed`,
`response.failed` or `response.incomplete`.

### Chat Completions

```python
stream = client.chat.completions.create(
    model="gpt-6-astra", stream=True,
    messages=[{"role": "user", "content": "Count to five."}],
    stream_options={"include_usage": True},   # optional: a final chunk with usage
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The stream is a series of `chat.completion.chunk` objects, ending with
`data: [DONE]`. The first chunk carries `role: "assistant"`, and the last
one carries `finish_reason` (`"stop"`, or `"tool_calls"` when the model
called a function, §10). With `include_usage`, one more chunk follows with
empty `choices` and the `usage`. A failure arrives as a `data: {"error":
{...}}` chunk, which the `openai` SDK raises as an exception.

### Raw SSE (Rust, or any HTTP client)

```rust
use eventsource_stream::Eventsource;
use futures::StreamExt;

let body = serde_json::json!({
    "model": "gpt-6-astra",
    "input": "Hello",
    "stream": true
});
let mut events = reqwest::Client::new()
    .post("http://127.0.0.1:8766/v1/responses")
    .json(&body)
    .send().await?
    .bytes_stream()
    .eventsource();

while let Some(ev) = events.next().await {
    let ev = ev?;
    println!("{}: {}", ev.event, ev.data);
}
```

Responses frames carry an `event:` line with the event type. Chat
Completions frames are `data:` only.

---

## 6. Multi-turn conversations

There are three ways to continue a conversation. Pick one.

### A. Resend the whole history (stateless, works like the real API)

Send every earlier turn each time. The server keeps nothing between requests.

```python
history = [{"role": "user", "content": "Remember the number 42."}]
r1 = client.responses.create(model="gpt-6-astra", input=history)

history += [
    {"role": "assistant", "content": r1.output_text},
    {"role": "user", "content": "What number did I tell you?"},
]
r2 = client.responses.create(model="gpt-6-astra", input=history)
```

- Earlier turns reach the model as real user / assistant / system messages,
  not as text pasted into the prompt.
- You can append `r1.output` items directly. Reasoning and
  `web_search_call` items in it are skipped.
- `function_call` / `function_call_output` items are accepted too (§10).
- For Chat Completions this is just `messages`.

This is the most robust option: it survives server restarts.

### B. `previous_response_id` (standard Responses API)

Send only the new message, and point at the previous response. The server
keeps the conversation in memory.

```python
r1 = client.responses.create(model="gpt-6-astra", input="Remember the number 42.")
r2 = client.responses.create(
    model="gpt-6-astra",
    input="What number did I tell you?",
    previous_response_id=r1.id,
)
```

- **Branching works.** Two requests can both continue `r1`, and each sees
  only `r1`'s history.
- You can change `instructions`, `model` or `tools` between turns.
- `store: false` keeps a response out of memory, so it can't be continued.
- The server keeps the last 1000 responses (at most 256 MB, since stored
  history can include images). Older ones return **404**; resend the history
  instead (§6A).

### C. `session_id` (Conduix extension)

Create a session explicitly, then send only the new turn each time. This is
useful when you want to name, list, or delete conversations. It works on both
`/v1/responses` and `/v1/chat/completions`.

```python
import httpx

sid = httpx.post("http://127.0.0.1:8766/v1/sessions", json={}).json()["session_id"]

for q in ["Remember the number 42.", "What number did I tell you?"]:
    r = client.responses.create(
        model="gpt-6-astra",
        input=q,
        extra_body={"session_id": sid},   # non-standard field → extra_body
    )
    print(r.output_text)

httpx.delete(f"http://127.0.0.1:8766/v1/sessions/{sid}")
```

`POST /v1/sessions` takes optional defaults: `model`, `effort` and
`instructions`. A request's own `model` / `reasoning.effort` override the
session's. `instructions` are fixed when the session starts. A request that
sends different ones adds them to the conversation as a system message.

Session rules:

- **One request at a time per session.** A second request waits for the first
  one to finish. Use separate sessions to run requests in parallel.
- **Idle sessions expire** after 30 minutes (`CONDUIX_SESSION_IDLE_TIMEOUT_S`).
  There is a cap of 100 sessions at once; the least recently used idle one
  makes room.
- **Sessions can't use tools** (§10). Use §6A or §6B for tool loops.
- **Sessions are kept in memory only.** A server restart loses them, and so
  does any conversation continued with `previous_response_id`.

---

## 7. Models and reasoning effort

List what your plan can use:

```python
for m in client.models.list():
    print(m.id, m.model_extra["efforts"])
```

Each model also reports `is_default`, `default_effort`, `efforts` and
`input_modalities`. Models seen on the current plan:

| Model | Default | Default effort | Supported `effort` |
| ----- | ------- | -------------- | ------------------ |
| `gpt-6-astra`   | ✅ | low    | low, medium, high, xhigh, max, ultra |
| `gpt-6-sol`     |    | medium | low, medium, high, xhigh, max, ultra |
| `gpt-6-luna`    |    | medium | low, medium, high, xhigh, max |
| `gpt-5.6-sol`   |    | low    | low, medium, high, xhigh, max, ultra |
| `gpt-5.6-terra` |    | medium | low, medium, high, xhigh, max, ultra |
| `gpt-5.6-luna`  |    | medium | low, medium, high, xhigh, max |
| `gpt-5.5`       |    | medium | low, medium, high, xhigh |

If `model` is omitted, the server default is used (`CONDUIX_DEFAULT_MODEL`,
or else Codex's default, `gpt-6-astra`). Always check `/v1/models`, because
the list depends on the plan. An unknown model returns **400**.

Set how hard the model thinks with `reasoning.effort`:

```python
r = client.responses.create(
    model="gpt-6-sol",
    input="Prove that sqrt(2) is irrational.",
    reasoning={"effort": "high"},
)
```

For Chat Completions, use `reasoning_effort="high"`. An effort level the model
doesn't support returns **400**. Higher effort means slower responses and more
of your plan quota used.

**Reasoning summaries.** You can ask for them with
`reasoning={"effort": "high", "summary": "auto"}` (or `"concise"` /
`"detailed"`). They come back as `reasoning` items in `output`. They only
appear when the model actually reasons: at low effort, or on an easy
question, there may be none. A hard problem at high effort can produce dozens
of short reasoning items. Chat Completions never returns reasoning.

---

## 8. Images

Send images as **base64 data URLs**. All current models accept image input.

```python
import base64

b64 = base64.b64encode(open("chart.png", "rb").read()).decode()

r = client.responses.create(
    model="gpt-6-astra",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What does this chart show?"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ],
    }],
)
```

For Chat Completions, use `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.

**`https://` image URLs are rejected with 400** (`invalid_image_url`).
Download the image and send it as a data URL instead.

- Types: PNG, JPEG, GIF, WebP. Up to 20 MB per image.
- Images can only be in `user` messages. `file_id` images are not supported.
- Images in earlier turns are remembered: in resent history, with
  `previous_response_id`, and in sessions.
- Sending an image to a model without image input returns **400**.

---

## 9. Structured output (JSON schema)

```python
r = client.responses.create(
    model="gpt-6-astra",
    input="Extract: 'Alice is 31 and lives in Kyiv.'",
    text={"format": {
        "type": "json_schema",
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
            "additionalProperties": False,
        },
        "strict": True,
    }},
)
```

`r.output_text` is then a JSON string matching the schema. For Chat
Completions, use `response_format={"type": "json_schema", "json_schema":
{"name": ..., "schema": ...}}`.

Only `json_schema` is supported; `{"type": "json_object"}` returns 400. An
invalid schema returns **400** `invalid_json_schema`, with the upstream
message.

---

## 10. Function tools

Declare functions as usual. When the model calls one, the response ends with
a `function_call` item. Run it and send the result back.

### Responses API

```python
import json

tools = [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]

def get_weather(city):
    return json.dumps({"city": city, "temp_c": 17, "sky": "cloudy"})

r = client.responses.create(model="gpt-6-astra", input="Weather in Kyiv?", tools=tools)
while calls := [o for o in r.output if o.type == "function_call"]:
    r = client.responses.create(
        model="gpt-6-astra", tools=tools, previous_response_id=r.id,
        input=[{"type": "function_call_output", "call_id": c.call_id,
                "output": get_weather(**json.loads(c.arguments))} for c in calls],
    )
print(r.output_text)
```

Instead of `previous_response_id`, you can also resend the whole history
with the `function_call` and `function_call_output` items in it (§6A).

### Chat Completions

```python
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

messages = [{"role": "user", "content": "Weather in Oslo?"}]
while True:
    choice = client.chat.completions.create(
        model="gpt-6-astra", tools=tools, messages=messages).choices[0]
    if choice.finish_reason != "tool_calls":
        break
    messages.append(choice.message.model_dump(exclude_none=True))
    for tc in choice.message.tool_calls:
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": get_weather(**json.loads(tc.function.arguments))})
print(choice.message.content)
```

When streaming, the call arrives as `delta.tool_calls` chunks (id and name
first, then the arguments), and the stream ends with `finish_reason:
"tool_calls"`.

### Rules

- **Send the same `tools` on every request of the loop.**
- **One function call comes back per response.** A model that wants several
  calls makes them over successive round trips. Handle every
  `function_call` in `output` anyway.
- **Unknown call ids are rejected.** A `function_call_output` whose
  `call_id` doesn't match an earlier call returns **400**.
- **`tool_choice`:** `"none"` turns tools off. `"required"` and forcing a
  specific function can't be enforced and behave like `"auto"`.
- **No sessions:** tools can't be combined with `session_id` (400).
- **Other hosted tools return 400:** file search, code interpreter, MCP and
  computer use. Function names must match `[A-Za-z0-9_-]{1,64}`.
- **Reasoning doesn't carry over:** the model's hidden reasoning isn't kept
  across a tool call. This is the same as OpenAI's stateless function
  calling.

---

## 11. Web search

```python
r = client.responses.create(model="gpt-6-astra", tools=[{"type": "web_search"}],
                            input="What is the latest stable Python release?")
print(r.output_text)
```

- The answer cites its sources as markdown links.
- `output` includes `web_search_call` items, with `search`, `open_page` or
  `find_in_page` actions.
- `web_search_preview` works too.
- For Chat Completions, pass `web_search_options={}`.
- You can combine web search with function tools in one request.

Searches run live by default. The server operator can set
`CONDUIX_WEB_SEARCH_MODE=cached` to use Codex's search cache instead. Without
the tool, the model has no web access.

---

## 12. Usage and token counts

Each response carries OpenAI-style `usage`:

```json
"usage": {
  "input_tokens": 4481,
  "input_tokens_details": { "cached_tokens": 4224, "cache_write_tokens": 0 },
  "output_tokens": 47,
  "output_tokens_details": { "reasoning_tokens": 34 },
  "total_tokens": 4528
}
```

- **Chat Completions naming:** the same numbers appear as `prompt_tokens`,
  `completion_tokens`, `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens`.
- **Every model call is counted.** A turn that calls a tool or searches the
  web makes several model calls, and `usage` covers all of them.
- **Expect about 4.4k input tokens even for "hi".** That is Codex's own fixed
  prompt overhead. Much of it is cached (`cached_tokens`).
- **Plan limits, not money.** The counts are real, but they count against
  your plan's usage limits, not a dollar bill. `GET /health` shows how much
  of the usage window is used.

---

## 13. Errors

Errors use OpenAI's error envelope:

```json
{ "error": { "type": "rate_limit_exceeded", "message": "...", "param": null, "code": "usage_limit_exceeded" } }
```

| Status | `code` | Meaning | What to do |
| ------ | ------ | ------- | ---------- |
| 400 | `unsupported_value`, `model_not_found`, `invalid_json_schema`, `context_length_exceeded`, ... | Bad request: unsupported `effort`, unknown model, invalid schema, unknown tool `call_id`, conversation too long | Fix the request; `param` names the field |
| 400 | `invalid_image_url`, `unsupported_image_media_type`, `invalid_base64_image`, `image_too_large`, `empty_image_file` | Bad image (§8) | Send a valid base64 data URL |
| 401 | `codex_unauthorized` | Codex's login is no longer valid | Server operator runs `codex login` |
| 404 | `not_found` | Unknown `session_id` / `previous_response_id` / model (expired or server restarted) | Start a new conversation, or resend the history (§6A) |
| 429 | `usage_limit_exceeded` | **Plan usage limit reached.** The message says when it resets; `retry-after` is set and `x-should-retry: false` stops the `openai` SDK retrying | Wait for the reset; retrying won't help |
| 429 | `rate_limit_exceeded` | Transient rate limit | Retry with backoff (the `openai` SDK does this automatically) |
| 503 | `server_overloaded`, `codex_unavailable`, `session_limit` | Upstream overloaded, the Codex process died, or all 100 sessions are busy | Retry; for `codex_unavailable`, restart Conduix |
| 500 / 502 | `server_error`, `upstream_error` | Codex failed during the turn | Retry once; if it persists, check server logs |

**While streaming**, a failure never drops the connection:

- **Responses API:** an `error` event, then `response.failed`, whose
  `response.error` carries the details. Always handle the `error` event.
- **Chat Completions:** a `data: {"error": {...}}` chunk, then `[DONE]`.
  The `openai` SDK raises it as an exception.

---

## 14. What is ignored or not supported

| Parameter / feature | Behaviour |
| ------------------- | --------- |
| `temperature`, `top_p`, `stop`, `max_output_tokens`, `max_tokens`, `seed`, `user`, `parallel_tool_calls`, `logprobs` | Accepted and **ignored**: Codex has no setting for them |
| `metadata` | Echoed back in the response |
| `n` > 1 (Chat Completions) | 400 |
| `text.format` / `response_format` `json_object` | 400; use `json_schema` |
| `tool_choice` `"required"` or a named function | Behaves like `"auto"` |
| Web search citations | Markdown links in the text; no `url_citation` annotations |
| Other hosted tools (file search, code interpreter, MCP, computer use) | 400 |
| Tools together with `session_id` | 400 |
| Legacy function calling (Chat Completions) | `function_call` / `role: "function"` messages return 400, and a `functions` parameter is ignored; use `tools` |
| `https://` image URLs, `file_id` images | 400 |
| Embeddings, audio, image generation, files, Assistants, batch | Not available |
| Parallel requests on one session | Queued, one at a time |

The model has **no access** to your files or shell, and no network access
beyond the web search you enable. Conduix runs Codex in a locked-down mode:
- Approvals are declined.
- The sandbox is read-only, in an empty workspace.
- Codex's own agent tools are off.

---

## 15. Tips

- **Pick the right multi-turn style.** Use §6A for full control and
  resilience to server restarts. Use §6B or §6C to send fewer tokens per turn.
- **Run requests in parallel across sessions**, not within one.
- **Reuse one client object.** Conduix keeps one Codex process for all
  requests, so there is no per-request startup cost.
- **Keep effort low for chat.** Raise it only when the task needs it, because
  it costs both latency and plan quota.
- **Keep tool loops short.** Every tool round trip is another request, and the
  history is replayed each time.
- **It runs alongside Conduit.** Conduit (Claude, Anthropic API) is on `:8765`
  and Conduix (OpenAI API) is on `:8766`.
