# Conduix API Usage Guide

How to connect a client to Conduix and use it.

> **Status: specification.** This guide describes the v1 API planned in
> [`BRIEF.md`](./BRIEF.md). The server is **not built yet**. Event names, usage
> fields, models, and effort values below come from the step-2 SDK probe
> (2026-09-26, `openai-codex` 0.157.1). Everything else is the design target
> and may change during implementation.

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
   {"status": "ok", "codex_version": "0.157.1", "auth": "chatgpt", "plan_type": "plus"}
   ```

If `/health` does not show `"auth": "chatgpt"`, the server refuses to serve
requests. This is on purpose, so nothing is ever billed to an API key.

A Swagger console for trying requests by hand is at
<http://127.0.0.1:8766/docs>.

---

## 3. Endpoints

| Method | Path                    | Use it for |
| ------ | ----------------------- | ---------- |
| POST   | `/v1/responses`         | **Recommended.** OpenAI Responses API |
| POST   | `/v1/chat/completions`  | OpenAI Chat Completions, for older clients and tools |
| GET    | `/v1/models`            | Models your plan can use |
| POST   | `/v1/sessions`          | Create a server-side conversation → `{session_id}` |
| GET    | `/v1/sessions`          | List active sessions |
| DELETE | `/v1/sessions/{id}`     | End a session |
| GET    | `/health`               | Liveness + login status |

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

### Responses API

```python
with client.responses.stream(model="gpt-6-astra", input="Count to five.") as s:
    for event in s:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    final = s.get_final_response()
```

Events you will see, in order:

| Event | Meaning |
| ----- | ------- |
| `response.created` | Request accepted |
| `response.output_item.added` | A message (or reasoning) item started |
| `response.output_text.delta` | Next chunk of text |
| `response.output_text.done` | Full text of the item |
| `response.output_item.done` | Item finished |
| `response.completed` | Done. Carries the final `response`, including `usage` |
| `error` | Something failed mid-stream (see §11) |

### Chat Completions

```python
stream = client.chat.completions.create(
    model="gpt-6-astra", stream=True,
    messages=[{"role": "user", "content": "Count to five."}],
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The stream is a series of `chat.completion.chunk` objects, ending with
`data: [DONE]`.

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

---

## 6. Multi-turn conversations

There are three ways to continue a conversation. Pick one.

### A. Resend the whole history (stateless, works like the real API)

Send every earlier turn each time. The server keeps nothing.

```python
history = [{"role": "user", "content": "Remember the number 42."}]
r1 = client.responses.create(model="gpt-6-astra", input=history)

history += [
    {"role": "assistant", "content": r1.output_text},
    {"role": "user", "content": "What number did I tell you?"},
]
r2 = client.responses.create(model="gpt-6-astra", input=history)
```

This is the simplest option, but every turn resends the full history.

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

Session rules:

- **One request at a time per session.** A second request waits for the first
  one to finish. Use separate sessions to run requests in parallel.
- **Idle sessions expire** after 30 minutes (`CONDUIX_SESSION_IDLE_TIMEOUT_S`).
  There is a cap of 100 sessions at once.
- **Sessions are kept in memory only.** A server restart loses them, and so
  does any conversation continued with `previous_response_id`.

---

## 7. Models and reasoning effort

List what your plan can use:

```python
for m in client.models.list():
    print(m.id)
```

Models seen on the current plan:

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
the list depends on the plan.

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
`reasoning={"effort": "high", "summary": "auto"}`. They come back as
`reasoning` items in `output`. This is **not guaranteed**: at low effort the
summary came back empty in testing. Chat Completions never returns reasoning.

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

**`https://` image URLs are rejected with 400.** Download the image and send
it as a data URL instead.

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
    }},
)
```

This is planned for v1 through Codex's `output_schema` and has not been
verified yet.

---

## 10. Usage and token counts

Each response carries OpenAI-style `usage`:

```json
"usage": {
  "input_tokens": 4481,
  "input_tokens_details": { "cached_tokens": 4224 },
  "output_tokens": 47,
  "output_tokens_details": { "reasoning_tokens": 34 },
  "total_tokens": 4528
}
```

- **Expect about 4.4k input tokens even for "hi".** That is Codex's own fixed
  prompt overhead. From the second turn on, most of it is cached
  (`cached_tokens`).
- The counts are real, but they count against your **plan's usage limits**,
  not a dollar bill.

---

## 11. Errors

Errors use OpenAI's error envelope:

```json
{ "error": { "type": "rate_limit_exceeded", "message": "...", "code": "..." } }
```

| Status | Meaning | What to do |
| ------ | ------- | ---------- |
| 400 | Bad request: unsupported `effort`, remote image URL, invalid body | Fix the request |
| 401 / 503 | Codex is not logged in, or the login is not a ChatGPT account | Server operator runs `codex login` |
| 404 | Unknown `session_id` / `previous_response_id` (expired or server restarted) | Start a new conversation, or resend the history (§6A) |
| 429 | **Plan usage limit reached.** The message includes the reset time when Codex provides it | Don't retry in a loop, because it won't clear until the quota resets |
| 429 / 503 | Transient overload | Retry with backoff (the `openai` SDK does this automatically) |
| 500 / 502 | Codex failed during the turn | Retry once; if it persists, check server logs |

**While streaming**, a failure arrives as an SSE `error` event before the
stream ends, not as a dropped connection. Always handle the `error` event.

---

## 12. What is ignored or not supported (v1)

| Parameter / feature | Behaviour |
| ------------------- | --------- |
| `temperature`, `top_p`, `stop`, `max_output_tokens` | Accepted, but may be **ignored**; `/docs` lists which are applied |
| Function tools (`tools`, `tool_choice`) | **Not in v1** (planned for v1.1) |
| Web search tool | **Not in v1** (planned for v1.1) |
| Embeddings, audio, image generation, files, Assistants, batch | Not available |
| `https://` image URLs | Rejected (400) |
| Parallel requests on one session | Queued, one at a time |

The model has **no access** to your files, shell, or network. Conduix runs
Codex in a locked-down, chat-only mode.

---

## 13. Tips

- **Pick the right multi-turn style.** Use §6A for full control and
  resilience to server restarts. Use §6B or §6C to send fewer tokens per turn.
- **Run requests in parallel across sessions**, not within one.
- **Reuse one client object.** Conduix keeps one Codex process for all
  requests, so there is no per-request startup cost.
- **Keep effort low for chat.** Raise it only when the task needs it, because
  it costs both latency and plan quota.
- **It runs alongside Conduit.** Conduit (Claude, Anthropic API) is on `:8765`
  and Conduix (OpenAI API) is on `:8766`.
