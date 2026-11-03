# Brief for Conduix: verified citations over inline documents

From FounderDesk, 2026-11-03. FounderDesk wants Conduix as a second answer
backend next to Conduit, picked per answer from a vendor/model dropdown. The
answer pipeline (Ask, the Review drawer's Explain) depends on one thing
Conduix does not have today: **citations that point at exact character spans
of the documents sent with the question**, interleaved with the answer text.
This brief describes that feature as a Conduix extension to the Responses API.
Everything else FounderDesk needs already exists (§6).

Conduix's own `RAG_INTEGRATION.md` §2 says the gap is in the wire format and
offers two client-side conventions. FounderDesk can live with convention B,
but every client would re-implement the same prompt, parsing and quote
verification. Doing it once, server-side, behind a request flag, makes
Conduix a drop-in for any Anthropic-style RAG client.

---

## 1. What FounderDesk sends

Per question, one stateless request (no `previous_response_id`, no
`session_id`): the system prompt, the earlier turns as plain text, then the
newest user turn holding up to ~40 retrieved windows and the question.

Each window is a document with four fields:

| Field | Example | Notes |
|---|---|---|
| `id` | `01K8…:3-7` | Opaque; FounderDesk resolves citations by it. |
| `title` | `Checking Plus Custom Metric Rules` | |
| `context` | `IN RANGE — dated 2026-11-02 · transcript` | Labels the model must repeat, not infer. |
| `text` | the window's text, up to ~6k characters | Transcripts, notes, tickets. Untrusted. |

## 2. The extension

### 2.1 Request

A new content part type on user messages, next to `input_text`:

```json
{"type": "x_document",
 "id": "doc-1", "title": "Ops update", "context": "IN RANGE — dated 2026-03-14",
 "text": "The warehouse moved to Tallinn in March. The lease runs to 2029.",
 "citations": true}
```

- Allowed only in the newest user message, before the question text. Order
  defines `document_index` (0-based), as with Anthropic `document` blocks.
- `citations: false` (or absent) means the document is context only.
- Any name works if you prefer another (`input_document`, or a top-level
  `documents` array plus `x_citations: true`); keep it off the standard
  OpenAI field names so a real OpenAI client never sends it by accident.
- Documents are data: Conduix should wrap them so the model treats their
  content as evidence, never as instructions.

### 2.2 Response, non-streaming

The answer's `output_text` parts carry annotations of a new type. Split the
text so each annotated part ends where its claim ends (the Anthropic shape:
several text parts, the cited ones annotated), **or** keep one part and give
each annotation the offsets of the answer span it supports. Either works;
say which in the docs.

```json
{"type": "output_text",
 "text": "The warehouse is in Tallinn.",
 "annotations": [
   {"type": "x_document_citation",
    "document_index": 0,
    "document_id": "doc-1",
    "cited_text": "The warehouse moved to Tallinn in March.",
    "start_char_index": 0,
    "end_char_index": 40}
 ]}
```

### 2.3 Response, streaming (the important part)

FounderDesk streams answers to the screen. Citations must arrive **in order
with the text**, so a citation applies to the text streamed since the
previous citation (or the start). Reuse the standard Responses event that
already has this slot:

```
event: response.output_text.delta
data: {"type":"response.output_text.delta", ..., "delta":"The warehouse is in Tallinn."}

event: response.output_text.annotation.added
data: {"type":"response.output_text.annotation.added", "item_id":"...", "output_index":0,
       "content_index":0, "annotation_index":0,
       "annotation":{"type":"x_document_citation","document_index":0,"document_id":"doc-1",
                     "cited_text":"The warehouse moved to Tallinn in March.",
                     "start_char_index":0,"end_char_index":40}}
```

The answer text itself must contain no citation markers by the time it
reaches the client.

### 2.4 How you might build it

Your call; one approach that streams:

1. Instruct the model (in Conduix's own wrapper, not the client's
   instructions) to follow each claim with an inline marker carrying the
   document index and an exact quote, e.g. `⟦0|The warehouse moved to Tallinn⟧`.
2. While streaming, hold back text from the first `⟦` until the marker
   closes; emit the text before it as deltas, parse the marker, verify the
   quote, emit an `annotation.added` event (or drop it), and continue.
3. Verification: `document_index` in range, and the quote found in that
   document's `text`, exactly or with whitespace runs matched loosely. The
   offsets you report are those of the span **in the document text**, and
   `cited_text` is that slice, so `text[start:end] == cited_text` holds.

Convention B run server-side (JSON answer in parts, each with quotes, then
re-emitted as deltas and annotations) is acceptable for a first version.
It doesn't stream, but the client contract stays the same.

## 3. Invariants FounderDesk relies on (please test them)

1. `0 <= document_index < number of x_document parts` on every annotation.
2. `document.text.slice(start_char_index, end_char_index) === cited_text`,
   in UTF-16 code units (JavaScript string indices). Say so if you count
   code points instead; transcripts contain non-ASCII text.
3. No annotation is ever emitted for a quote that is not in its document.
   Report the count of dropped quotes, e.g.
   `usage.x_citations: {"emitted": 3, "dropped": 1}`.
4. In a stream, an annotation follows the text it supports and precedes the
   next sentence's text.
5. The streamed text contains no marker syntax, even when the model emits a
   malformed or unterminated marker. Strip it, and count it as dropped.
6. An answer with no citations is valid, for example "The documents do not
   say."

## 4. Stop reasons, usage, errors

Already fine, listed so they stay that way:

- `status: completed | incomplete` (+ `incomplete_details.reason`
  `max_output_tokens`) maps to FounderDesk's `end | max_tokens`. A model
  refusal should be distinguishable; a `refusal` content part is fine.
- `usage.input_tokens`, `output_tokens`, `input_tokens_details.cached_tokens`.
- `429 usage_limit_exceeded` with `retry-after` and `x-should-retry: false`;
  `503 codex_unavailable`; `400 model_not_found` / `unsupported_value`.

## 5. Model picker

FounderDesk's dropdown lists Conduix with its default model. It will read
`GET /v1/models` and take the entry with `is_default: true` (falling back to
`gpt-6-astra`), plus `efforts` to map its low/medium/high. Nothing new is
needed; please keep `is_default` stable.

## 6. Already there, no change needed

- `reasoning.effort` per request; structured output via `text.format`
  `json_schema` (FounderDesk's briefs and extraction use this through
  `generateStructured`).
- `GET /health` for the Settings page's reachability check, without calling a
  model.

## 7. Test vector

Documents:

```
[0] id "a"  "Olena: We decided to keep server-side sessions for the first release."
[1] id "b"  "Ira: Deployment is blocked by API-142.\nOlena: We need to confirm the cookie limits with the vendor."
```

Question: `What was decided, and what blocks deployment?`

Expected shape: two text spans, an `x_document_citation` on document 0
quoting `We decided to keep server-side sessions for the first release`,
and one on document 1 quoting `Deployment is blocked by API-142`. Every
`cited_text` must satisfy invariant 2.

## 8. Out of scope

PDF or image documents, web search citations, and sessions or
`previous_response_id` for document turns. FounderDesk resends history and
sends documents only on the newest turn.
