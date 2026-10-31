"""Live check of the recipes in docs/conduix/RAG_INTEGRATION.md.

Runs three requests at low effort against a running Conduix server:

  1. documents as input_text + a json_schema answer with exact quotes, through
     the openai SDK's parse helper; every quote is verified as a substring
  2. an input_file part, which must be a 400
  3. the same question streamed, with inline [n] markers

Usage (server up on :8766, or CONDUIX_BASE_URL set):

    conda run -n conduix --no-capture-output python scripts/rag_live_check.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

from openai import BadRequestError, OpenAI
from pydantic import BaseModel

BASE_URL = os.environ.get("CONDUIX_BASE_URL", "http://127.0.0.1:8766/v1")
MODEL = os.environ.get("CONDUIX_CHECK_MODEL", "gpt-6-astra")

DOCS = [
    {"id": "ops-2026-03", "title": "Ops update, March 2026", "context": "BACKGROUND, dated 2026-03-14",
     "text": "The warehouse moved to Tallinn in March. The lease runs to 2029."},
    {"id": "hr-2026-09", "title": "HR note, September 2026", "context": "dated 2026-09-02",
     "text": "Headcount is 37 as of September, up from 31 in June."},
]
QUESTION = "Where is the warehouse and how many people work there?"


class Citation(BaseModel):
    document_index: int
    quote: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]


def doc_parts(docs: list[dict]) -> list[dict]:
    return [{"type": "input_text",
             "text": f'<document index="{i}" id="{d["id"]}" title="{d["title"]}" '
                     f'context="{d["context"]}">\n{d["text"]}\n</document>'}
            for i, d in enumerate(docs)]


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key="not-used")
    failures = 0

    # 1. structured answer with verified quotes
    t = time.time()
    r = client.responses.parse(
        model=MODEL, reasoning={"effort": "low"},
        instructions="Answer only from the documents. Quote each supporting passage exactly as written.",
        input=[{"role": "user", "content": doc_parts(DOCS) + [{"type": "input_text", "text": QUESTION}]}],
        text_format=Answer,
    )
    a = r.output_parsed
    print(f"[1] structured: {time.time() - t:.1f}s, {r.usage.input_tokens} in / {r.usage.output_tokens} out")
    print(f"    answer: {a.answer}")
    if not a.citations:
        print("    FAIL: no citations")
        failures += 1
    for c in a.citations:
        ok = 0 <= c.document_index < len(DOCS) and c.quote in DOCS[c.document_index]["text"]
        print(f"    {'ok  ' if ok else 'FAIL'} doc {c.document_index}: {c.quote!r}")
        failures += not ok

    # 2. input_file must be rejected
    try:
        client.responses.create(model=MODEL, input=[{"role": "user", "content": [
            {"type": "input_file", "filename": "a.txt", "file_data": "data:text/plain;base64,aGVsbG8="},
            {"type": "input_text", "text": "hi"}]}])
        print("[2] input_file: FAIL, accepted")
        failures += 1
    except BadRequestError as e:
        detail = e.body.get("message") if isinstance(e.body, dict) else str(e)  # the SDK unwraps `error`
        print(f"[2] input_file: ok, 400 {detail}")

    # 3. streaming with inline markers
    t, first, text = time.time(), None, ""
    instructions = ("Answer only from the documents. End each sentence that uses a document with that "
                    "document's marker, [1] for index 0, [2] for index 1, and so on. Use no other "
                    "citation format.")
    with client.responses.stream(
        model=MODEL, reasoning={"effort": "low"}, instructions=instructions,
        input=[{"role": "user", "content": doc_parts(DOCS) + [{"type": "input_text", "text": QUESTION}]}],
    ) as s:
        for ev in s:
            if ev.type == "response.output_text.delta":
                first = first or time.time() - t
                text += ev.delta
        final = s.get_final_response()
    markers = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
    in_range = all(0 < n <= len(DOCS) for n in markers)
    annotations = [c.annotations for o in final.output if o.type == "message" for c in o.content]
    print(f"[3] streaming: first token {first or 0:.1f}s, total {time.time() - t:.1f}s")
    print(f"    text: {text}")
    print(f"    {'ok  ' if markers and in_range else 'FAIL'} markers {markers}; annotations {annotations}")
    failures += not (markers and in_range)

    print("PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
