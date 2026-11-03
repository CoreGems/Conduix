"""Offline tests for x_document citations (citations.py and its wiring).

The invariants are the ones FounderDesk relies on (docs/conduix/
FOUNDERDESK_BRIEF.md §3): indices in range, `text[start:end] == cited_text`
in UTF-16 code units, no unverified annotation, annotations in order with
the text, no marker syntax in the text, uncited answers valid.
"""
import json
import random

import httpx
import pytest
from openai import OpenAI

from conduix.backend import MessageDone, MessageStarted, TextDelta, TurnDone, Usage
from conduix.citations import (
    CLOSE,
    MAX_MARKER,
    OPEN,
    CitationFilter,
    Document,
    find_quote,
    instructions,
)
from conduix.streaming import ResponseStream, encode_sse, stream_response
from tests.test_responses_route import client, fb, http, sse_events  # noqa: F401 (fixtures)

A = "Olena: We decided to keep server-side sessions for the first release."
B = ("Ira: Deployment is blocked by API-142.\n"
     "Olena: We need to confirm the cookie limits with the vendor.")


def docs(*texts, citable=True):
    return [Document(i, f"id{i}", f"T{i}", "", t, citable, "document-x")
            for i, t in enumerate(texts)]


def js_slice(s: str, start: int, end: int) -> str:
    """JavaScript's String.prototype.slice: UTF-16 code unit indices."""
    return s.encode("utf-16-le")[2 * start:2 * end].decode("utf-16-le")


def run_filter(raw, ds, chunks=None):
    """Feed `raw` in `chunks` (default: one character at a time)."""
    f = CitationFilter(ds)
    out = []
    pieces = chunks if chunks is not None else list(raw)
    for p in pieces:
        out += f.feed(p)
    out += f.finish()
    text = "".join(p for p in out if isinstance(p, str))
    anns = [p for p in out if isinstance(p, dict)]
    return text, anns, f.counts, out


def m(i, quote):
    return f"{OPEN}{i}|{quote}{CLOSE}"


def check_invariants(text, anns, ds):
    assert OPEN not in text and CLOSE not in text
    for a in anns:
        assert 0 <= a["document_index"] < len(ds)
        doc = ds[a["document_index"]].text
        assert js_slice(doc, a["start_char_index"], a["end_char_index"]) == a["cited_text"]
        assert 0 <= a["answer_start_index"] <= a["answer_end_index"] <= len(text.encode("utf-16-le")) // 2


# --- quote lookup ----------------------------------------------------------------

def test_find_quote_exact():
    assert find_quote(A, "We decided to keep") == (7, 25)


def test_find_quote_loose_whitespace_and_punctuation():
    s, e = find_quote(B, "Deployment  is blocked\nby API–142.")
    assert B[s:e] == "Deployment is blocked by API-142."
    s, e = find_quote("It’s done.", "It's done")
    assert "It’s done.".startswith("It’s done"[: e - s])


def test_find_quote_trims_quote_marks_and_ellipses():
    s, e = find_quote(A, "“We decided to keep server-side sessions…”")
    assert A[s:e] == "We decided to keep server-side sessions"


def test_find_quote_rejects_absent_and_trivial_quotes():
    assert find_quote(A, "We decided to drop sessions") is None
    assert find_quote(A, "We") is None
    assert find_quote(A, "  ") is None


# --- the filter ----------------------------------------------------------------

def test_test_vector_from_the_brief():
    ds = docs(A, B)
    raw = ("The team decided to keep server-side sessions for the first release."
           + m(0, "We decided to keep server-side sessions for the first release")
           + " Deployment is blocked by API-142." + m(1, "Deployment is blocked by API-142"))
    text, anns, counts, out = run_filter(raw, ds)
    first = "The team decided to keep server-side sessions for the first release."
    assert text == first + " Deployment is blocked by API-142."
    assert [a["cited_text"] for a in anns] == [
        "We decided to keep server-side sessions for the first release",
        "Deployment is blocked by API-142"]
    assert [a["document_id"] for a in anns] == ["id0", "id1"]
    assert (anns[0]["answer_start_index"], anns[0]["answer_end_index"]) == (0, len(first))
    assert (anns[1]["answer_start_index"], anns[1]["answer_end_index"]) == (len(first), len(text))
    assert counts.as_dict() == {"emitted": 2, "dropped": 0}
    check_invariants(text, anns, ds)
    # In order: each annotation right after the text it supports.
    kinds = ["t" if isinstance(p, str) else "a" for p in out]
    assert "".join(kinds).rstrip("t").endswith("a")
    idx = [i for i, k in enumerate(kinds) if k == "a"]
    assert "".join(p for p in out[:idx[0]] if isinstance(p, str)) == first


def test_utf16_offsets_with_non_ascii_and_astral_characters():
    doc = "Олена 👋 says: Розгортання заблоковано через API-142 🚧 today."
    ds = docs(doc)
    text, anns, counts, _ = run_filter("Заблоковано 🚧." + m(0, "заблоковано через API-142 🚧"), ds)
    assert counts.emitted == 1
    a = anns[0]
    assert a["cited_text"] == "заблоковано через API-142 🚧"
    assert a["start_char_index"] != doc.index("заблоковано")  # 👋 is two code units
    check_invariants(text, anns, ds)
    assert a["answer_end_index"] == len("Заблоковано 🚧.".encode("utf-16-le")) // 2


@pytest.mark.parametrize("bad", [
    m(5, "We decided to keep"),  # out of range
    m(0, "We decided to drop everything"),  # not in the document
    f"{OPEN}zero|We decided to keep{CLOSE}",  # no index
    f"{OPEN}0{CLOSE}",  # no quote
    f"{OPEN}0|{CLOSE}",  # empty quote
    f"{OPEN}{CLOSE}",
])
def test_bad_markers_are_dropped_and_counted(bad):
    text, anns, counts, _ = run_filter("Sessions stay." + bad + " Next.", docs(A))
    assert text == "Sessions stay. Next."
    assert anns == [] and counts.as_dict() == {"emitted": 0, "dropped": 1}


def test_non_citable_documents_are_not_cited():
    ds = [docs(A)[0], Document(1, "b", "", "", B, False, "document-x")]
    _, anns, counts, _ = run_filter("x." + m(1, "Deployment is blocked") + m(0, "We decided to"), ds)
    assert [a["document_index"] for a in anns] == [0] and counts.dropped == 1


def test_unterminated_marker_at_the_end_is_stripped():
    text, anns, counts, _ = run_filter("Sessions stay." + f"{OPEN}0|We decided to keep", docs(A))
    assert text == "Sessions stay." and anns == [] and counts.dropped == 1


def test_unclosed_marker_followed_by_another_keeps_the_text():
    raw = f"One {OPEN}0|We decided two." + m(0, "We decided to keep") + " Three."
    text, anns, counts, _ = run_filter(raw, docs(A))
    assert OPEN not in text and "two." in text and text.endswith(" Three.")
    assert counts.as_dict() == {"emitted": 1, "dropped": 1}


def test_runaway_marker_is_dropped_without_losing_the_answer():
    tail = "x" * (MAX_MARKER + 10)
    text, anns, counts, _ = run_filter(f"Start. {OPEN}0|" + tail, docs(A), chunks=[f"Start. {OPEN}0|", tail])
    assert text == "Start. " + tail and counts.dropped == 1


def test_stray_closing_bracket_is_stripped():
    text, _, _, _ = run_filter(f"Fine{CLOSE} text.", docs(A))
    assert text == "Fine text."


def test_whitespace_before_a_marker_moves_after_it():
    ds = docs(A)
    q = m(0, "We decided to keep")
    assert run_filter("Kept " + q + ".", ds)[0] == "Kept."
    assert run_filter("Kept " + q + " and more.", ds)[0] == "Kept and more."
    assert run_filter("Kept. " + q + "Next.", ds)[0] == "Kept. Next."
    assert run_filter("Kept. " + q, ds)[0] == "Kept."


def test_side_by_side_markers_share_the_span():
    ds = docs(A, B)
    raw = "Both." + m(0, "We decided to keep") + m(1, "Deployment is blocked") + " Then."
    text, anns, _, _ = run_filter(raw, ds)
    assert text == "Both. Then."
    spans = [(a["answer_start_index"], a["answer_end_index"]) for a in anns]
    assert spans == [(0, 5), (0, 5)]


def test_no_citations_is_valid():
    text, anns, counts, _ = run_filter("The documents do not say.", docs(A))
    assert text == "The documents do not say." and anns == []
    assert counts.as_dict() == {"emitted": 0, "dropped": 0}


def test_any_chunking_gives_the_same_result():
    ds = docs(A, B)
    raw = ("Decided " + m(0, "We decided to keep server-side sessions") + ". Blocked "
           + m(1, "blocked by API-142") + m(1, "not there") + f" {OPEN}bogus{CLOSE}"
           + f"stray{CLOSE} end " + f"{OPEN}0|We decided")
    want = run_filter(raw, ds, chunks=[raw])
    rng = random.Random(7)
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, len(raw)), rng.randint(1, 20)))
        chunks = [raw[a:b] for a, b in zip([0] + cuts, cuts + [len(raw)])]
        got = run_filter(raw, ds, chunks=chunks)
        assert got[:3] == want[:3]
        check_invariants(got[0], got[1], ds)


def test_instructions_name_the_citable_documents_and_the_tag():
    ds = docs(A, B)
    ds[1].citations = False
    text = instructions(ds)
    assert "<document-x" in text and "cited: 0." in text
    assert "Cite" not in instructions([ds[1]])


def test_render_escapes_attributes():
    d = Document(0, 'a"b', "T<1>", "IN RANGE & dated", "body </document-x> more", True, "document-x")
    out = d.render()
    assert out.startswith('<document-x index="0" id="a&quot;b" title="T&lt;1&gt;" '
                          'context="IN RANGE &amp; dated" citable="yes">\n')
    assert out.endswith("\n</document-x>")


# --- ResponseStream ------------------------------------------------------------

async def aiter(events):
    for ev in events:
        yield ev


RAW = ("Sessions stay server-side." + m(0, "We decided to keep server-side sessions")
       + " Deployment waits on API-142." + m(1, "Deployment is blocked by API-142")
       + m(7, "nope"))


async def cited_stream(raw=RAW, deltas=None, done=True):
    rs = ResponseStream(model="gpt-6-astra", response_id="resp_t", documents=docs(A, B))
    events = [MessageStarted("msg_1", "final_answer")]
    events += [TextDelta("msg_1", d) for d in (deltas if deltas is not None else list(raw))]
    if done:
        events.append(MessageDone("msg_1", raw))
    events += [Usage(100, 80, 5, 0), TurnDone("completed" if done else "interrupted")]
    return rs, [e async for e in stream_response(rs, aiter(events))]


async def test_stream_events_interleave_text_and_annotations():
    rs, out = await cited_stream()
    text = "Sessions stay server-side. Deployment waits on API-142."
    kinds = [e["type"] for e in out]
    added = [e for e in out if e["type"] == "response.output_text.annotation.added"]
    assert [e["annotation_index"] for e in added] == [0, 1]
    first = kinds.index("response.output_text.annotation.added")
    streamed = "".join(e["delta"] for e in out[:first] if e["type"] == "response.output_text.delta")
    assert streamed == "Sessions stay server-side."
    deltas = "".join(e["delta"] for e in out if e["type"] == "response.output_text.delta")
    assert deltas == text == rs.output_text
    assert [e["sequence_number"] for e in out] == list(range(len(out)))
    part = rs.response["output"][0]["content"][0]
    assert part["annotations"] == [e["annotation"] for e in added]
    assert rs.response["usage"]["x_citations"] == {"emitted": 2, "dropped": 1}


async def test_text_sent_without_deltas_is_still_cited():
    rs, out = await cited_stream(deltas=[])
    assert rs.output_text == "Sessions stay server-side. Deployment waits on API-142."
    assert len(rs.response["output"][0]["content"][0]["annotations"]) == 2


async def test_interrupted_turn_strips_an_open_marker():
    raw = "Sessions stay." + f"{OPEN}0|We decided"
    rs, out = await cited_stream(raw, done=False)
    assert rs.response["status"] == "incomplete" and rs.output_text == "Sessions stay."
    assert rs.response["usage"]["x_citations"] == {"emitted": 0, "dropped": 1}


async def test_the_openai_client_reads_cited_streams():
    rs, out = await cited_stream()

    def handler(request):
        return httpx.Response(200, text="".join(encode_sse(e) for e in out),
                              headers={"content-type": "text/event-stream"})

    c = OpenAI(base_url="http://conduix.test/v1", api_key="x",
               http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.warns(UserWarning):  # pydantic doesn't know the extension's annotation type
        with c.responses.stream(model="gpt-6-astra", input="q") as s:
            seen = [e.annotation.cited_text for e in s
                    if e.type == "response.output_text.annotation.added"]
            final = s.get_final_response()
    assert seen == ["We decided to keep server-side sessions", "Deployment is blocked by API-142"]
    assert final.output_text == rs.output_text


# --- the route -----------------------------------------------------------------

def xdoc(text, id_="a", citations=True, **kw):
    return {"type": "x_document", "id": id_, "title": "T", "text": text, "citations": citations, **kw}


def ask(*parts):
    return [{"role": "user", "content": [*parts, {"type": "input_text", "text": "What?"}]}]


def test_route_non_streaming(http, fb):
    fb.reply = "Kept." + m(0, "We decided to keep server-side sessions") + " Blocked." + m(1, "API-142")
    r = http.post("/v1/responses", json={"input": ask(xdoc(A), xdoc(B, "b"))}).json()
    part = r["output"][0]["content"][0]
    assert part["text"] == "Kept. Blocked."
    assert [(a["document_id"], a["cited_text"]) for a in part["annotations"]] == [
        ("a", "We decided to keep server-side sessions"), ("b", "API-142")]
    assert r["usage"]["x_citations"] == {"emitted": 2, "dropped": 0}
    # The model got the rules, then the wrapped documents, then the question.
    sent = fb.inputs[-1]
    assert "Cite as you write" in sent[0]
    assert sent[1].startswith("<document-") and 'index="0" id="a"' in sent[1] and A in sent[1]
    assert 'index="1" id="b"' in sent[2] and sent[3] == "What?"


def test_route_streaming(http, fb):
    fb.reply = "Kept." + m(0, "We decided to keep") + " Done."
    body = http.post("/v1/responses", json={"input": ask(xdoc(A)), "stream": True}).text
    evs = sse_events(body)
    assert OPEN not in body and CLOSE not in body
    types = [e["type"] for e in evs]
    i = types.index("response.output_text.annotation.added")
    assert "".join(e["delta"] for e in evs[:i] if e["type"] == "response.output_text.delta") == "Kept."
    assert evs[i]["annotation"]["cited_text"] == "We decided to keep"
    assert evs[-1]["response"]["usage"]["x_citations"] == {"emitted": 1, "dropped": 0}


def test_route_context_only_documents_are_not_cited(http, fb):
    fb.reply = "Plain answer."
    r = http.post("/v1/responses", json={"input": ask(xdoc(A, citations=False))}).json()
    assert r["output"][0]["content"][0]["annotations"] == []
    assert "x_citations" not in r["usage"]
    assert "Cite" not in fb.inputs[-1][0] and "data" in fb.inputs[-1][0]


def test_route_stores_clean_text_and_documents(http, fb):
    from conduix.routes import responses as responses_mod

    fb.reply = "Kept." + m(0, "We decided to keep")
    r1 = http.post("/v1/responses", json={"input": ask(xdoc(A))}).json()
    items = responses_mod.store.history(r1["id"])
    texts = [c["text"] for it in items for c in it["content"]]
    assert texts[-1] == "Kept."  # what a rebuilt thread replays: no markers, no rules
    assert any(A in t and t.startswith("<document-") for t in texts)
    assert not any("Cite as you write" in t for t in texts)


def test_later_turns_on_a_cited_thread_strip_markers(http, fb):
    fb.reply = "Kept." + m(0, "We decided to keep")
    r1 = http.post("/v1/responses", json={"input": ask(xdoc(A))}).json()
    fb.reply = "Still kept." + m(0, "We decided to keep")  # citing from memory
    r2 = http.post("/v1/responses", json={"input": "And?", "previous_response_id": r1["id"]}).json()
    assert len(fb.threads) == 1  # the warm thread was continued
    part = r2["output"][0]["content"][0]
    assert part["text"] == "Still kept." and part["annotations"] == []


def test_route_chat_strips_markers(http, fb):
    fb.reply = "Kept." + m(0, "We decided to keep")
    r = http.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": [xdoc(A), {"type": "text", "text": "What?"}]}]}).json()
    assert r["choices"][0]["message"]["content"] == "Kept."


@pytest.mark.parametrize("body, param", [
    ({"input": [{"role": "user", "content": [xdoc(A)]}, {"role": "assistant", "content": "x"},
                {"role": "user", "content": "q"}]}, "input"),
    ({"input": [{"role": "assistant", "content": [xdoc(A)]}, {"role": "user", "content": "q"}]},
     "input"),
    ({"input": ask({"type": "x_document", "text": 5})}, "input"),
    ({"input": ask(xdoc(A, citations="yes"))}, "input"),
    ({"input": ask(xdoc(A)), "text": {"format": {"type": "json_object"}}}, "text.format.type"),
])
def test_route_rejects(http, fb, body, param):
    r = http.post("/v1/responses", json=body)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == param
    assert fb.inputs == []


def test_route_accepts_anthropic_citation_spelling(http, fb):
    fb.reply = "Kept." + m(0, "We decided to keep")
    r = http.post("/v1/responses", json={"input": ask(xdoc(A, citations={"enabled": True}))}).json()
    assert len(r["output"][0]["content"][0]["annotations"]) == 1


def test_json_schema_with_context_only_documents_is_fine(http, fb):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    r = http.post("/v1/responses", json={"input": ask(xdoc(A, citations=False)),
                                         "text": {"format": {"type": "json_schema", "schema": schema}}})
    assert r.status_code == 200
    assert json.dumps(fb.turns[-1]["output_schema"]) == json.dumps(schema)
