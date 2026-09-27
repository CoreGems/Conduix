"""Offline tests for streaming.py.

Parity is checked with the real `openai` client: our events are served over a
mock HTTP transport and parsed by `client.responses.stream()` (which rebuilds
the response from events and rejects inconsistent sequences) and by
`client.responses.create()`.
"""
import json

import httpx
import pytest
from openai import OpenAI
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from conduix.backend import (
    MessageDone,
    MessageStarted,
    ReasoningDone,
    ReasoningStarted,
    ReasoningSummaryDelta,
    TextDelta,
    TurnDone,
    TurnError,
    Usage,
)
from conduix.streaming import ResponseStream, collect_response, encode_sse, stream_response

EVENTS = TypeAdapter(ResponseStreamEvent)

# The probe's "hello" turn (scratch/probe_*.jsonl), with a reasoning item.
HELLO = [
    ReasoningStarted("rs_1"),
    ReasoningDone("rs_1", []),
    MessageStarted("msg_1", "final_answer"),
    TextDelta("msg_1", "Hi"),
    TextDelta("msg_1", " there,"),
    TextDelta("msg_1", " friend!"),
    MessageDone("msg_1", "Hi there, friend!"),
    Usage(4418, 4200, 9, 0),
    TurnDone("completed"),
]


async def aiter(events):
    for ev in events:
        yield ev


async def run(events, **kw):
    rs = ResponseStream(model="gpt-6-astra", response_id="resp_test", **kw)
    out = [e async for e in stream_response(rs, aiter(events))]
    return rs, out


def openai_client(sse_events=None, response=None) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            body = "".join(encode_sse(e) for e in sse_events)
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=response)

    return OpenAI(base_url="http://conduix.test/v1", api_key="x",
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


async def test_every_event_is_a_valid_openai_event():
    _, out = await run(HELLO)
    for e in out:
        EVENTS.validate_python(e)
    assert [e["sequence_number"] for e in out] == list(range(len(out)))
    assert out[0]["type"] == "response.created" and out[-1]["type"] == "response.completed"


async def test_event_order_for_a_message():
    _, out = await run(HELLO[2:])
    assert [e["type"] for e in out] == [
        "response.created", "response.in_progress",
        "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.delta", "response.output_text.delta",
        "response.output_text.done", "response.content_part.done", "response.output_item.done",
        "response.completed",
    ]


async def test_created_snapshot_is_not_mutated_later():
    _, out = await run(HELLO)
    assert out[0]["response"]["status"] == "in_progress"
    assert out[0]["response"]["output"] == []


async def test_openai_stream_helper_accepts_our_events():
    _, out = await run(HELLO)
    client = openai_client(sse_events=out)
    with client.responses.stream(model="gpt-6-astra", input="hi") as stream:
        deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
        final = stream.get_final_response()
    assert "".join(deltas) == "Hi there, friend!"
    assert final.output_text == "Hi there, friend!"
    assert final.usage.input_tokens_details.cached_tokens == 4200


async def test_openai_create_parses_collected_response():
    rs = ResponseStream(model="gpt-6-astra")
    resp = await collect_response(rs, aiter(HELLO))
    r = openai_client(response=resp).responses.create(model="gpt-6-astra", input="hi")
    assert r.id.startswith("resp_") and r.status == "completed"
    assert r.output_text == "Hi there, friend!"
    assert [o.type for o in r.output] == ["reasoning", "message"]
    assert r.output[1].phase == "final_answer"
    assert r.usage.total_tokens == 4427


async def test_reasoning_summary_streams():
    events = [
        ReasoningStarted("rs_1"),
        ReasoningSummaryDelta("rs_1", 0, "Thinking "),
        ReasoningSummaryDelta("rs_1", 0, "hard"),
        ReasoningDone("rs_1", ["Thinking hard", "Second part"]),  # part 1 never streamed
        MessageDone("msg_1", "42"),  # no MessageStarted / deltas at all
        TurnDone("completed"),
    ]
    rs, out = await run(events)
    for e in out:
        EVENTS.validate_python(e)
    final = Response.model_validate(rs.response)
    assert [s.text for s in final.output[0].summary] == ["Thinking hard", "Second part"]
    assert final.output_text == "42"
    # The stream helper must agree with the final response.
    with openai_client(sse_events=out).responses.stream(model="m", input="x") as s:
        got = s.get_final_response()
    assert [x.text for x in got.output[0].summary] == ["Thinking hard", "Second part"]
    assert got.output_text == "42"


async def test_failed_turn_closes_open_items():
    events = [
        MessageStarted("msg_1"),
        TextDelta("msg_1", "partial"),
        TurnError("usage limit reached"),
        TurnDone("failed"),
    ]
    rs, out = await run(events)
    for e in out:
        EVENTS.validate_python(e)
    assert out[-1]["type"] == "response.failed"
    final = Response.model_validate(rs.response)
    assert final.status == "failed"
    assert final.error.message == "usage limit reached"
    assert final.output[0].status == "incomplete" and final.output_text == "partial"


async def test_turn_error_on_turn_done_wins():
    rs, _ = await run([TurnError("earlier"), TurnDone("failed", TurnError("final"))])
    assert rs.response["error"]["message"] == "final"


async def test_interrupted_is_incomplete():
    rs, out = await run([MessageStarted("m"), TurnDone("interrupted")])
    assert out[-1]["type"] == "response.incomplete"
    assert Response.model_validate(rs.response).status == "incomplete"


async def test_stream_without_turn_done_still_terminates():
    rs, out = await run([MessageStarted("m"), TextDelta("m", "x")])
    assert out[-1]["type"] == "response.failed"
    assert rs.finished


async def test_events_after_finish_are_ignored():
    rs, out = await run(HELLO + [TextDelta("msg_1", "late")])
    assert out[-1]["type"] == "response.completed"
    assert rs.output_text == "Hi there, friend!"


async def test_request_fields_echoed():
    rs, _ = await run(HELLO, instructions="be brief", previous_response_id="resp_prev",
                      reasoning={"effort": "low", "summary": None})
    r = Response.model_validate(rs.response)
    assert r.instructions == "be brief"
    assert r.previous_response_id == "resp_prev"
    assert r.reasoning.effort == "low"


def test_encode_sse():
    frame = encode_sse({"type": "response.output_text.delta", "delta": "é"})
    assert frame.startswith("event: response.output_text.delta\ndata: ")
    assert frame.endswith("\n\n") and "é" in frame
