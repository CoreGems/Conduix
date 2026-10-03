"""Offline tests for image generation (step 13 / v1.2).

Route level: the fake backend from test_responses_route ("IMAGE" makes it
generate a 1x1 PNG when the turn allows image generation; "IMAGELIMIT" fails
on a usage limit). Backend level: the real SDK + backend.py against
tests/fake_app_server.py.
"""
import openai
import pytest
from openai.types import ImagesResponse
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from conduix.backend import (
    Backend,
    BlockedItem,
    ImageGenerationCall,
    TurnDone,
    base_instructions,
    discard_saved_image,
    map_notification,
)
from conduix.errors import APIError
from conduix.responses_store import ResponseStore, StoredResponse
from conduix.schema import (
    GENERATED_IMAGE_NOTE,
    image_data_url,
    image_info,
    parse_input,
    parse_tools,
    split_turn,
)
from conduix.streaming import ResponseStream
from tests.test_backend_fake_server import FAKE, app_client, stats  # noqa: F401
from tests.test_responses_route import PNG_B64, FakeBackend, client, fb, http, sse_events  # noqa: F401

EVENTS = TypeAdapter(ResponseStreamEvent)
M = "gpt-6-astra"
TOOL = {"type": "image_generation"}
PNG_URL = "data:image/png;base64," + PNG_B64
# How a generated image shows up in the fake backend's history dump.
REPLAYED = f"user:{GENERATED_IMAGE_NOTE};user:<image {len(PNG_URL)}>"


# --- Responses API ----------------------------------------------------------------

def test_image_generation_tool(client, fb):
    r = client.responses.create(model=M, input="IMAGE", tools=[TOOL])
    Response.model_validate(r.model_dump())
    image = r.output[0]
    assert image.type == "image_generation_call" and image.status == "completed"
    assert image.result == PNG_B64 and image.revised_prompt == "A dot."
    assert (image.output_format, image.size, image.background) == ("png", "1x1", "opaque")
    assert fb.threads[-1].image_generation and fb.turns[-1]["allow_image_generation"] is True


def test_no_image_generation_unless_asked(client, fb):
    r = client.responses.create(model=M, input="IMAGE")
    assert [o.type for o in r.output] == ["message"]
    assert not fb.threads[-1].image_generation and fb.turns[-1]["allow_image_generation"] is False


def test_tool_options_are_accepted(client):
    r = client.responses.create(model=M, input="IMAGE", tools=[
        {"type": "image_generation", "size": "1024x1536", "quality": "high", "model": "gpt-image-1"}])
    assert r.output[0].type == "image_generation_call"


def test_stream_events(http):
    r = http.post("/v1/responses", json={"model": M, "input": "IMAGE", "tools": [TOOL],
                                         "stream": True})
    events = sse_events(r.text)
    for e in events:
        EVENTS.validate_python(e)
    assert [e["type"] for e in events][2:7] == [
        "response.output_item.added", "response.image_generation_call.in_progress",
        "response.image_generation_call.generating", "response.image_generation_call.completed",
        "response.output_item.done",
    ]
    assert events[2]["item"]["result"] is None and events[6]["item"]["result"] == PNG_B64
    assert events[-1]["response"]["output"][0]["result"] == PNG_B64


def test_openai_stream_helper(client):
    with client.responses.stream(model=M, input="IMAGE", tools=[TOOL]) as s:
        final = s.get_final_response()
    assert final.output[0].type == "image_generation_call" and final.output[0].result == PNG_B64


def test_failed_image_is_a_failed_item(client):
    r = client.responses.create(model=M, input="IMAGELIMIT", tools=[TOOL])
    Response.model_validate(r.model_dump())
    assert r.status == "completed"
    assert (r.output[0].type, r.output[0].status, r.output[0].result) == (
        "image_generation_call", "failed", None)


def test_with_session_id_is_400(client, http):
    sid = http.post("/v1/sessions", json={}).json()["session_id"]
    with pytest.raises(openai.BadRequestError) as ei:
        client.responses.create(model=M, input="IMAGE", tools=[TOOL], extra_body={"session_id": sid})
    assert ei.value.body["param"] == "tools"


def test_chat_completions_has_no_image_generation(client):
    with pytest.raises(openai.BadRequestError) as ei:
        client.chat.completions.create(model=M, tools=[TOOL],
                                       messages=[{"role": "user", "content": "IMAGE"}])
    assert ei.value.body["param"] == "tools[0].type"


# --- multi-turn: the model sees the image it generated ---------------------------------

def test_same_tools_continue_the_thread(client, fb):
    r1 = client.responses.create(model=M, input="IMAGE", tools=[TOOL])
    r2 = client.responses.create(model=M, input="B", tools=[TOOL], previous_response_id=r1.id)
    assert len(fb.threads) == 1  # the image is already in the thread: nothing to replay
    assert r2.output_text.startswith(f"ctx=user:IMAGE;image:{r1.output[0].id};")


def test_previous_response_rebuild_replays_the_image(client, fb):
    r1 = client.responses.create(model=M, input="IMAGE", tools=[TOOL])
    r2 = client.responses.create(model=M, input="B", previous_response_id=r1.id)  # no tool now
    assert len(fb.threads) == 2
    assert r2.output_text == f"ctx=user:IMAGE;{REPLAYED};assistant:{r1.output_text};user:B"


def test_output_items_resent_as_history(client):
    r1 = client.responses.create(model=M, input="IMAGE", tools=[TOOL], store=False)
    r2 = client.responses.create(model=M, input=[
        {"role": "user", "content": "IMAGE"},
        *[o.model_dump(exclude_none=True) for o in r1.output],
        {"role": "user", "content": "B"},
    ])
    assert r2.output_text == f"ctx=user:IMAGE;{REPLAYED};assistant:{r1.output_text};user:B"


def test_image_named_by_id(client):
    r1 = client.responses.create(model=M, input="IMAGE", tools=[TOOL])
    r2 = client.responses.create(model=M, tools=[TOOL], input=[
        {"role": "user", "content": "B"},
        {"type": "image_generation_call", "id": r1.output[0].id},
    ])
    assert r2.output_text == f"ctx=user:B;{REPLAYED}"  # both are this turn's input


def test_unknown_image_id_is_400(client):
    with pytest.raises(openai.BadRequestError) as ei:
        client.responses.create(model=M, input=[
            {"role": "user", "content": "B"}, {"type": "image_generation_call", "id": "ig_nope"}])
    assert ei.value.code == "image_not_found"


def test_image_from_an_unstored_response_can_not_be_named(client):
    r1 = client.responses.create(model=M, input="IMAGE", tools=[TOOL], store=False)
    with pytest.raises(openai.BadRequestError):
        client.responses.create(model=M, input=[
            {"role": "user", "content": "B"},
            {"type": "image_generation_call", "id": r1.output[0].id}])


# --- /v1/images/generations -----------------------------------------------------------

def test_images_generate(client, fb):
    r = client.images.generate(model="gpt-image-1", prompt="IMAGE")
    ImagesResponse.model_validate(r.model_dump())
    assert r.data[0].b64_json == PNG_B64 and r.data[0].revised_prompt == "A dot."
    assert (r.output_format, r.size, r.background) == ("png", "1x1", "opaque")
    thread = fb.threads[-1]
    assert thread.image_generation and "exactly one image" in thread.instructions
    assert thread.interrupted  # the turn was stopped once the image arrived
    assert fb.manager.list() == []


def test_images_generate_drives_a_plan_model_when_named(client, fb, http):
    client.images.generate(model="gpt-5.5", prompt="IMAGE")
    assert fb.turns[-1]["model"] == "gpt-5.5"
    client.images.generate(prompt="IMAGE")  # no model, or an image model: the default
    assert fb.turns[-1]["model"] is None


def test_images_generate_passes_size_and_background_as_a_hint(client, fb):
    client.images.generate(prompt="IMAGE of a cat", size="1536x1024", background="transparent")
    assert fb.threads[-1].items[0] == (
        "user", "IMAGE of a cat\n\n(Requested: size 1536x1024, transparent background.)")


@pytest.mark.parametrize("extra, param", [
    ({"n": 2}, "n"),
    ({"response_format": "url"}, "response_format"),
    ({"stream": True}, "stream"),
    ({"prompt": "  "}, "prompt"),
])
def test_images_generate_bad_requests(http, extra, param):
    r = http.post("/v1/images/generations", json={"prompt": "IMAGE", **extra})
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_images_generate_without_an_image_is_a_502_that_is_not_retried(http):
    r = http.post("/v1/images/generations", json={"prompt": "just chat"})
    assert r.status_code == 502 and r.headers["x-should-retry"] == "false"
    err = r.json()["error"]
    assert err["code"] == "no_image_generated" and "ctx=user:just chat" in err["message"]


def test_images_generate_usage_limit_is_a_429_with_the_reset_time(client, http):
    with pytest.raises(openai.RateLimitError) as ei:
        client.images.generate(prompt="IMAGELIMIT")
    assert "usage limit reached while generating an image (resets at " in ei.value.message
    r = http.post("/v1/images/generations", json={"prompt": "IMAGELIMIT"})
    assert r.headers["x-should-retry"] == "false" and int(r.headers["retry-after"]) > 3000


def test_images_generate_plan_quota_is_a_429(client):
    with pytest.raises(openai.RateLimitError) as ei:
        client.images.generate(prompt="QUOTA")
    assert ei.value.code == "usage_limit_exceeded"


# --- units -----------------------------------------------------------------------------

def test_parse_tools():
    assert parse_tools([TOOL]) == ([], False, True)
    assert parse_tools([TOOL], "none") == ([], False, False)
    with pytest.raises(APIError) as ei:
        parse_tools([{"type": "code_interpreter"}])
    assert "image_generation" in ei.value.message


def test_image_info():
    assert image_info(PNG_B64) == ("png", "1x1")
    assert image_info("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEB") == ("jpeg", None)
    assert image_info("UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoB") == ("webp", None)
    assert image_info("not base64!") == ("png", None)
    assert image_data_url(PNG_B64) == PNG_URL


def test_image_generation_call_items_in_input():
    call = {"type": "image_generation_call", "id": "ig_1", "status": "completed", "result": PNG_B64}
    [msg] = parse_input([call])
    assert msg.generated and msg.role == "user" and msg.parts[0] == GENERATED_IMAGE_NOTE
    assert msg.parts[1].url == PNG_URL
    # A failed call copied from an earlier output has no image: dropped.
    assert parse_input([{"type": "image_generation_call", "id": "ig_2", "status": "failed"}]) == []
    with pytest.raises(APIError) as ei:
        parse_input([{**call, "result": "@@@"}])
    assert ei.value.code == "invalid_base64_image"
    found = parse_input([{"type": "image_generation_call", "id": "ig_1"}],
                        {"ig_1": PNG_URL}.get)
    assert found[0].parts[1].url == PNG_URL
    # The empty message that closes an image response is dropped too.
    empty = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}
    assert parse_input([call, empty]) == [msg]


def test_split_turn_keeps_a_generated_image_in_the_history():
    user = lambda text: {"role": "user", "content": text}  # noqa: E731
    call = {"type": "image_generation_call", "id": "ig_1", "result": PNG_B64}
    # [draw, image, change it]: the image ends the first user turn
    history, new = split_turn(parse_input([user("draw"), call, user("change it")]))
    assert [m.parts[0] for m in history] == ["draw", GENERATED_IMAGE_NOTE]
    assert [m.parts[0] for m in new] == ["change it"]
    # [change it, image]: OpenAI's "edit this one" form; both are the new turn
    history, new = split_turn(parse_input([user("draw"), {"role": "assistant", "content": "ok"},
                                           user("change it"), call]))
    assert len(history) == 2 and [m.parts[0] for m in new] == ["change it", GENERATED_IMAGE_NOTE]
    with pytest.raises(APIError):
        split_turn(parse_input([call]))  # an image alone is not a turn


def test_map_notification():
    tid = {"thread_id": "t1", "turn_id": "u1"}
    started = {"id": "exec-1", "type": "imageGeneration", "status": "in_progress", "result": ""}
    assert map_notification("item/started", {"item": started, **tid}) == [
        ImageGenerationCall("exec-1", "in_progress")]
    done = {**started, "status": "completed", "result": PNG_B64, "revised_prompt": "A dot.",
            "saved_path": "C:\\x\\generated_images\\t1\\exec-1.png", "transparent_background": False}
    assert map_notification("item/completed", {"item": done, **tid}) == [ImageGenerationCall(
        "exec-1", "completed", PNG_B64, "A dot.", False, "C:\\x\\generated_images\\t1\\exec-1.png")]
    failed = {**started, "status": "failed", "failure": {
        "type": "usageLimitExceeded", "limit_id": "image_generation", "resets_at": 1791131515}}
    assert map_notification("item/completed", {"item": failed, **tid}) == [ImageGenerationCall(
        "exec-1", "failed", limit_exceeded=True, resets_at=1791131515)]


def test_stream_closes_an_open_image_as_failed():
    rs = ResponseStream(model=M)
    rs.feed(ImageGenerationCall("ig_1", "in_progress"))
    events = rs.feed(TurnDone("interrupted"))
    assert [e["type"] for e in events] == ["response.output_item.done", "response.incomplete"]
    assert rs.response["output"][0]["status"] == "failed"


def test_store_finds_an_image_by_id():
    store = ResponseStore()
    store.add(StoredResponse("resp_1", "s", None, [], None, images={"ig_1": PNG_URL}))
    assert store.image("ig_1") == PNG_URL and store.image("ig_2") is None


def test_discard_saved_image(tmp_path):
    saved = tmp_path / "generated_images" / "thread-1" / "exec-1.png"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"x")
    discard_saved_image(str(saved))
    assert not saved.exists() and not saved.parent.exists()
    assert (tmp_path / "generated_images").exists()

    elsewhere = tmp_path / "docs" / "keep.png"  # not Codex's image folder: left alone
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"x")
    discard_saved_image(str(elsewhere))
    assert elsewhere.exists()
    discard_saved_image(None)


# --- backend + real SDK against the fake app-server ------------------------------------

@pytest.fixture
async def be():
    b = Backend(launch_args=FAKE)
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


async def test_thread_with_image_generation_keeps_the_lockdown(be):
    await be.start_thread(image_generation=True)
    [p] = (await stats(be))["thread_start_params"]
    assert p["config"]["features.image_generation"] is True
    assert p["config"]["features.shell_tool"] is False and p["config"]["web_search"] == "disabled"
    assert p["approvalPolicy"] == "never" and p["sandbox"] == "read-only"
    assert p["baseInstructions"] == base_instructions(image_generation=True)
    assert "You can generate images." in p["baseInstructions"]


async def test_image_events_and_codex_copy_on_disk_is_removed(be):
    from tests.fake_app_server import GENERATED_IMAGES

    thread = await be.start_thread(image_generation=True)
    events = [ev async for ev in be.run_turn(thread, ["IMAGE"], allow_image_generation=True)]
    started, done = [e for e in events if isinstance(e, ImageGenerationCall)]
    assert (started.status, done.status, done.result) == ("in_progress", "completed", PNG_B64)
    assert done.revised_prompt == "A dot." and done.transparent_background is False
    assert done.saved_path.endswith(".png") and not (GENERATED_IMAGES / thread.id).exists()
    assert events[-1].status == "completed" and be.blocked_items == {}


async def test_image_limit_failure(be):
    thread = await be.start_thread(image_generation=True)
    events = [ev async for ev in be.run_turn(thread, ["IMAGELIMIT"], allow_image_generation=True)]
    failed = [e for e in events if isinstance(e, ImageGenerationCall)][-1]
    assert (failed.status, failed.result, failed.limit_exceeded, failed.resets_at) == (
        "failed", "", True, 1791131515)


async def test_image_generation_is_blocked_unless_the_turn_allows_it(be):
    from tests.fake_app_server import GENERATED_IMAGES

    thread = await be.start_thread(image_generation=True)
    events = [ev async for ev in be.run_turn(thread, ["IMAGE"])]
    assert not any(isinstance(e, (ImageGenerationCall, BlockedItem)) for e in events)
    assert be.blocked_items == {"imageGeneration": 1}
    assert not (GENERATED_IMAGES / thread.id).exists()  # removed all the same


async def test_no_image_item_on_a_plain_thread(be):
    thread = await be.start_thread()
    events = [ev async for ev in be.run_turn(thread, ["IMAGE"], allow_image_generation=True)]
    assert not any(isinstance(e, ImageGenerationCall) for e in events)


def test_full_stack_offline(app_client):
    client, http = app_client
    r = client.responses.create(model=M, input="IMAGE", tools=[TOOL])
    assert r.output[0].type == "image_generation_call" and r.output[0].result == PNG_B64
    r2 = client.responses.create(model=M, input="B", previous_response_id=r.id)  # rebuilt thread
    assert f"user:{GENERATED_IMAGE_NOTE};user:<image {len(PNG_URL)}>" in r2.output_text

    image = client.images.generate(model="gpt-image-1", prompt="IMAGE")
    assert image.data[0].b64_json == PNG_B64 and image.size == "1x1"
    assert http.get("/health").json()["blocked_agent_items"] == {}


def test_full_stack_offline_usage_limit(app_client):
    """The limit failure, in the shape the Codex binary serializes it
    (`failure: {type: "usageLimitExceeded", limitId, resetsAt}`), through the
    real SDK, backend.py and both routes."""
    client, http = app_client
    with pytest.raises(openai.RateLimitError) as ei:
        client.images.generate(prompt="IMAGELIMIT")
    assert ei.value.status_code == 429 and ei.value.code == "usage_limit_exceeded"
    assert ei.value.body["message"] == ("ChatGPT plan usage limit reached while generating an "
                                        "image (resets at 2026-10-04 16:31 UTC)")
    assert ei.value.response.headers["x-should-retry"] == "false"
    assert "retry-after" in ei.value.response.headers

    # Responses API: the item fails, the turn carries on and the response completes.
    r = client.responses.create(model=M, input="IMAGELIMIT", tools=[TOOL])
    Response.model_validate(r.model_dump())
    assert r.status == "completed" and [o.type for o in r.output] == [
        "image_generation_call", "message"]
    assert (r.output[0].status, r.output[0].result) == ("failed", None)

    events = sse_events(http.post("/v1/responses", json={
        "model": M, "input": "IMAGELIMIT", "tools": [TOOL], "stream": True}).text)
    for e in events:
        EVENTS.validate_python(e)
    kinds = [e["type"] for e in events]
    assert "response.image_generation_call.completed" not in kinds
    assert kinds[-1] == "response.completed"
    [done] = [e for e in events if e["type"] == "response.output_item.done"
              and e["item"]["type"] == "image_generation_call"]
    assert done["item"]["status"] == "failed"

    # A failed image leaves nothing to replay: the next turn sees no image.
    r2 = client.responses.create(model=M, input="B", previous_response_id=r.id)
    assert "<image" not in r2.output_text and r2.output_text.endswith("user:B")
    assert http.get("/health").json()["blocked_agent_items"] == {}
