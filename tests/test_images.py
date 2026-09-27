"""Offline tests for image input (step 10): validation, both endpoints, history,
and the response store's byte budget."""
import base64
import struct
import zlib

import openai
import pytest

from conduix import schema
from conduix.backend import ImagePart, _to_run_input
from conduix.errors import APIError
from conduix.responses_store import ResponseStore, StoredResponse
from conduix.schema import parse_input
from tests.test_responses_route import FakeBackend, client, fb, http  # noqa: F401


def png_1x1() -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


PNG = "data:image/png;base64," + base64.b64encode(png_1x1()).decode()


def responses_image(url):
    return [{"role": "user", "content": [
        {"type": "input_text", "text": "What is this?"},
        {"type": "input_image", "image_url": url},
    ]}]


def chat_image(url):
    return [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
    ]}]


# --- validation ------------------------------------------------------------------

def test_valid_data_url_is_kept_in_order():
    [msg] = parse_input(responses_image(PNG))
    assert msg.parts == ["What is this?", ImagePart(PNG)]


def test_chat_string_form_is_accepted():
    [msg] = parse_input([{"role": "user", "content": [{"type": "image_url", "image_url": PNG}]}])
    assert msg.parts == [ImagePart(PNG)]


@pytest.mark.parametrize("url, code", [
    ("https://example.com/cat.png", "invalid_image_url"),
    ("http://example.com/cat.png", "invalid_image_url"),
    ("cat.png", "invalid_image_url"),
    ("data:image/png,rawbytes", "invalid_image_url"),  # not base64
    ("data:image/svg+xml;base64,PHN2Zz4=", "unsupported_image_media_type"),
    ("data:image/png;base64,***", "invalid_base64_image"),
    ("data:image/png;base64,", "empty_image_file"),
    (None, "invalid_image_url"),
])
def test_bad_images_are_rejected(url, code):
    with pytest.raises(APIError) as ei:
        parse_input(responses_image(url))
    assert ei.value.status == 400 and ei.value.code == code and ei.value.param == "input"


def test_remote_url_message_says_what_to_do():
    with pytest.raises(APIError) as ei:
        parse_input(responses_image("https://example.com/cat.png"))
    assert "base64 data URL" in ei.value.message


def test_too_large(monkeypatch):
    monkeypatch.setattr(schema, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(APIError) as ei:
        parse_input(responses_image(PNG))
    assert ei.value.code == "image_too_large"


def test_file_id_is_rejected():
    with pytest.raises(APIError):
        parse_input([{"role": "user", "content": [{"type": "input_image", "file_id": "file_1"}]}])


def test_only_user_messages_carry_images():
    with pytest.raises(APIError):
        parse_input([{"role": "assistant", "content": [{"type": "input_image", "image_url": PNG}]},
                     {"role": "user", "content": "hi"}])


def test_to_item_and_sdk_input():
    [msg] = parse_input(responses_image(PNG))
    assert msg.to_item()["content"] == [
        {"type": "input_text", "text": "What is this?"},
        {"type": "input_image", "image_url": PNG},
    ]
    text, image = _to_run_input(msg.parts)
    assert text.text == "What is this?" and image.url == PNG


# --- routes ----------------------------------------------------------------------

def test_responses_image_reaches_the_turn(client, fb):
    r = client.responses.create(model="gpt-6-astra", input=responses_image(PNG))
    assert r.output_text == f"ctx=user:What is this?;user:<image {len(PNG)}>"


def test_chat_image_reaches_the_turn(client, fb):
    c = client.chat.completions.create(model="gpt-6-astra", messages=chat_image(PNG))
    assert c.choices[0].message.content.endswith(f"user:<image {len(PNG)}>")


def test_image_in_history_is_injected(client, fb):
    history = responses_image(PNG) + [{"role": "assistant", "content": "A dot."},
                                      {"role": "user", "content": "What color?"}]
    client.responses.create(model="gpt-6-astra", input=history)
    assert fb.threads[-1].items[:3] == [
        ("user", "What is this?"), ("user", f"<image {len(PNG)}>"), ("assistant", "A dot."),
    ]


def test_image_survives_a_previous_response_rebuild(client, http, fb):
    r1 = client.responses.create(model="gpt-6-astra", input=responses_image(PNG))
    for s in fb.manager.list():
        http.delete(f"/v1/sessions/{s.id}")
    r2 = client.responses.create(model="gpt-6-astra", input="And now?", previous_response_id=r1.id)
    assert f"user:<image {len(PNG)}>" in r2.output_text


def test_text_only_model_rejects_images(client):
    with pytest.raises(openai.BadRequestError) as ei:
        client.responses.create(model="gpt-5.5", input=responses_image(PNG))
    assert ei.value.body["param"] == "model"


def test_chat_remote_url_param_is_messages(client):
    with pytest.raises(openai.BadRequestError) as ei:
        client.chat.completions.create(model="gpt-6-astra",
                                       messages=chat_image("https://example.com/x.png"))
    assert ei.value.body["param"] == "messages" and ei.value.code == "invalid_image_url"


# --- store budget ----------------------------------------------------------------

def _rec(rid, text):
    return StoredResponse(id=rid, session_id="s", parent_id=None, instructions=None,
                          items=[{"type": "message", "role": "user",
                                  "content": [{"type": "input_image", "image_url": text}]}])


def test_store_evicts_oldest_over_byte_budget():
    s = ResponseStore(max_bytes=250)
    for i in range(4):
        s.add(_rec(f"r{i}", "x" * 100))
    assert [r for r in ("r0", "r1", "r2", "r3") if s.get(r)] == ["r2", "r3"]
    assert s.total_bytes == 200


def test_store_keeps_newest_even_if_alone_over_budget():
    s = ResponseStore(max_bytes=50)
    s.add(_rec("big", "x" * 100))
    assert s.get("big") is not None and s.total_bytes == 100
