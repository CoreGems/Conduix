"""Offline tests for backend.py. Payloads mirror scratch/probe_*.jsonl (step 2)."""
import os

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
    map_notification,
    scrub_api_keys,
)

TID = {"thread_id": "t1", "turn_id": "u1"}


def test_user_message_echo_is_ignored():
    item = {"type": "userMessage", "id": "x", "content": [{"type": "text", "text": "hi"}]}
    assert map_notification("item/started", {"item": item, **TID}) == []
    assert map_notification("item/completed", {"item": item, **TID}) == []


def test_agent_message_lifecycle():
    started = {"id": "msg_1", "phase": "final_answer", "text": "", "type": "agentMessage"}
    assert map_notification("item/started", {"item": started, **TID}) == [
        MessageStarted("msg_1", "final_answer")
    ]
    assert map_notification(
        "item/agentMessage/delta", {"delta": "Hi", "item_id": "msg_1", **TID}
    ) == [TextDelta("msg_1", "Hi")]
    done = {**started, "text": "Hi there, friend!"}
    assert map_notification("item/completed", {"item": done, **TID}) == [
        MessageDone("msg_1", "Hi there, friend!")
    ]


def test_reasoning_lifecycle():
    item = {"id": "rs_1", "type": "reasoning", "summary": [], "content": []}
    assert map_notification("item/started", {"item": item, **TID}) == [ReasoningStarted("rs_1")]
    assert map_notification(
        "item/reasoning/summaryTextDelta",
        {"delta": "Thinking", "item_id": "rs_1", "summary_index": 0, **TID},
    ) == [ReasoningSummaryDelta("rs_1", 0, "Thinking")]
    done = {**item, "summary": ["Thinking about it"]}
    assert map_notification("item/completed", {"item": done, **TID}) == [
        ReasoningDone("rs_1", ["Thinking about it"])
    ]


def test_usage_uses_last_not_total():
    payload = {
        "token_usage": {
            "last": {"input_tokens": 4418, "cached_input_tokens": 4200, "output_tokens": 9,
                     "reasoning_output_tokens": 2, "total_tokens": 4427,
                     "cache_write_input_tokens": 118},
            "total": {"input_tokens": 9999, "cached_input_tokens": 0, "output_tokens": 99,
                      "reasoning_output_tokens": 0, "total_tokens": 10098},
            "model_context_window": 258400,
        },
        **TID,
    }
    assert map_notification("thread/tokenUsage/updated", payload) == [Usage(4418, 4200, 9, 2, 118)]


def test_turn_completed():
    turn = {"id": "u1", "status": "completed", "items": []}
    assert map_notification("turn/completed", {"turn": turn, "thread_id": "t1"}) == [
        TurnDone("completed")
    ]


def test_turn_failed_carries_error():
    err = {"message": "usage limit", "codex_error_info": "usageLimitExceeded"}
    turn = {"id": "u1", "status": "failed", "items": [], "error": err}
    assert map_notification("turn/completed", {"turn": turn, "thread_id": "t1"}) == [
        TurnDone("failed", TurnError("usage limit", "usageLimitExceeded"))
    ]


def test_error_notification_retrying_is_dropped():
    payload = {"error": {"message": "reconnecting"}, "will_retry": True, **TID}
    assert map_notification("error", payload) == []


def test_error_notification_final():
    payload = {"error": {"message": "boom", "additional_details": "d"}, "will_retry": False, **TID}
    assert map_notification("error", payload) == [TurnError("boom", None, "d")]


def test_agentic_items_are_dropped(caplog):
    item = {"id": "c1", "type": "commandExecution", "command": "dir"}
    assert map_notification("item/started", {"item": item, **TID}) == []
    assert "commandExecution" in caplog.text


def test_unknown_methods_ignored():
    assert map_notification("turn/started", {"turn": {"id": "u1"}, **TID}) == []
    assert map_notification("thread/somethingNew", {}) == []


def test_scrub_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    assert scrub_api_keys() == ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY" not in os.environ
