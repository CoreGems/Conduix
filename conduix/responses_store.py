"""In-memory store of completed responses, for `previous_response_id`.

Each record keeps only what its own turn added to the conversation (its
input messages and its output messages) plus its parent. Walking the parents
rebuilds the full model-visible history, so a response can be continued even
after its Codex thread was evicted, and branching from an older response
gets the right history rather than whatever the thread holds now.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

MAX_RESPONSES = 1000
# Images make records large (base64 data URLs, up to 20 MB each), so the
# store is also capped by size; the oldest records go first.
MAX_BYTES = 256 * 1024 * 1024


def _size(items: list[dict[str, Any]]) -> int:
    return sum(
        len(c.get("text") or c.get("image_url") or "")
        for item in items for c in item.get("content", [])
    )


@dataclass
class StoredResponse:
    id: str
    session_id: str  # the (possibly implicit) session whose thread ran it
    parent_id: str | None
    items: list[dict[str, Any]]  # raw Responses items this turn added
    instructions: str | None
    created_at: float = field(default_factory=time.time)
    size: int = 0  # approximate bytes of text + image data, set by the store


class ResponseStore:
    def __init__(self, max_responses: int = MAX_RESPONSES, max_bytes: int = MAX_BYTES) -> None:
        self._max = max_responses
        self._max_bytes = max_bytes
        self._bytes = 0
        self._data: OrderedDict[str, StoredResponse] = OrderedDict()

    def add(self, rec: StoredResponse) -> None:
        rec.size = _size(rec.items)
        old = self._data.pop(rec.id, None)
        if old is not None:
            self._bytes -= old.size
        self._data[rec.id] = rec
        self._bytes += rec.size
        # Always keep the newest record, even if it alone is over budget.
        while len(self._data) > 1 and (len(self._data) > self._max
                                       or self._bytes > self._max_bytes):
            _, evicted = self._data.popitem(last=False)
            self._bytes -= evicted.size

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def get(self, rid: str) -> StoredResponse | None:
        rec = self._data.get(rid)
        if rec is not None:
            self._data.move_to_end(rid)
        return rec

    def history(self, rid: str) -> list[dict[str, Any]]:
        """All items from the start of the chain through `rid`, oldest first.

        Raises KeyError if an ancestor was evicted: the chain can't be rebuilt.
        """
        chain, cur = [], rid
        while cur is not None:
            rec = self._data[cur]
            chain.append(rec)
            cur = rec.parent_id
        return [item for rec in reversed(chain) for item in rec.items]


store = ResponseStore()
