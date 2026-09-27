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


@dataclass
class StoredResponse:
    id: str
    session_id: str  # the (possibly implicit) session whose thread ran it
    parent_id: str | None
    items: list[dict[str, Any]]  # raw Responses items this turn added
    instructions: str | None
    created_at: float = field(default_factory=time.time)


class ResponseStore:
    def __init__(self, max_responses: int = MAX_RESPONSES) -> None:
        self._max = max_responses
        self._data: OrderedDict[str, StoredResponse] = OrderedDict()

    def add(self, rec: StoredResponse) -> None:
        self._data[rec.id] = rec
        self._data.move_to_end(rec.id)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

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
