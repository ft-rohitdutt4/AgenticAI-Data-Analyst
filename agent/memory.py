"""
agent/memory.py
LangGraph node: persists the latest step result into Redis (or a local dict
fallback) and refreshes state["context_summary"] for the next node.

Also exposes a thin AgentMemory class used by the synthesiser to retrieve
the full history at the end of the graph.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from agent.state import AgentState

#optional Redis 
try:
    import redis as _redis_lib

    _redis_url = os.getenv("REDIS_URL", "")
    _r = _redis_lib.from_url(_redis_url) if _redis_url else None
except Exception:
    _r = None

_local: dict[str, list] = {}
_TTL = 3600  # 1 h

# Low-level store helpers

def _load(key: str) -> list:
    if _r:
        try:
            raw = _r.get(key)
            return json.loads(raw) if raw else []
        except Exception:
            pass
    return _local.get(key, [])


def _save(key: str, data: list) -> None:
    if _r:
        try:
            _r.setex(key, _TTL, json.dumps(data, default=str))
            return
        except Exception:
            pass
    _local[key] = data


def _key(session_id: str) -> str:
    return f"agent:steps:{session_id}"

# AgentMemory class  (used by synthesiser / external callers)

class AgentMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id

    def add_step(self, step: str, tool: str, result: Any) -> None:
        steps = _load(_key(self.session_id))
        steps.append({"step": step, "tool": tool, "result": result, "ts": time.time()})
        _save(_key(self.session_id), steps)

    def get_all(self) -> list:
        return _load(_key(self.session_id))

    def get_context_summary(self) -> str:
        steps = _load(_key(self.session_id))
        if not steps:
            return "No prior steps."
        lines = []
        for i, s in enumerate(steps, 1):
            snippet = str(s["result"])[:600]
            lines.append(f"Step {i} [{s['tool']}]: {s['step']}\nResult: {snippet}")
        return "\n\n".join(lines)

    def clear(self) -> None:
        _save(_key(self.session_id), [])

# LangGraph node

def memory_node(state: AgentState) -> dict:
    """
    LangGraph node — 'memory'.

    Called after executor_node.
    Persists the last executed step to Redis/in-memory store and
    rebuilds context_summary for the next executor call.

    Reads:  state["executed_steps"], state["session_id"]
    Writes: state["context_summary"]
    """
    session_id = state["session_id"]
    executed_steps: list = state.get("executed_steps", [])

    if executed_steps:
        last = executed_steps[-1]
        mem = AgentMemory(session_id)
        mem.add_step(
            step=last["step"],
            tool=last["tool"],
            result=last["result"],
        )

    # Rebuild summary from store
    mem = AgentMemory(session_id)
    summary = mem.get_context_summary()

    return {"context_summary": summary}