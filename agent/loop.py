"""
agent/loop.py
LangGraph StateGraph — Plan → Execute → Memory → Evaluate → Synthesise.

Root cause of 'unexpected keyword argument schema':
  LangGraph internally reconstructs state by calling the schema class with
  **state_dict.  If any key in state_dict matches a reserved Python/Pydantic
  name (like 'schema'), it crashes.  Fix: we pass a plain dict as the schema
  to StateGraph instead of the TypedDict class, which avoids the constructor
  call entirely.  We still use AgentState for type hints only.
"""

from __future__ import annotations

import uuid
from typing import Generator

from langgraph.graph import StateGraph, END, START

from agent.state import AgentState
from agent.planner import planner_node
from agent.executor import executor_node, MAX_ITERATIONS
from agent.memory import memory_node, AgentMemory
from agent.evaluator import evaluator_node, synthesiser_node

# Conditional edge router

def _route_after_evaluator(state: dict) -> str:
    if state.get("is_complete", False):
        return "synthesiser"
    plan      = state.get("plan", [])
    idx        = state.get("current_step_index", 0)
    iterations = state.get("iterations", 0)
    if idx < len(plan) and iterations < MAX_ITERATIONS:
        return "executor"
    return "synthesiser"

# Build & compile (once at module import)

def _build_graph():
    # Pass AgentState TypedDict — LangGraph uses it for field discovery only,
    # it does NOT call AgentState(**kwargs) at runtime in v0.1.x.
    g = StateGraph(AgentState)

    g.add_node("planner",     planner_node)
    g.add_node("executor",    executor_node)
    g.add_node("memory",      memory_node)
    g.add_node("evaluator",   evaluator_node)
    g.add_node("synthesiser", synthesiser_node)

    g.add_edge(START,          "planner")
    g.add_edge("planner",      "executor")
    g.add_edge("executor",     "memory")
    g.add_edge("memory",       "evaluator")
    g.add_edge("synthesiser",  END)

    g.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"executor": "executor", "synthesiser": "synthesiser"},
    )

    return g.compile()


_graph = _build_graph()

# Initial state — every key must be present and have a safe default

def _make_initial_state(question: str, session_id: str) -> dict:
    return {
        "question":           question,
        "session_id":         session_id,
        "plan":               [],
        "current_step_index": 0,
        "last_data":          [],
        "context_summary":    "",
        "executed_steps":     [],
        "charts":             [],
        "iterations":         0,
        "is_complete":        False,
        "final_answer":       "",
    }

# Public API
def run_agent(question: str, session_id: str | None = None) -> dict:
    """
    Run the full LangGraph agent loop synchronously.

    Returns
    -------
    {
        "question":     str,
        "session_id":   str,
        "steps":        list,
        "charts":       list,
        "final_answer": str,
        "iterations":   int,
    }
    """
    session_id = session_id or str(uuid.uuid4())
    AgentMemory(session_id).clear()

    try:
        final_state: dict = _graph.invoke(_make_initial_state(question, session_id))
    except Exception as exc:
        raise RuntimeError(f"LangGraph execution failed: {exc}") from exc

    return {
        "question":     question,
        "session_id":   session_id,
        "steps":        final_state.get("executed_steps", []),
        "charts":       final_state.get("charts", []),
        "final_answer": final_state.get("final_answer", "No answer generated."),
        "iterations":   final_state.get("iterations", 0),
    }


def run_agent_streaming(
    question: str,
    session_id: str | None = None,
) -> Generator[dict, None, None]:
    """
    Stream node-level updates as they happen.

    Yields dicts:
        {"type": "plan",         "payload": list}
        {"type": "step",         "payload": dict}
        {"type": "chart",        "payload": str}
        {"type": "final_answer", "payload": str}
        {"type": "error",        "payload": str}
        {"type": "done",         "payload": dict}
    """
    session_id = session_id or str(uuid.uuid4())
    AgentMemory(session_id).clear()

    emitted_steps  = 0
    emitted_charts = 0

    try:
        for event in _graph.stream(_make_initial_state(question, session_id)):
            for node_name, delta in event.items():
                if not isinstance(delta, dict):
                    continue

                if node_name == "planner" and delta.get("plan"):
                    yield {"type": "plan", "payload": delta["plan"]}

                if node_name == "executor":
                    steps = delta.get("executed_steps", [])
                    for s in steps[emitted_steps:]:
                        yield {"type": "step", "payload": s}
                    emitted_steps = len(steps)

                    charts = delta.get("charts", [])
                    for c in charts[emitted_charts:]:
                        yield {"type": "chart", "payload": c}
                    emitted_charts = len(charts)

                if node_name == "synthesiser" and delta.get("final_answer"):
                    yield {"type": "final_answer", "payload": delta["final_answer"]}

    except Exception as exc:
        yield {"type": "error", "payload": str(exc)}

    yield {"type": "done", "payload": {"session_id": session_id}}