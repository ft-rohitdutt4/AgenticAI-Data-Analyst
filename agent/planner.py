"""
agent/planner.py
LangGraph node: takes the user question, calls the LLM, and produces
an ordered list of analysis steps stored in state["plan"].
"""

from __future__ import annotations

import json
import re

from database.connection import get_schema_text
from llm.client import chat
from llm.prompts import ANALYST_SYSTEM, PLANNER_PROMPT
from agent.state import AgentState

# LangGraph node function

def planner_node(state: AgentState) -> dict:
    """
    LangGraph node — 'planner'.

    Reads:  state["question"]
    Writes: state["plan"], state["current_step_index"], plus resets run state.
    """
    question = state["question"]
    schema = get_schema_text()

    system_msg = ANALYST_SYSTEM.format(schema=schema)
    user_msg = PLANNER_PROMPT.format(question=question)

    raw = chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    steps = _parse_plan(raw, question)

    # Only return keys this node is responsible for setting.
    # Do NOT reset executed_steps/charts — they belong to executor.
    return {
        "plan":               steps,
        "current_step_index": 0,
        "iterations":         0,
        "is_complete":        False,
        "last_data":          [],
        "context_summary":    "",
    }

# Helpers

def _parse_plan(raw: str, question: str) -> list[dict]:
    """Extract JSON plan array from the LLM response."""
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)

    if not json_match:
        return _fallback_plan(question)

    try:
        steps: list[dict] = json.loads(json_match.group())
    except json.JSONDecodeError:
        return _fallback_plan(question)

    # Sanitise tool names
    valid_tools = {
        "sql_query", "pandas_analysis",
        "anomaly_detection", "visualization", "final_answer",
    }
    for s in steps:
        if s.get("tool") not in valid_tools:
            s["tool"] = "sql_query"

    # Always end with final_answer
    if not steps or steps[-1].get("tool") != "final_answer":
        steps.append(
            {
                "step": "Synthesise findings into a clear business answer",
                "tool": "final_answer",
                "reasoning": "Always produce a human-readable conclusion.",
            }
        )

    return steps


def _fallback_plan(question: str) -> list[dict]:
    return [
        {
            "step": question,
            "tool": "sql_query",
            "reasoning": "Fallback: direct SQL query.",
        },
        {
            "step": "Summarise findings",
            "tool": "final_answer",
            "reasoning": "Generate final answer.",
        },
    ]