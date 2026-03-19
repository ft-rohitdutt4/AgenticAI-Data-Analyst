"""
agent/executor.py
LangGraph node: picks the current step from the plan, dispatches to the
correct tool, and appends the outcome to state["executed_steps"].
"""

from __future__ import annotations

from agent.state import AgentState
from tools.sql_tool import run_sql_tool
from tools.analysis_tool import run_analysis_tool
from tools.visualization_tool import run_visualization_tool
from tools.anomaly_tool import run_anomaly_tool

# Maximum number of executor iterations before forced termination
MAX_ITERATIONS = 5


def executor_node(state: AgentState) -> dict:
    """
    LangGraph node — 'executor'.

    Reads:
        state["plan"]               — full ordered plan
        state["current_step_index"] — which step to execute now
        state["context_summary"]    — prior memory for SQL prompts
        state["last_data"]          — rows from last sql_query
        state["iterations"]         — safety counter
        state["executed_steps"]     — accumulated steps so far
        state["charts"]             — accumulated charts so far

    Writes:
        state["executed_steps"]     — full updated list
        state["charts"]             — full updated list
        state["last_data"]          — replaced when sql_query runs
        state["current_step_index"] — incremented
        state["iterations"]         — incremented
    """
    plan: list[dict] = state.get("plan", [])
    idx: int = state.get("current_step_index", 0)
    context: str = state.get("context_summary", "")
    last_data: list[dict] = list(state.get("last_data", []))

    # Existing accumulated lists (we extend, not replace)
    existing_steps: list = list(state.get("executed_steps", []))
    existing_charts: list = list(state.get("charts", []))

    # Guard: nothing left to do
    if idx >= len(plan):
        return {
            "current_step_index": idx,
            "iterations": state.get("iterations", 0),
            "executed_steps": existing_steps,
            "charts": existing_charts,
        }

    step = plan[idx]
    tool = step.get("tool", "sql_query")
    task = step.get("step", "")

    # final_answer is handled by synthesiser_node — skip here
    if tool == "final_answer":
        return {
            "current_step_index": idx + 1,
            "iterations": state.get("iterations", 0) + 1,
            "executed_steps": existing_steps,
            "charts": existing_charts,
        }

    outcome = _dispatch(tool, task, context, last_data)

    new_step = {
        "step":       task,
        "tool":       tool,
        "result":     outcome["result"],
        "chart_json": outcome.get("chart_json"),
        "error":      outcome.get("error"),
    }

    updated_steps = existing_steps + [new_step]
    updated_charts = existing_charts + ([outcome["chart_json"]] if outcome.get("chart_json") else [])

    updates: dict = {
        "executed_steps":     updated_steps,
        "charts":             updated_charts,
        "current_step_index": idx + 1,
        "iterations":         state.get("iterations", 0) + 1,
    }

    # Replace last_data only when sql_query returned rows
    if tool == "sql_query" and outcome["result"].get("data"):
        updates["last_data"] = outcome["result"]["data"]

    return updates

# Dispatcher

def _dispatch(tool: str, task: str, context: str, last_data: list[dict]) -> dict:
    """Route to the appropriate tool function and normalise the response."""

    if tool == "sql_query":
        result = run_sql_tool(task, context)
        return {"result": result, "error": result.get("error")}

    elif tool == "pandas_analysis":
        result = run_analysis_tool(task, last_data)
        return {"result": result, "error": result.get("error")}

    elif tool == "anomaly_detection":
        result = run_anomaly_tool(task, last_data)
        return {"result": result, "error": result.get("error")}

    elif tool == "visualization":
        viz = run_visualization_tool(task, last_data)
        result = {"chart_type": viz.get("chart_type"), "error": viz.get("error")}
        return {
            "result": result,
            "chart_json": viz.get("chart_json"),
            "error": viz.get("error"),
        }

    else:
        err = f"Unknown tool: {tool}"
        return {"result": {"error": err}, "error": err}