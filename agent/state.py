"""
agent/state.py
Agent state definition for LangGraph.

We define AgentState as a plain Python TypedDict.  The field name 'schema'
is intentionally avoided — LangGraph passes state keys as **kwargs when it
reconstructs state internally, and 'schema' conflicts with Pydantic / JSON
Schema validation internals.

All list fields default to [] via the graph's initial_state dict.
"""

from __future__ import annotations
from typing_extensions import TypedDict


class AgentState(TypedDict):
    #inputs 
    question:           str
    session_id:         str

    #planner output
    plan:               list        # [{step, tool, reasoning}, ...]
    current_step_index: int

    # running state
    last_data:          list        # rows from most recent sql_query
    context_summary:    str

    # accumulated outputs
    executed_steps:     list        # list of step result dicts
    charts:             list        # list of Plotly JSON strings

    #control
    iterations:         int
    is_complete:        bool

    #final output
    final_answer:       str