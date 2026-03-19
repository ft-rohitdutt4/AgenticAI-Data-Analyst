
from __future__ import annotations
from database.connection import get_schema_text
from llm.client import chat
from llm.prompts import (
    ANALYST_SYSTEM,
    EVALUATOR_PROMPT,
    FINAL_ANSWER_PROMPT,
)
from agent.state import AgentState

MAX_ITERATIONS = 5


def evaluator_node(state: AgentState) -> dict:
    """
    LangGraph node — 'evaluator'.

    Reads:  state["question"], state["context_summary"], state["iterations"],
            state["plan"], state["current_step_index"]
    Writes: state["is_complete"]
    """
    question: str = state["question"]
    summary: str = state.get("context_summary", "No prior steps.")
    iterations: int = state.get("iterations", 0)
    plan: list[dict] = state.get("plan", [])
    idx: int = state.get("current_step_index", 0)

    # Force completion if hard limits reached
    if iterations >= MAX_ITERATIONS:
        return {"is_complete": True}

    # All non-final steps consumed → done
    non_final = [s for s in plan if s.get("tool") != "final_answer"]
    if idx > len(non_final):
        return {"is_complete": True}

    # Ask the LLM
    schema = get_schema_text()
    system_msg = ANALYST_SYSTEM.format(schema=schema)
    user_msg = EVALUATOR_PROMPT.format(question=question, steps_summary=summary)

    response = chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=80,
    ).strip()

    complete = response.upper().startswith("YES")
    return {"is_complete": complete}


def synthesiser_node(state: AgentState) -> dict:
    """
    LangGraph node — 'synthesiser'.

    Terminal node: generates the final business answer from all findings.

    Reads:  state["question"], state["context_summary"]
    Writes: state["final_answer"]
    """
    question: str = state["question"]
    findings: str = state.get("context_summary", "No analysis steps were recorded.")

    schema = get_schema_text()
    system_msg = ANALYST_SYSTEM.format(schema=schema)
    user_msg = FINAL_ANSWER_PROMPT.format(question=question, findings=findings)

    answer = chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return {"final_answer": answer}