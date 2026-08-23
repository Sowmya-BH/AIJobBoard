"""Assemble the LangGraph state machine.

LangSmith tracing: set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY (plus
optional LANGCHAIN_PROJECT). LangGraph auto-instruments every node run to
LangSmith when those are present — no code change needed; we just surface the
status at build time.
"""
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState
from . import nodes


def _langsmith_status():
    on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")
    return on and bool(os.environ.get("LANGCHAIN_API_KEY"))


def route_after_review(state) -> str:
    """Conditional edge out of human_review.

    add_info  → loop back to ats_score (re-score with the augmented profile)
    done      → END
    anything else (ask / cover_letter / tailored_resume / interview_questions) → extras
    """
    action = state.get("next_action")
    if action in ("add_info", "upload_resume"):   # both trigger a re-score
        return "ats_score"
    if action == "done":
        return END
    return "extras"


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)

    g.add_node("scout", nodes.scout_node)
    g.add_node("parser", nodes.parser_node)
    g.add_node("match", nodes.match_node)
    g.add_node("select_job", nodes.select_job_node)
    g.add_node("ats_score", nodes.ats_node)
    g.add_node("human_review", nodes.human_review_node)
    g.add_node("extras", nodes.extras_node)

    # parallel fork: both branches start from START
    g.add_edge(START, "scout")
    g.add_edge(START, "parser")
    # fan-in: match waits for BOTH scout and parser (LangGraph super-step)
    g.add_edge("scout", "match")
    g.add_edge("parser", "match")

    g.add_edge("match", "select_job")
    g.add_edge("select_job", "ats_score")
    g.add_edge("ats_score", "human_review")

    # HITL loop
    g.add_conditional_edges("human_review", route_after_review,
                            {"ats_score": "ats_score", "extras": "extras", END: END})
    # after producing an extra / answering, return to review for the next action
    g.add_edge("extras", "human_review")

    return g.compile(checkpointer=checkpointer or MemorySaver())
