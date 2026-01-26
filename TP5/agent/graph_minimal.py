from langgraph.graph import StateGraph, END
from TP5.agent.nodes.maybe_retrieve import maybe_retrieve
from TP5.agent.nodes.draft_reply import draft_reply
from TP5.agent.nodes.check_evidence import check_evidence
from TP5.agent.nodes.rewrite_query import rewrite_query
from TP5.agent.nodes.finalize import finalize

from TP5.agent.state import AgentState
from TP5.agent.routing import route
from TP5.agent.nodes.classify_email import classify_email
from TP5.agent.nodes.stubs import (
    stub_reply,
    stub_ask_clarification,
    stub_escalate,
    stub_ignore,
)

def after_check(state: AgentState) -> str:
    if state.evidence_ok or not state.decision.needs_retrieval:
        return "end"
    if state.budget.retrieval_attempts < state.budget.max_retrieval_attempts:
        return "rewrite"
    return "end"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_email", classify_email)
    g.add_node("reply", draft_reply)
    g.add_node("ask_clarification", stub_ask_clarification)
    g.add_node("escalate", stub_escalate)
    g.add_node("ignore", stub_ignore)
    g.add_node("maybe_retrieve", maybe_retrieve)
    g.add_node("check_evidence", check_evidence)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("finalize", finalize)

    g.set_entry_point("classify_email")
    
    g.add_conditional_edges(
        "classify_email",
        route,
        {
            "reply": "maybe_retrieve",
            "ask_clarification": "ask_clarification",
            "escalate": "escalate",
            "ignore": "ignore",
        },
    )

    g.add_conditional_edges("check_evidence", after_check, {
    "end": "finalize",
    "rewrite": "rewrite_query",
    })

    g.add_edge("ask_clarification", "finalize")
    g.add_edge("escalate", "finalize")
    g.add_edge("ignore", "finalize")
    g.add_edge("maybe_retrieve", "reply")
    g.add_edge("reply", "check_evidence")
    g.add_edge("rewrite_query", "maybe_retrieve")
    g.add_edge("finalize", END)

    return g.compile()
