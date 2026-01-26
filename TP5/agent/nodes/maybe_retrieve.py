from TP5.agent.logger import log_event
from TP5.agent.state import AgentState, RetrievalSpec
from TP5.agent.tools.rag_tool import rag_search_tool


def maybe_retrieve(state: AgentState) -> AgentState:
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "draft_reply", "status": "budget_exceeded"})
        return state

    state.budget.steps_used += 1

    log_event(state.run_id, "node_start", {"node": "maybe_retrieve"})

    if not state.decision.needs_retrieval:
        log_event(state.run_id, "node_end", {"node": "maybe_retrieve", "status": "skipped"})
        return state

    if not state.budget.can_call_tool() or not state.budget.can_retrieve():
        state.add_error("Budget retrieval/tool dépassé")
        log_event(state.run_id, "node_end", {"node": "maybe_retrieve", "status": "budget_exceeded"})
        return state

    query = state.decision.retrieval_query.strip() or state.body[:200]
    spec = RetrievalSpec(query=query, k=5, filters={})
    state.retrieval_spec = spec

    state.budget.retrieval_attempts += 1
    state.budget.tool_calls_used += 1

    state.evidence = rag_search_tool(
        run_id=state.run_id,
        query=spec.query,
        k=spec.k,
        filters=spec.filters,
    )

    log_event(state.run_id, "node_end", {
        "node": "maybe_retrieve",
        "status": "ok",
        "n_docs": len(state.evidence),
    })
    return state
