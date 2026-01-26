from TP5.agent.logger import log_event
from TP5.agent.state import AgentState


def check_evidence(state: AgentState) -> AgentState:
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "draft_reply", "status": "budget_exceeded"})
        return state

    state.budget.steps_used += 1

    log_event(state.run_id, "node_start", {"node": "check_evidence"})

    state.evidence_ok = bool(state.last_draft_had_valid_citations)

    log_event(state.run_id, "node_end", {
        "node": "check_evidence",
        "status": "ok",
        "evidence_ok": state.evidence_ok,
        "last_draft_had_valid_citations": state.last_draft_had_valid_citations,
        "retrieval_attempts": state.budget.retrieval_attempts,
    })
    return state
