from TP5.agent.logger import log_event
from TP5.agent.state import AgentState


def stub_reply(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_reply"})
    state.draft_v1 = (
        "Bonjour,\n\n"
        "Merci pour votre email. J’ai bien pris connaissance de votre demande et je reviens vers vous rapidement.\n\n"
        "Cordialement,"
    )
    log_event(state.run_id, "node_end", {"node": "stub_reply", "status": "ok"})
    return state


def stub_ask_clarification(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ask_clarification"})
    state.draft_v1 = (
        "Bonjour,\n\n"
        "Merci pour votre message. Pour pouvoir vous répondre, pouvez-vous préciser :\n"
        "- le contexte exact (UE / module / procédure concernée) ?\n"
        "- la date ou l’échéance associée, s’il y en a une ?\n\n"
        "Merci,\nCordialement,"
    )
    log_event(state.run_id, "node_end", {"node": "stub_ask_clarification", "status": "ok"})
    return state


def stub_escalate(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_escalate"})
    state.actions.append({
        "type": "handoff_human",
        "summary": "Escalade requise : demande sensible ou nécessitant validation humaine.",
    })
    log_event(state.run_id, "node_end", {"node": "stub_escalate", "status": "ok"})
    return state


def stub_ignore(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ignore"})
    state.actions.append({
        "type": "ignore",
        "reason": "Message hors périmètre / non prioritaire pour l’assistant.",
    })
    log_event(state.run_id, "node_end", {"node": "stub_ignore", "status": "ok"})
    return state
