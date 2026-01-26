import re
from typing import List

from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

RE_CIT = re.compile(r"\[(doc_\d+)\]")


def _extract_citations(text: str) -> List[str]:
    return sorted(set(RE_CIT.findall(text or "")))


def finalize(state: AgentState) -> AgentState:
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "draft_reply", "status": "budget_exceeded"})
        return state

    state.budget.steps_used += 1

    log_event(state.run_id, "node_start", {"node": "finalize"})

    intent = state.decision.intent

    if intent == "reply":
        cits = _extract_citations(state.draft_v1)
        state.final_kind = "reply"
        if cits:
            state.final_text = state.draft_v1.strip() + "\n\nSources: " + " ".join(f"[{c}]" for c in cits)
        else:
            state.final_text = state.draft_v1.strip() or (
                "Bonjour,\n\nMerci pour votre message. Je reviens vers vous dès que possible.\n\nCordialement,"
            )

    elif intent == "ask_clarification":
        state.final_kind = "clarification"
        state.final_text = state.draft_v1.strip() or (
            "Bonjour,\n\nMerci pour votre message. Pour pouvoir vous répondre, pouvez-vous préciser :\n"
            "1) le contexte exact (UE / module / procédure concernée) ?\n"
            "2) la date / échéance associée ?\n"
            "3) la question précise à laquelle vous souhaitez une réponse ?\n\n"
            "Cordialement,"
        )

    elif intent == "escalate":
        state.final_kind = "handoff"
        state.actions.append({
            "type": "handoff_packet",
            "run_id": state.run_id,
            "email_id": state.email_id,
            "summary": (
                f"Escalade : demande nécessitant validation humaine. "
                f"Sujet='{state.subject}'. Expéditeur='{state.sender}'."
            ),
            "evidence_ids": [d.doc_id for d in state.evidence],
        })
        state.final_text = (
            "Bonjour,\n\nVotre demande nécessite une validation humaine. "
            "Je la transmets avec un résumé et les éléments disponibles.\n\nCordialement,"
        )

    else:
        state.final_kind = "ignore"
        state.final_text = "Message ignoré (hors périmètre / non actionnable)."

    log_event(state.run_id, "node_end", {"node": "finalize", "status": "ok", "final_kind": state.final_kind})
    return state
