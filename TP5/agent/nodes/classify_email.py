import json
import re

from langchain_ollama import ChatOllama

from TP5.agent.logger import log_event
from TP5.agent.prompts import ROUTER_PROMPT
from TP5.agent.state import AgentState, Decision

# NOTE: modifiez PORT dans le code si nécessaire (local / serveur)
PORT = "11434"
LLM_MODEL = "mistral:7b-instruct"

REPAIR_PROMPT = """\
SYSTEM:
Tu es un correcteur de JSON. Tu ne modifies pas la sémantique.
Tu transforms l'output en JSON strict conforme au schéma.

USER:
Schéma attendu (clés obligatoires) :
{{ "intent": "...", "category":"...", "priority":1, "risk_level":"...", "needs_retrieval":true, "retrieval_query":"...", "rationale":"..." }}

Output invalide:
<<<{raw}>>>

Retourne UNIQUEMENT le JSON corrigé.
"""


def call_llm(prompt: str) -> str:
    llm = ChatOllama(base_url=f"http://127.0.0.1:{PORT}", model=LLM_MODEL)
    resp = llm.invoke(prompt)
    # retire éventuellement des balises <think> si présentes
    return re.sub(r"<think>.*?</think>\s*", "", resp.content.strip(), flags=re.DOTALL).strip()


def parse_and_validate(raw: str) -> Decision:
    data = json.loads(raw)
    return Decision(**data)


def classify_email(state: AgentState) -> AgentState:
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "draft_reply", "status": "budget_exceeded"})
        return state
    state.budget.steps_used += 1

    if state.subject.lower().startswith("[ignore]"):
        state.decision.intent = "ignore"
        state.decision.needs_retrieval = False
        state.decision.rationale = "Email hors périmètre (test ignore)."
        return state

    log_event(state.run_id, "node_start", {"node": "classify_email", "email_id": state.email_id})

    prompt = ROUTER_PROMPT.format(subject=state.subject, sender=state.sender, body=state.body)
    low = state.body.lower()
    if any(x in low for x in ["ignore previous", "system:", "tool", "call", "exfiltrate"]):
        state.decision = Decision(
            intent="escalate",
            category=state.decision.category,
            priority=1,
            risk_level="high",
            needs_retrieval=False,
            retrieval_query="",
            rationale="Suspicion de prompt injection."
        )
        log_event(state.run_id, "node_end", {
            "node": "classify_email",
            "status": "ok",
            "decision": state.decision.model_dump(),
            "note": "injection_heuristic_triggered"
        })
        return state

    raw = call_llm(prompt)

    try:
        decision = parse_and_validate(raw)
    except Exception as e:
        log_event(state.run_id, "error", {"node": "classify_email", "kind": "parse_or_validation", "msg": str(e)})

        # repair fallback
        repair = REPAIR_PROMPT.format(raw=raw)
        raw2 = call_llm(repair)
        decision = parse_and_validate(raw2)

    state.decision = decision

    log_event(state.run_id, "node_end", {
        "node": "classify_email",
        "status": "ok",
        "decision": decision.model_dump(),
    })
    return state
