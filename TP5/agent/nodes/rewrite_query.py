import json
import re
from langchain_ollama import ChatOllama

from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

PORT = "11434"
LLM_MODEL = "mistral:7b-instruct"

REWRITE_PROMPT = """\
SYSTEM:
Tu réécris UNE requête de recherche car la première a renvoyé peu de résultats OU une evidence peu exploitable.
Tu proposes UNE requête alternative plus spécifique et courte (max 12 mots).
Tu n'inventes pas de contenu, seulement une requête.

USER:
Sujet: {subject}
Expéditeur: {sender}
Corps (début):
{body_head}

Query initiale: "{q1}"
Nombre de documents: {n_docs}

Retourne UNIQUEMENT ce JSON:
{{"query_rewrite":"..."}}
"""

def call_llm(prompt: str) -> str:
    llm = ChatOllama(base_url=f"http://127.0.0.1:{PORT}", model=LLM_MODEL)
    resp = llm.invoke(prompt)
    return re.sub(r"<think>.*?</think>\s*", "", resp.content.strip(), flags=re.DOTALL).strip()

def rewrite_query(state: AgentState) -> AgentState:
    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "draft_reply", "status": "budget_exceeded"})
        return state
    state.budget.steps_used += 1
    log_event(state.run_id, "node_start", {"node": "rewrite_query"})

    q1 = state.decision.retrieval_query.strip() or state.body[:200]
    prompt = REWRITE_PROMPT.format(
        subject=state.subject,
        sender=state.sender,
        body_head=state.body[:300],
        q1=q1,
        n_docs=len(state.evidence),
    )

    raw = call_llm(prompt)
    try:
        q2 = json.loads(raw).get("query_rewrite", "").strip()
    except Exception:
        q2 = ""

    if not q2:
        q2 = q1
        state.add_error("rewrite_query: empty rewrite")

    # on limite la taille (sécurité)
    state.decision.retrieval_query = q2[:120]

    log_event(state.run_id, "node_end", {"node": "rewrite_query", "status": "ok", "q2": state.decision.retrieval_query})
    return state
