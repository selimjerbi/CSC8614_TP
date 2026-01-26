from TP5.agent.state import AgentState


def route(state: AgentState) -> str:
    """
    Routing déterministe (testable). Le LLM propose une décision,
    mais le code choisit la branche d'exécution.
    """
    intent = state.decision.intent

    if intent == "reply":
        return "reply"
    if intent == "ask_clarification":
        return "ask_clarification"
    if intent == "escalate":
        return "escalate"
    return "ignore"
