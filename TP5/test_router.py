import uuid

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.nodes.classify_email import classify_email

if __name__ == "__main__":
    emails = load_all_emails()
    e = emails[0]

    state = AgentState(
        run_id=str(uuid.uuid4()),
        email_id=e["email_id"],
        subject=e["subject"],
        sender=e["from"],
        body=e["body"],
    )

    state = classify_email(state)
    print(state.decision.model_dump_json(indent=2))
