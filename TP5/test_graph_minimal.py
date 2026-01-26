import uuid

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

if __name__ == "__main__":
    emails = load_all_emails()
    print("NB EMAILS:", len(emails))
    print("KEYS:", emails[0].keys() if emails else None)
    print("EMAIL IDS:", [e.get("email_id") for e in emails])

    e = next(x for x in emails if x.get("path", "").endswith("E11.md"))


    print("EMAIL ID:", e["email_id"])
    print("BODY PREVIEW:\n", e["body"][:300])


    state = AgentState(
        run_id=str(uuid.uuid4()),
        email_id=e["email_id"],
        subject=e["subject"],
        sender=e["from"],
        body=e["body"],
    )

    app = build_graph()
    out = app.invoke(state)

    print("=== DECISION ===")
    print(out["decision"].model_dump_json(indent=2))
    print("\n=== DRAFT_V1 ===")
    print(out["draft_v1"])
    print("\n=== ACTIONS ===")
    print(out["actions"])
    print("\n=== EVIDENCE ===")
    print([{"doc_id": d.doc_id, "doc_type": d.doc_type, "source": d.source} for d in out["evidence"]])
    print("\n=== FINAL ===")
    print("kind =", out["final_kind"])
    print(out["final_text"])
