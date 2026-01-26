# TP5/run_batch.py
import os
import uuid
from typing import List

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

OUT_MD = os.path.join("TP5", "batch_results.md")
RUNS_DIR = os.path.join("TP5", "runs")


def md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def count_in_jsonl(path: str, needle: str) -> int:
    """Compte le nombre de lignes contenant `needle` dans un JSONL (fallback robuste)."""
    if not os.path.exists(path):
        return 0
    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if needle in line:
                c += 1
    return c


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)

    emails = load_all_emails()
    # garder 8–12 si tu en as plus
    emails = emails[:12]

    app = build_graph()

    rows: List[str] = []
    rows.append("| email_id | subject | intent | category | risk | final_kind | tool_calls | retrieval_attempts | notes |")
    rows.append("|---|---|---|---|---|---|---:|---:|---|")

    for e in emails:
        run_id = str(uuid.uuid4())

        state = AgentState(
            run_id=run_id,
            email_id=e["email_id"],
            subject=e["subject"],
            sender=e["from"],
            body=e["body"],
        )

        out = app.invoke(state)

        intent = out["decision"].intent
        category = out["decision"].category
        risk = out["decision"].risk_level
        final_kind = out["final_kind"]

        # pointeur vers log JSONL (supposé écrit par log_event)
        run_log = os.path.join(RUNS_DIR, f"{run_id}.jsonl")

        # métriques simples (robuste sans dépendre du schéma exact)
        # tool_calls : compte des appels tool, et à défaut des mentions "rag_search"
        tool_calls = count_in_jsonl(run_log, '"event": "tool_call"')
        if tool_calls == 0:
            tool_calls = count_in_jsonl(run_log, '"tool":')  # fallback

        # retrieval_attempts : compte des tentatives de retrieval (node start/end ou event dédié)
        retrieval_attempts = count_in_jsonl(run_log, "retrieval")
        # éviter de surcompter des occurrences trop larges : on favorise "rag_search" si présent
        rag_calls = count_in_jsonl(run_log, '"tool": "rag_search"')
        if rag_calls > 0:
            retrieval_attempts = rag_calls

        notes = f"runs/{run_id}.jsonl"

        rows.append(
            "| "
            + " | ".join([
                md_escape(out.get("email_id", e["email_id"])),
                md_escape(out.get("subject", e["subject"]))[:60],
                md_escape(intent),
                md_escape(category),
                md_escape(risk),
                md_escape(final_kind),
                str(tool_calls),
                str(retrieval_attempts),
                md_escape(notes),
            ])
            + " |"
        )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
