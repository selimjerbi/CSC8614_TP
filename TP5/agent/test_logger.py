from TP5.agent.logger import log_event

if __name__ == "__main__":
    run_id = "TEST_RUN"
    log_event(run_id, "node_start", {"node": "classify_email"})
    log_event(run_id, "node_end", {"node": "classify_email", "status": "ok"})
    print("OK - check TP5/runs/TEST_RUN.jsonl")
