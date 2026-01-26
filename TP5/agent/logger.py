import json
import os
from datetime import datetime
from typing import Any, Dict

RUNS_DIR = os.path.join("TP5", "runs")


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def log_event(run_id: str, event: str, data: Dict[str, Any]) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{run_id}.jsonl")

    payload = {
        "run_id": run_id,
        "ts": now_iso(),
        "event": event,
        "data": data,
    }

    # 1 event = 1 ligne JSON
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")
