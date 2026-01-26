from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, conint


Intent = Literal["reply", "ask_clarification", "escalate", "ignore"]
Category = Literal["admin", "teaching", "research", "other"]
RiskLevel = Literal["low", "med", "high"]


class Decision(BaseModel):
    intent: Intent = "reply"
    category: Category = "other"
    priority: conint(ge=1, le=5) = 3
    risk_level: RiskLevel = "low"
    needs_retrieval: bool = True
    retrieval_query: str = ""
    rationale: str = "1 phrase max."


class RetrievalSpec(BaseModel):
    query: str
    k: conint(ge=1, le=10) = 5
    filters: Dict[str, Any] = Field(default_factory=dict)


class EvidenceDoc(BaseModel):
    doc_id: str               # ex: "doc_1"
    doc_type: str             # ex: "email" / "admin_pdf"
    source: str               # ex: filename / message_id
    snippet: str              # court extrait citables
    score: Optional[float] = None


class ToolCallRecord(BaseModel):
    tool_name: str
    args_hash: str
    status: Literal["ok", "error"] = "ok"
    latency_ms: int = 0
    error: str = ""


class Budget(BaseModel):
    max_steps: int = 8
    steps_used: int = 0
    max_tool_calls: int = 6
    tool_calls_used: int = 0
    max_retrieval_attempts: int = 2
    retrieval_attempts: int = 0

    def can_step(self) -> bool:
        return self.steps_used < self.max_steps

    def can_call_tool(self) -> bool:
        return self.tool_calls_used < self.max_tool_calls

    def can_retrieve(self) -> bool:
        return self.retrieval_attempts < self.max_retrieval_attempts


class AgentState(BaseModel):
    run_id: str
    email_id: str
    subject: str
    sender: str
    body: str

    final_text: str = ""
    final_kind: str = ""  # reply / clarification / handoff / ignore

    decision: Decision = Field(default_factory=Decision)
    retrieval_spec: Optional[RetrievalSpec] = None
    evidence: List[EvidenceDoc] = Field(default_factory=list)

    draft_v1: str = ""
    draft_v2: str = ""

    evidence_ok: bool = False
    last_draft_had_valid_citations: bool = False

    actions: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
