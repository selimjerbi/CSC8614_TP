import os
import time
import hashlib
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from TP5.agent.logger import log_event
from TP5.agent.state import EvidenceDoc

CHROMA_DIR = os.path.join("TP5", "chroma_db")

COLLECTION_NAME = "tp4_rag"

EMBEDDING_MODEL = "nomic-embed-text"

PORT = "11434"


def _hash_args(args: Dict[str, Any]) -> str:
    raw = repr(sorted(args.items())).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _format_snippet(doc: Document, max_len: int = 320) -> str:
    txt = doc.page_content.strip().replace("\n", " ")
    return (txt[:max_len] + "...") if len(txt) > max_len else txt


def rag_search_tool(
    run_id: str,
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[EvidenceDoc]:
    """
    Tool RAG : retourne des EvidenceDoc citables.
    """
    filters = filters or {}
    t0 = time.time()
    args = {"query": query, "k": k, "filters": filters}
    args_hash = _hash_args(args)

    if (not query.strip()) or (k > 10):
        log_event(run_id, "tool_call", {
            "tool": "rag_search",
            "args_hash": args_hash,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "error",
            "error": "invalid_args"
        })
        return []


    try:
        emb = OllamaEmbeddings(base_url=f"http://127.0.0.1:{PORT}", model=EMBEDDING_MODEL)
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )

        # NOTE: filters pas utilisés ici (simple MVP)
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        evidence: List[EvidenceDoc] = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            evidence.append(
                EvidenceDoc(
                    doc_id=f"doc_{i}",
                    doc_type=str(meta.get("doc_type", "unknown")),
                    source=str(meta.get("source", "unknown")),
                    snippet=_format_snippet(d),
                    score=None,  # Chroma/LangChain ne fournit pas toujours un score ici
                )
            )

        log_event(run_id, "tool_call", {
            "tool": "rag_search",
            "args_hash": args_hash,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "ok",
            "k": k,
            "n_docs": len(evidence),
        })
        return evidence

    except Exception as e:
        log_event(run_id, "tool_call", {
            "tool": "rag_search",
            "args_hash": args_hash,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "error",
            "error": str(e),
        })
        return []
