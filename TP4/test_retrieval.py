"""
test_retrieval.py
Teste la recherche documentaire (retrieval) sans appeler le LLM.

Usage:
  python TP4/test_retrieval.py "VOTRE QUESTION"
"""

import os
import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = os.path.join("TP4", "chroma_db")
COLLECTION_NAME = "tp4_rag"

# IMPORTANT: même modèle d'embedding que pour l'index
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K = 5
PORT = "11434"  # Ollama local (Option A)


def main():
    if len(sys.argv) < 2:
        print('Usage: python TP4/test_retrieval.py "VOTRE QUESTION"')
        sys.exit(1)

    question = sys.argv[1]

    emb = OllamaEmbeddings(base_url=f"http://127.0.0.1:{PORT}", model=EMBEDDING_MODEL)

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=CHROMA_DIR,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    print("=" * 80)
    print(f"[QUERY] {question}")
    print(f"[RESULTS] top-{TOP_K}")
    print(f"[CONFIG] embedding_model={EMBEDDING_MODEL} port={PORT}")
    print("=" * 80)

    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        source = meta.get("source", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        excerpt = d.page_content[:300].replace("\n", " ")
        print(f"\n[{i}] ({doc_type}) {source}")
        print(f"     {excerpt} ...")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
