from __future__ import annotations
import re
from typing import List, Tuple
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.schema import Document
from .llm import get_llm

NAME_ONLY = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ' -]{3,}$")

def build_chain(store):
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=store.as_retriever(search_type="mmr", search_kwargs={"k": 4}),
        return_source_documents=True,
    )

def _is_name_only(q: str) -> bool:
    return bool(NAME_ONLY.fullmatch(q.strip()))

def answer(query: str,
           use_fb: bool,
           df_results: pd.DataFrame,
           df_records: pd.DataFrame,
           main_chain,
           fb_chain=None) -> Tuple[str, List[Document]]:
    if _is_name_only(query):
        parts = []
        sources: List[Document] = []
        res = df_results[df_results["name"].str.contains(query, case=False, na=False)]
        rec = df_records[df_records["name"].str.contains(query, case=False, na=False)]
        if not res.empty:
            parts.append("### Race results\n\n" + res.to_markdown(index=False))
        if not rec.empty:
            parts.append("### Course records\n\n" + rec.to_markdown(index=False))
        if not parts:
            answer_text = f"No runner named **{query}** in uploaded data."
        else:
            answer_text = "\n\n".join(parts)
        if use_fb and fb_chain:
            fb_out = fb_chain({"query": query})
            answer_text += "\n\n### FB Mentions\n\n" + fb_out["result"]
            sources.extend(fb_out["source_documents"])
        return answer_text, sources
    main_out = main_chain({"query": query})
    answer_text = main_out["result"]
    sources = main_out["source_documents"]
    if use_fb and fb_chain:
        fb_out = fb_chain({"query": query})
        answer_text += "\n\n---\n\n" + fb_out["result"]
        sources.extend(fb_out["source_documents"])
    return answer_text, sources
