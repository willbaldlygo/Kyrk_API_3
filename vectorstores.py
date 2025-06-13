from __future__ import annotations
from pathlib import Path
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

EMBED_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def _df_to_docs(df, tag: str) -> List[Document]:
    docs = []
    for idx, (_, row) in enumerate(df.iterrows()):
        docs.append(Document(page_content=row.to_json(),
                             metadata={"where": tag, "row_id": idx}))
    return docs

def build_or_update_fb_store(html_docs: List[Document], path: str | Path = "stores/fb_vdb"):
    path = Path(path)
    if path.exists():
        store = FAISS.load_local(str(path), get_embeddings())
        store.add_documents(html_docs)
    else:
        store = FAISS.from_documents(html_docs, get_embeddings())
    store.save_local(str(path))
    return store
