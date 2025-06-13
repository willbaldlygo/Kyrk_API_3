from pathlib import Path
from typing import List
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

__all__ = ["load_fb_docs"]

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

def load_fb_docs(html_file: str | Path) -> List[Document]:
    html_file = Path(html_file)
    docs = BSHTMLLoader(str(html_file)).load()
    split_docs = _SPLITTER.split_documents(docs)
    for idx, d in enumerate(split_docs):
        d.metadata.update({"post_id": f"post_{idx}", "where": "facebook", "source_file": html_file.name})
    return split_docs
