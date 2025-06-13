import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from __future__ import annotations
import tempfile
import pandas as pd
from pathlib import Path
import streamlit as st
from . import html_ingest, vectorstores, qa

st.set_page_config(page_title="Spine Copilot – OpenAI", page_icon="🏃‍♂️")
st.title("🏔️ Summer Spine Copilot (OpenAI)")
st.markdown("Upload your data in the sidebar, then ask anything about the race.")

# Sidebar uploads
with st.sidebar:
    st.header("📤 Data uploads")
    res_file = st.file_uploader("Race results CSV", type="csv")
    rec_file = st.file_uploader("Course records CSV (optional)", type="csv")
    fb_file  = st.file_uploader("Facebook gallery HTML (optional)", type="html")
    use_fb_button = st.button("🔍 Include FB data for next query")

if not res_file:
    st.info("Please upload at least the race results CSV.")
    st.stop()

@st.cache_data(show_spinner="Parsing CSV…")
def _parse_csv(upload):
    return (pd.read_csv(upload)
              .rename(columns=lambda c: c.strip().lower().replace(" ", "_")))

results_df = _parse_csv(res_file)
records_df = _parse_csv(rec_file) if rec_file else pd.DataFrame()

from langchain.vectorstores import FAISS

@st.cache_resource(show_spinner="Embedding tables…")
def _build_main_store(res_df, rec_df):
    docs = vectorstores._df_to_docs(res_df, "results")  # type: ignore
    if not rec_df.empty:
        docs += vectorstores._df_to_docs(rec_df, "records")  # type: ignore
    return FAISS.from_documents(docs, vectorstores.get_embeddings())

main_store = _build_main_store(results_df, records_df)
main_chain = qa.build_chain(main_store)

fb_chain = None
if fb_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(fb_file.getvalue())
        tmp_path = tmp.name
    fb_docs = html_ingest.load_fb_docs(tmp_path)
    fb_store = vectorstores.build_or_update_fb_store(fb_docs)
    fb_chain = qa.build_chain(fb_store)

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.chat_message(role).markdown(msg, unsafe_allow_html=True)

prompt = st.chat_input("Ask me a question…")
if prompt:
    st.chat_message("user").markdown(prompt)
    with st.spinner("Thinking…"):
        answer_md, srcs = qa.answer(prompt,
                                    use_fb_button,
                                    results_df,
                                    records_df,
                                    main_chain,
                                    fb_chain)
    st.chat_message("assistant").markdown(answer_md)
    if srcs:
        with st.expander("Sources"):
            for s in srcs:
                m = s.metadata
                tag = f"{m.get('where', 'main')} » {m.get('post_id', m.get('row_id', '-'))}"
                st.markdown(f"**{tag}** — {s.page_content[:400]}…")
    st.session_state.chat.extend([("user", prompt), ("assistant", answer_md)])
