import streamlit as st
import re
from hybrid_retrieve import HybridRAGRetriever
from agent import ask_ollama

st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖")

st.markdown(
    "<h1 style='text-align: center;'>Hybrid RAG Chatbot (PDF Q&A)</h1>",
    unsafe_allow_html=True,
)
st.info(
    "🔎 **This is a Retrieval-Augmented Generation (RAG) chatbot.** "
    "It answers questions using vector search, BM25, and web search to retrieve facts "
    "from all PDFs stored in the `data` folder and from the public web."
)


def escape_markdown(text):
    return re.sub(r'([_*`])', r'\\\1', text)

@st.cache_resource(show_spinner="Loading PDFs, building indices ...")
def get_retriever():
    return HybridRAGRetriever()
retriever = get_retriever()

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(escape_markdown(msg))
    else:
        st.chat_message("assistant").markdown(escape_markdown(msg))


user_prompt = st.chat_input("Ask any question about your PDFs/data...")

if user_prompt:
    st.session_state.history.append(("user", user_prompt))
    st.chat_message("user").markdown(escape_markdown(user_prompt))

    with st.spinner("Retrieving relevant context and generating answer..."):
        context, top_reranked = retriever.hybrid_retrieve(user_prompt)
        answer = ask_ollama(user_prompt, context)

    st.session_state.history.append(("assistant", answer))
    st.chat_message("assistant").markdown(escape_markdown(answer))

   
    if top_reranked:
        with st.expander("Show supporting context (PDF/Web sources)"):
            for idx, (chunk, meta) in enumerate(top_reranked, 1):
                fname = meta.get('filename', 'unknown')
                chunk_preview = chunk[:333] + ("..." if len(chunk) > 333 else "")
                st.markdown(f"**{idx}. From:** `{fname}`\n\n```\n{chunk_preview}\n```")