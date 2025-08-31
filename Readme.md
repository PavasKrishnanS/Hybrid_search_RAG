🧠 Hybrid RAG Chatbot (PDF Q&A)

A Streamlit-based RAG (Retrieval Augmented Generation) chatbot that answers questions using your local PDFs and live web search. Combines vector search, BM25, DuckDuckGo, and cross-encoder reranking for high-quality, cited answers. Powered by your local Ollama LLM.

🚀 Features
🔎 PDF Question Answering: Just drop PDFs into data/ and start asking.
🧬 Hybrid Retrieval: Vector (semantic) + BM25 (keyword) + live web search.
🤖 LLM Answers: Uses Ollama/gemma3 to answer strictly from context.
🏅 Cross-Encoder Reranking: Ensures the most relevant results.
💬 User-Friendly UI: Chat UI with expandable supporting context (cited by source).


🗂️ Folder Structure

HYBRID_SEARCH/
├── agent.py
├── app.py
├── hybrid_retrieve.py
├── data/         # <-- Put your PDFs here
├── vector_db/    # C



Quickstart
Install requirements


pip install -r requirements.txt
Put PDFs in /data folder

Run Ollama locally

Download: https://ollama.com/download
Launch a model (e.g.):

ollama run gemma3
Start the chatbot


streamlit run app.py
Chat and explore!

🛠️ Configuration
Change LLM: Edit the model name in agent.py.
Tune retrieval: Change CHUNK_SIZE, CHUNK_OVERLAP, or search parameters in hybrid_retrieve.py.
Paths: All PDFs should go into data/.
💡 How It Works
Chunk and Index PDFs (vector & BM25)
Hybrid Search: Vector search, BM25 keyword search, and web snippets.
Rerank: All results reranked by cross-encoder.
LLM Q&A: Top context passed to LLM for answer, with sources shown in chat.