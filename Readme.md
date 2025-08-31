# 🧠 Hybrid RAG Chatbot (PDF Q\&A)

A Streamlit-based RAG (Retrieval Augmented Generation) chatbot that answers questions using your local PDFs and live web search. It combines vector search, BM25, DuckDuckGo, and cross-encoder reranking for high-quality, cited answers. The chatbot is powered by your local Ollama LLM.

<img width="1949" height="929" alt="image" src="https://github.com/user-attachments/assets/65bf71db-d8d1-4685-b415-be266cb5a298" />


## 🚀 Features

  * **🔎 PDF Question Answering**: Just drop PDFs into the `data/` folder and start asking questions.
  * **🧬 Hybrid Retrieval**: The system uses a combination of vector (semantic) and BM25 (keyword) search to find relevant information from your documents, supplemented by live web search.
  * **🤖 LLM Answers**: The chatbot uses your local Ollama LLM (e.g., Gemma 3) to answer questions strictly from the provided context, preventing hallucinations.
  * **🏅 Cross-Encoder Reranking**: A cross-encoder model reranks all retrieved results to ensure that only the most relevant information is passed to the LLM.
  * **💬 User-Friendly UI**: The chatbot features a clean chat interface with expandable sections that show the supporting context and sources for each answer.

-----

## 🗂️ Folder Structure

```
HYBRID_SEARCH_RAG/
├── agent.py               # Handles the LLM interaction and final answer generation.
├── app.py                 # The main Streamlit application file.
├── hybrid_retrieve.py     # Contains the logic for hybrid search and reranking.
├── data/                  # <-- Put your PDFs here.
└── vector_db/             # Caches the vector and BM25 indices for your PDFs.
```

-----

## 🚀 Quickstart

1.  **Install Requirements**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Add PDFs**
    Place your PDF files into the `data/` folder.

3.  **Run Ollama Locally**

      * Download and install Ollama: [https://ollama.com/download](https://ollama.com/download)
      * Launch a language model from your terminal (e.g., Gemma 3):
        ```bash
        ollama run gemma3
        ```

4.  **Start the Chatbot**

      * From the root directory of the project, run the Streamlit application:
        ```bash
        streamlit run app.py
        ```

5.  **Chat and Explore\!**

      * The chatbot will open in your web browser. You can now ask questions about the PDFs you added and receive cited answers.

-----

## 🛠️ Configuration

  * **Change LLM**: To use a different Ollama model, simply edit the `agent.py` file and change the model name.
  * **Tune Retrieval**: The retrieval performance can be adjusted by changing parameters like `CHUNK_SIZE` and `CHUNK_OVERLAP` in `hybrid_retrieve.py`. You can also fine-tune the search parameters for the vector, BM25, or web components.
  * **Paths**: All PDFs must be placed inside the `data/` folder for the chatbot to index them correctly.

-----

## 💡 How It Works

1.  **Chunk and Index PDFs**: The system first processes your PDFs, splitting them into manageable text chunks. These chunks are then indexed for both vector search (semantic meaning) and BM25 (keyword matching).

2.  **Hybrid Search**: When a user asks a question, the system performs three parallel searches:

      * **Vector Search**: Finds document chunks semantically similar to the query.
      * **BM25 Keyword Search**: Finds chunks with matching keywords.
      * **Web Search**: Uses a tool like DuckDuckGo to get real-time context from the web.

3.  **Rerank**: All results from the hybrid search are passed through a **cross-encoder model**. This model reranks the results based on their relevance to the original query, ensuring that only the most useful information proceeds to the next step.

4.  **LLM Q\&A**: The top-ranked results are bundled as context and passed to the local Ollama LLM. The LLM generates an answer based *only* on this context, reducing the likelihood of hallucinations. The source of each piece of information is cited and displayed in the chatbot's UI.


