import os
import numpy as np
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS

DATA_DIR = "./data"
VECTOR_DB = "./vector_db/chroma_db"
COLLECTION = "hybrid_rag"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

class HybridRAGRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')
        self.chunks, self.metas, self.bm25, self.collection = self._build_indices()

    def _extract_pdf_texts(self):
        texts, files = [], []
        for file in os.listdir(DATA_DIR):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(DATA_DIR, file)
                text = ''
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t:
                                text += t + "\n"
                except Exception as e:
                    print(f"Could not open {file}: {e}")
                    continue
                if text.strip():
                    texts.append(text)
                    files.append(file)
        return texts, files

    def _chunk_texts(self, texts, files):
        chunks, metas = [], []
        for txt, fname in zip(texts, files):
            start = 0
            while start < len(txt):
                chunk = txt[start:start + CHUNK_SIZE]
                if len(chunk.strip()) > 10:
                    chunks.append(chunk)
                    metas.append({"filename": fname})
                start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks, metas

    def _build_indices(self):
        texts, files = self._extract_pdf_texts()
        chunks, metas = self._chunk_texts(texts, files)
        if not chunks:
            print("No PDF data or chunks found!")
            return [], [], None, None
        # embeddings
        embeddings = self.encoder.encode(chunks)
        bm25 = BM25Okapi([c.split() for c in chunks])
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB)
        collection = chroma_client.get_or_create_collection(name=COLLECTION)
        # Clean and add new
        all_ids = collection.get(include=[])["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metas)):
            collection.add(ids=[f"chunk_{i}"], embeddings=[emb.tolist()], documents=[chunk], metadatas=[meta])
        return chunks, metas, bm25, collection

    def vector_search(self, query, top_k=4, threshold=0.5):
        user_emb = self.encoder.encode([query])[0]
        res = self.collection.query(
            query_embeddings=[user_emb.tolist()],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        docs = res["documents"][0]
        dists = res.get("distances", [[1]*top_k])[0]
        metas_ = res["metadatas"][0]
        relevant = [(doc, meta) for doc, dist, meta in zip(docs, dists, metas_) if dist < threshold]
        if not relevant and docs and metas_:
            relevant = [(docs[0], metas_[0])]
        return relevant

    def bm25_search(self, query, top_k=4):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.split())
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], self.metas[i]) for i in idxs]
    def duckduckgo_search(self, query, top_k=2):
        try:
            with DDGS() as ddgs:
                return [
                    (r.get('body') or r.get('snippet') or "", {"filename": "duckduckgo"})
                    for r in ddgs.text(query, max_results=top_k)
                ]
        except Exception as e:
            
            print(f"[WARN] DuckDuckGo web search failed: {e}")
            return []

    def rerank(self, query, tuples, top_n=4):
        docs = [d for d, _ in tuples]
        pairs = [(query, doc) for doc in docs]
        if not pairs:
            return []
        scores = self.cross_encoder.predict(pairs)
        reranked = [tuples[i] for i in np.argsort(scores)[::-1][:top_n]]
        return reranked

    def hybrid_retrieve(self, query, top_k_vec=4, top_k_bm25=4, top_k_web=2, top_k_rerank=4):
        vec_results = self.vector_search(query, top_k=top_k_vec)
        bm_results = self.bm25_search(query, top_k=top_k_bm25)
        web_results = self.duckduckgo_search(query, top_k=top_k_web)
        all_results = vec_results + bm_results + web_results
        top_reranked = self.rerank(query, all_results, top_n=top_k_rerank)
        context = "\n".join([doc for doc, meta in top_reranked])
        return context, top_reranked