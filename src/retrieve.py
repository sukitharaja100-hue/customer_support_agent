"""
Retrieval index used to ground reply drafts in the brand's actual historical
resolutions: given a new customer message, find the most similar past
customer messages this brand has replied to, and surface those real replies
as reference material for the drafting step (src/reply.py).

DEFAULT BACKEND: TF-IDF + cosine similarity (scikit-learn only, fully
offline). Same reasoning as src/cluster_explore.py -- no model weights to
download, keeps `make eval` reproducible in under 15 minutes with zero
network dependency beyond the LLM API calls you're already making for
classification/generation/judging.

OPTIONAL: sentence-transformers + FAISS for higher-recall semantic
retrieval on paraphrased queries (TF-IDF can miss "phone won't turn on" vs
"my iPhone won't power on" if they don't share n-grams). Swap in via
--backend sbert once you have internet access to pull the model.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        self.matrix = None
        self.query_texts = None
        self.reply_texts = None

    def fit(self, query_texts, reply_texts):
        self.query_texts = list(query_texts)
        self.reply_texts = list(reply_texts)
        self.matrix = self.vec.fit_transform(self.query_texts)
        return self

    def search(self, query, k=3):
        q_vec = self.vec.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        return [
            (self.query_texts[i], self.reply_texts[i], float(sims[i]))
            for i in top_idx
        ]


class SbertRetriever:
    """Optional higher-quality backend. Requires `pip install sentence-transformers faiss-cpu`
    and internet access to download `all-MiniLM-L6-v2` on first use."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.query_texts = None
        self.reply_texts = None

    def fit(self, query_texts, reply_texts):
        import faiss

        self.query_texts = list(query_texts)
        self.reply_texts = list(reply_texts)
        emb = self.model.encode(self.query_texts, show_progress_bar=False)
        emb = emb.astype("float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        return self

    def search(self, query, k=3):
        import faiss

        q = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, k)
        return [
            (self.query_texts[i], self.reply_texts[i], float(s))
            for i, s in zip(idx[0], scores[0])
        ]


def build_retriever(pairs_parquet: str, backend: str = "tfidf"):
    df = pd.read_parquet(pairs_parquet)
    cls = TfidfRetriever if backend == "tfidf" else SbertRetriever
    retriever = cls().fit(df["cust_text_clean"], df["brand_text_clean"])
    return retriever


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/interim/pairs.parquet")
    ap.add_argument("--query", default="my phone battery drains so fast")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    args = ap.parse_args()

    retriever = build_retriever(args.pairs, args.backend)
    for cust, reply, score in retriever.search(args.query, args.k):
        print(f"[{score:.3f}] Q: {cust}\n         A: {reply}\n")
