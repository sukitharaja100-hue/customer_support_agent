"""
Taxonomy discovery: cluster customer messages to check that the intent
labels in src/taxonomy.py actually match the shape of the data, rather
than being asserted from intuition.

DEFAULT BACKEND: TF-IDF + KMeans (scikit-learn only, fully offline, no
model weights to download). This was a deliberate choice for this repo --
see reports/decision_log.md -- so `make data` and this exploration step
never depend on internet access to a model hub, which matters for the
"reproduce in under 15 minutes on a clean checkout" requirement.

OPTIONAL HIGHER-QUALITY BACKEND: sentence-transformers embeddings +
HDBSCAN density clustering, which generally produces cleaner, more
semantically coherent clusters on short noisy text than TF-IDF+KMeans.
Enable it with --backend sbert if you have internet access to pull
`all-MiniLM-L6-v2` the first time it runs (`pip install sentence-transformers hdbscan`).

Usage:
    python -m src.cluster_explore --pairs data/interim/pairs.parquet --k 8
    python -m src.cluster_explore --pairs data/interim/pairs.parquet --backend sbert
"""

import argparse
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_tfidf(texts, k):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    k = min(k, max(2, X.shape[0] // 3)) if X.shape[0] < k * 3 else k
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, vec, km


def cluster_sbert_hdbscan(texts):
    from sentence_transformers import SentenceTransformer
    import hdbscan

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=True)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(texts) // 40))
    labels = clusterer.fit_predict(emb)
    return labels


def print_clusters(df, labels, n_examples=5):
    df = df.copy()
    df["cluster"] = labels
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        print(f"\n=== Cluster {cluster_id} (n={len(sub)}) ===")
        for txt in sub["cust_text_clean"].head(n_examples):
            print(f"  - {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/interim/pairs.parquet")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters (tfidf backend only)")
    ap.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    args = ap.parse_args()

    df = pd.read_parquet(args.pairs)
    if len(df) < 10:
        print(
            f"[cluster_explore] WARNING: only {len(df)} rows available -- "
            "this is the bundled demo sample, not enough to discover a "
            "reliable taxonomy from scratch. The frozen taxonomy in "
            "src/taxonomy.py was derived from a larger pull; re-run this "
            "script against the full Kaggle dataset to validate it, or to "
            "derive your own taxonomy for a different brand.",
            file=sys.stderr,
        )

    texts = df["cust_text_clean"].tolist()

    if args.backend == "tfidf":
        labels, _, _ = cluster_tfidf(texts, args.k)
    else:
        labels = cluster_sbert_hdbscan(texts)

    print_clusters(df, labels)


if __name__ == "__main__":
    main()
