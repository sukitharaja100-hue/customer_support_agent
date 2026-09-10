"""
Two baselines for intent classification, in increasing order of sophistication:

1. TrivialBaseline: always predicts the single most frequent intent.
   This is the floor -- any real model must beat this by a meaningful margin.

2. TfidfBaseline: TF-IDF + Logistic Regression, trained on CLUSTER-DERIVED
   WEAK LABELS (from src/cluster_explore.py mapped onto the frozen taxonomy),
   never on the golden set. The golden set is reserved for evaluation only,
   for every model in this repo -- see reports/decision_log.md.

Usage (train + evaluate against golden set):
    python -m src.classify_baseline --pairs data/interim/pairs.parquet \
        --golden data/golden/golden_set.csv
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.taxonomy import INTENT_NAMES


class TrivialBaseline:
    """Always predicts the majority class seen at fit time."""

    def __init__(self):
        self.majority_label = None

    def fit(self, y):
        self.majority_label = pd.Series(y).value_counts().idxmax()
        return self

    def predict(self, X):
        return [self.majority_label] * len(X)


class TfidfBaseline:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), lowercase=True, stop_words="english")
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        return self.clf.predict(X)

    def predict_with_confidence(self, texts):
        """Returns (label, confidence) per text, confidence = max class probability."""
        X = self.vec.transform(texts)
        probs = self.clf.predict_proba(X)
        labels = self.clf.classes_[probs.argmax(axis=1)]
        confidences = probs.max(axis=1)
        return list(zip(labels, confidences))

    def save(self, path):
        joblib.dump({"vec": self.vec, "clf": self.clf}, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        data = joblib.load(path)
        obj.vec, obj.clf = data["vec"], data["clf"]
        return obj


def weak_label_from_keywords(text: str) -> str:
    """
    Cheap stand-in for 'cluster label mapped to taxonomy name'.

    In a full run, this comes from src/cluster_explore.py: cluster each
    training row, manually name each cluster once (Section 5 of the build
    guide), then map cluster_id -> intent_name and attach it to every row
    in that cluster. That mapping step needs a human in the loop, so it is
    not automatable end-to-end -- this function is a deterministic,
    keyword-based approximation of that same mapping, good enough to make
    `make data` runnable without a human clustering pass baked into CI,
    but it is NOT what should be described in the report as the taxonomy
    discovery method. Use cluster_explore.py's actual output for that.
    """
    t = text.lower()
    if any(k in t for k in ["charged twice", "refund", "subscription", "cancel", "charge on my card", "unauthorized charge", "unauthorised charge"]):
        return "app_store_billing"
    if any(k in t for k in ["battery", "won't turn on", "drains", "charging port", "won't charge"]):
        return "battery_power"
    if any(k in t for k in ["apple id", "icloud", "2fa", "locked out", "sign in", "disabled"]):
        return "account_access"
    if any(k in t for k in ["crack", "screen", "water damage", "warranty", "applecare", "repair"]):
        return "device_damage_repair"
    if any(k in t for k in ["update", "ios", "crash", "freeze", "bug", "wifi disconnect"]):
        return "software_bug_update"
    if any(k in t for k in ["order", "delivery", "shipped", "tracking"]):
        return "order_shipping"
    if any(k in t for k in ["thank", "thanks", "great job", "love this", "appreciate"]):
        return "general_praise_or_vent"
    return "other_uncategorised"


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=INTENT_NAMES, zero_division=0, output_dict=True)
    print(f"\n=== {name} ===")
    print(f"accuracy={acc:.3f}  macro_f1={f1:.3f}")
    return {"name": name, "accuracy": acc, "macro_f1": f1, "report": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/interim/pairs.parquet")
    ap.add_argument("--golden", default="data/golden/golden_set.csv")
    ap.add_argument("--model_out", default="data/interim/tfidf_baseline.joblib")
    ap.add_argument("--metrics_out", default="reports/baseline_metrics.json")
    args = ap.parse_args()

    train_df = pd.read_parquet(args.pairs)
    train_df["weak_label"] = train_df["cust_text_clean"].apply(weak_label_from_keywords)

    golden = pd.read_csv(args.golden)

    results = {}

    trivial = TrivialBaseline().fit(train_df["weak_label"])
    trivial_preds = trivial.predict(golden["cust_text"])
    results["trivial"] = evaluate("Trivial (majority class)", golden["true_intent"], trivial_preds)

    tfidf = TfidfBaseline().fit(train_df["cust_text_clean"], train_df["weak_label"])
    tfidf_preds = tfidf.predict(golden["cust_text"].apply(lambda t: t))
    results["tfidf"] = evaluate("TF-IDF + Logistic Regression", golden["true_intent"], tfidf_preds)
    tfidf.save(args.model_out)

    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[classify_baseline] wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
