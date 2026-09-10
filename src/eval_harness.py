"""
The evaluation harness: runs every model in this repo over the golden set
and produces a single reports/metrics.json plus a couple of figures.

    python -m src.eval_harness

What it computes, always (no API key needed):
  - Trivial baseline intent accuracy/macro-F1
  - TF-IDF baseline intent accuracy/macro-F1 + confusion matrix
  - Escalation precision/recall/F1 using the TF-IDF baseline's confidence
    + TF-IDF retrieval score as inputs to src/escalate.py
  - A threshold sweep over CONFIDENCE_THRESHOLD x RETRIEVAL_THRESHOLD against
    the golden `should_escalate` column

What it additionally computes IF ANTHROPIC_API_KEY is set:
  - LLM few-shot classifier intent accuracy/macro-F1 + confusion matrix
  - LLM-drafted replies for every golden example, grounded via retrieval
  - LLM-judge rubric scores for every drafted reply
  - Judge-human agreement (Spearman correlation), IF
    data/golden/human_judge_ratings.csv has been filled in (copy
    human_judge_ratings_template.csv, score the drafts yourself blind,
    save as human_judge_ratings.csv -- see README)

Timing note: the LLM sections make one API call per golden example per
step (classify + reply + judge = up to 3 calls x len(golden)). At ~49
golden rows that is under 150 calls, comfortably inside a 15-minute budget
on Sonnet-class latency; scale accordingly if you grow the golden set.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src import classify_llm, judge as judge_mod, reply as reply_mod
from src.classify_baseline import TfidfBaseline, TrivialBaseline, weak_label_from_keywords
from src.escalate import CONFIDENCE_THRESHOLD, RETRIEVAL_THRESHOLD, decide_escalation
from src.retrieve import build_retriever
from src.taxonomy import INTENT_NAMES

RESULTS_DIR = "reports"
FIGURES_DIR = "reports/figures"


def _negative_sentiment(text: str) -> bool:
    from src.agent import detect_negative_sentiment
    return detect_negative_sentiment(text)


def run_intent_baselines(train_df, golden):
    results = {}

    train_df = train_df.copy()
    train_df["weak_label"] = train_df["cust_text_clean"].apply(weak_label_from_keywords)

    trivial = TrivialBaseline().fit(train_df["weak_label"])
    trivial_preds = trivial.predict(golden["cust_text"])
    results["trivial"] = {
        "accuracy": accuracy_score(golden["true_intent"], trivial_preds),
        "macro_f1": f1_score(golden["true_intent"], trivial_preds, average="macro", zero_division=0),
    }

    tfidf = TfidfBaseline().fit(train_df["cust_text_clean"], train_df["weak_label"])
    preds_confs = tfidf.predict_with_confidence(golden["cust_text"].tolist())
    tfidf_preds = [p for p, c in preds_confs]
    tfidf_confs = [c for p, c in preds_confs]
    results["tfidf"] = {
        "accuracy": accuracy_score(golden["true_intent"], tfidf_preds),
        "macro_f1": f1_score(golden["true_intent"], tfidf_preds, average="macro", zero_division=0),
    }

    cm = confusion_matrix(golden["true_intent"], tfidf_preds, labels=INTENT_NAMES)
    save_confusion_matrix(cm, INTENT_NAMES, "tfidf_confusion_matrix.png")

    llm_preds, llm_confs = None, None
    if classify_llm.available():
        llm_out = [classify_llm.classify(t) for t in golden["cust_text"]]
        llm_preds = [p for p, c in llm_out]
        llm_confs = [c for p, c in llm_out]
        results["llm"] = {
            "accuracy": accuracy_score(golden["true_intent"], llm_preds),
            "macro_f1": f1_score(golden["true_intent"], llm_preds, average="macro", zero_division=0),
        }
        cm_llm = confusion_matrix(golden["true_intent"], llm_preds, labels=INTENT_NAMES)
        save_confusion_matrix(cm_llm, INTENT_NAMES, "llm_confusion_matrix.png")
    else:
        results["llm"] = {"status": "SKIPPED -- ANTHROPIC_API_KEY not set"}

    return results, tfidf_preds, tfidf_confs, llm_preds, llm_confs


def save_confusion_matrix(cm, labels, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=120)
    plt.close(fig)


def run_escalation_eval(golden, preds, confs, retriever):
    y_true = golden["should_escalate"].astype(bool).tolist()
    y_pred = []
    reasons = []
    for text, intent, conf in zip(golden["cust_text"], preds, confs):
        retrieved = retriever.search(text, k=3)
        top1 = retrieved[0][2] if retrieved else 0.0
        sentiment_neg = _negative_sentiment(text)
        decision = decide_escalation(text, intent, conf, top1, sentiment_neg)
        y_pred.append(decision.escalate)
        reasons.append(decision.reason)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    false_negatives = [
        {"text": t, "reason": r}
        for t, p, gt, r in zip(golden["cust_text"], y_pred, y_true, reasons)
        if gt and not p
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_should_escalate": int(sum(y_true)),
        "n_predicted_escalate": int(sum(y_pred)),
        "false_negative_examples": false_negatives,
    }, reasons


def sweep_thresholds(golden, preds, confs, retriever):
    """Sweep confidence and retrieval thresholds, report precision/recall/F1
    at each point, so the chosen defaults in src/escalate.py are visibly
    justified rather than hand-picked."""
    y_true = golden["should_escalate"].astype(bool).tolist()
    top1_scores = [retriever.search(t, k=1)[0][2] if retriever.search(t, k=1) else 0.0 for t in golden["cust_text"]]

    rows = []
    for conf_thresh in [0.35, 0.45, 0.55, 0.65, 0.75]:
        for retr_thresh in [0.05, 0.10, 0.12, 0.20, 0.30]:
            y_pred = []
            for text, intent, conf, top1 in zip(golden["cust_text"], preds, confs, top1_scores):
                sentiment_neg = _negative_sentiment(text)
                # temporarily monkeypatch-free: replicate decide_escalation logic bounds via direct call
                decision = decide_escalation_with_thresholds(
                    text, intent, conf, top1, sentiment_neg, conf_thresh, retr_thresh
                )
                y_pred.append(decision)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            rows.append(
                {
                    "confidence_threshold": conf_thresh,
                    "retrieval_threshold": retr_thresh,
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1": round(f1, 3),
                }
            )
    return rows


def decide_escalation_with_thresholds(text, intent, conf, top1, sentiment_neg, conf_thresh, retr_thresh):
    """Same logic as src.escalate.decide_escalation but with swappable
    thresholds, for the sweep above -- kept separate so the production
    escalate.py stays simple and doesn't need a thresholds parameter
    threaded through every call site."""
    from src.escalate import _contains_safety_term
    from src.taxonomy import SAFETY_ESCALATE_INTENTS

    if _contains_safety_term(text):
        return True
    if intent in SAFETY_ESCALATE_INTENTS:
        return True
    if conf < conf_thresh:
        return True
    if top1 < retr_thresh:
        return True
    if sentiment_neg and intent == "app_store_billing":
        return True
    return False


def run_judge_and_generation(golden, retriever):
    if not classify_llm.available():
        return {"status": "SKIPPED -- ANTHROPIC_API_KEY not set"}, None

    drafts = []
    judge_scores = []
    for _, row in golden.iterrows():
        intent, _ = classify_llm.classify(row["cust_text"])
        retrieved = retriever.search(row["cust_text"], k=3)
        draft = reply_mod.draft_reply(row["cust_text"], intent, retrieved)
        score = judge_mod.judge_reply(row["cust_text"], draft, retrieved)
        drafts.append({"tweet_id": row["tweet_id"], "cust_text": row["cust_text"], "draft": draft})
        judge_scores.append({**score, "tweet_id": row["tweet_id"]})

    judge_df = pd.DataFrame(judge_scores)
    axes = ["groundedness", "correctness", "brand_voice", "actionability", "safety"]
    means = {ax: float(judge_df[ax].dropna().mean()) for ax in axes if ax in judge_df}

    drafts_df = pd.DataFrame(drafts)
    drafts_df.to_csv(os.path.join(RESULTS_DIR, "drafted_replies.csv"), index=False)
    judge_df.to_csv(os.path.join(RESULTS_DIR, "judge_scores.csv"), index=False)

    return {"mean_scores": means, "n_scored": len(judge_df)}, judge_df


def judge_human_agreement(judge_df):
    human_path = "data/golden/human_judge_ratings.csv"
    if judge_df is None or not os.path.exists(human_path):
        return {
            "status": (
                "NOT RUN -- fill in data/golden/human_judge_ratings.csv "
                "(copy human_judge_ratings_template.csv, score ~40 drafts "
                "yourself blind to the judge's scores using the identical "
                "rubric in src/judge.py) then re-run this harness."
            )
        }

    human_df = pd.read_csv(human_path)
    merged = judge_df.merge(human_df, on="tweet_id", suffixes=("_judge", "_human"))
    axes = ["groundedness", "correctness", "brand_voice", "actionability", "safety"]
    correlations = {}
    for ax in axes:
        human_col = f"human_{ax}"
        if human_col in merged and merged[human_col].notna().sum() > 2:
            valid = merged[[ax, human_col]].dropna()
            if len(valid) > 2:
                rho, p = spearmanr(valid[ax], valid[human_col])
                correlations[ax] = {"spearman_rho": float(rho), "p_value": float(p), "n": len(valid)}
    return {"n_examples_rated": len(merged), "spearman_by_axis": correlations}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_df = pd.read_parquet("data/interim/pairs.parquet")
    golden = pd.read_csv("data/golden/golden_set.csv")
    retriever = build_retriever("data/interim/pairs.parquet")

    print("[eval_harness] running intent classification baselines + LLM (if available)...")
    intent_results, tfidf_preds, tfidf_confs, llm_preds, llm_confs = run_intent_baselines(train_df, golden)

    print("[eval_harness] evaluating escalation policy...")
    esc_preds_source = (llm_preds, llm_confs) if llm_preds is not None else (tfidf_preds, tfidf_confs)
    escalation_results, _ = run_escalation_eval(golden, esc_preds_source[0], esc_preds_source[1], retriever)

    print("[eval_harness] sweeping escalation thresholds...")
    sweep = sweep_thresholds(golden, esc_preds_source[0], esc_preds_source[1], retriever)

    print("[eval_harness] running reply generation + LLM judge (if available)...")
    judge_results, judge_df = run_judge_and_generation(golden, retriever)

    print("[eval_harness] checking judge-human agreement...")
    agreement = judge_human_agreement(judge_df)

    metrics = {
        "golden_set_size": len(golden),
        "golden_set_sources": golden["source"].value_counts().to_dict(),
        "intent_classification": intent_results,
        "escalation": escalation_results,
        "escalation_threshold_sweep": sweep,
        "reply_quality_judge": judge_results,
        "judge_human_agreement": agreement,
        "config": {
            "confidence_threshold_used": CONFIDENCE_THRESHOLD,
            "retrieval_threshold_used": RETRIEVAL_THRESHOLD,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n[eval_harness] wrote {out_path}")
    print(json.dumps({
        "intent_classification": {k: v.get("accuracy", v.get("status")) for k, v in intent_results.items()},
        "escalation_f1": escalation_results["f1"],
        "reply_quality": judge_results,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
