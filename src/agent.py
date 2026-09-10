"""
SupportAgent: the single glue point calling classify -> retrieve -> reply ->
escalate for one incoming customer message. This is the interface both
eval_harness.py (batch, over the golden set) and this file's __main__ (a
one-off CLI demo) call -- kept deliberately framework-free and legible so
every step is inspectable and modifiable live.

Usage:
    python -m src.agent --text "my battery drains so fast since the update"
"""

import argparse

from src.clean import clean_tweet, has_negative_emoji
from src.escalate import decide_escalation
from src.retrieve import build_retriever


NEGATIVE_LEXICON = {
    "hate", "sucks", "suck", "worst", "disgrace", "horrible", "terrible",
    "furious", "angry", "frustrated", "annoyed", "ridiculous", "unacceptable",
    "nightmare", "done", "fed up",
}


def detect_negative_sentiment(text: str) -> bool:
    """Cheap lexicon-based sentiment flag, used only as an input to the
    escalation policy's billing+sentiment rule. Not a general sentiment
    classifier -- swap for a proper model if this becomes a bottleneck;
    logged as a 'next week' item in reports/report.md."""
    lowered = text.lower()
    if has_negative_emoji(text):
        return True
    return any(word in lowered for word in NEGATIVE_LEXICON)


class SupportAgent:
    def __init__(self, retriever, classify_fn, reply_fn, use_llm: bool):
        self.retriever = retriever
        self.classify_fn = classify_fn
        self.reply_fn = reply_fn
        self.use_llm = use_llm

    def handle(self, raw_text: str) -> dict:
        text = clean_tweet(raw_text)

        intent, confidence = self.classify_fn(text)
        retrieved = self.retriever.search(text, k=3)
        top1_score = retrieved[0][2] if retrieved else 0.0
        sentiment_negative = detect_negative_sentiment(text)

        decision = decide_escalation(text, intent, confidence, top1_score, sentiment_negative)

        draft = None
        if self.use_llm:
            # Always generate a grounded AI draft.
            # AUTO cases may use it automatically; ESCALATE cases show the
            # draft to the human reviewer for approval/editing.
            draft = self.reply_fn(text, intent, retrieved)

        return {
            "input": raw_text,
            "cleaned_input": text,
            "intent": intent,
            "intent_confidence": confidence,
            "top1_retrieval_score": top1_score,
            "grounded_on": retrieved,
            "sentiment_negative": sentiment_negative,
            "draft_reply": draft,
            "escalate": decision.escalate,
            "escalation_reason": decision.reason,
        }


def build_agent(pairs_parquet="data/interim/pairs.parquet", backend="tfidf"):
    from src import classify_llm, reply as reply_mod
    from src.classify_baseline import TfidfBaseline, weak_label_from_keywords

    retriever = build_retriever(pairs_parquet, backend=backend)

    if classify_llm.available():
        classify_fn = classify_llm.classify
        reply_fn = reply_mod.draft_reply
        use_llm = True
    else:
        # Graceful degradation: no API key -> use the TF-IDF baseline for
        # intent (with a flat 0.5 confidence, since LogisticRegression's
        # predict_proba isn't wired through here) and a canned templated
        # reply instead of a generated one. Good enough to exercise the
        # full pipeline end-to-end without a key; NOT what should be
        # reported as headline numbers -- see README.
        import pandas as pd

        train_df = pd.read_parquet(pairs_parquet)
        train_df["weak_label"] = train_df["cust_text_clean"].apply(weak_label_from_keywords)
        baseline = TfidfBaseline().fit(train_df["cust_text_clean"], train_df["weak_label"])

        def classify_fn(text):
            (pred, conf) = baseline.predict_with_confidence([text])[0]
            return pred, float(conf)

        def reply_fn(text, intent, retrieved):
            if retrieved:
                return retrieved[0][1]  # canned: most similar historical reply, verbatim
            return "Thanks for reaching out -- please DM us so we can help further."

        use_llm = False

    return SupportAgent(retriever, classify_fn, reply_fn, use_llm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--pairs", default="data/interim/pairs.parquet")
    args = ap.parse_args()

    agent = build_agent(args.pairs)
    result = agent.handle(args.text)

    print(f"Intent:       {result['intent']} (confidence={result['intent_confidence']:.2f})")
    print(f"Escalate:     {result['escalate']}")
    print(f"Reason:       {result['escalation_reason']}")
    if result["draft_reply"]:
        print(f"Draft reply:  {result['draft_reply']}")
    print(f"Top-1 retrieval score: {result['top1_retrieval_score']:.3f}")


if __name__ == "__main__":
    main()
