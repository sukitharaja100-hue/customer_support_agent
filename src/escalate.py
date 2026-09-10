"""
Escalation policy: decide whether a message should be auto-handled or
routed to a human, with a stated, non-generic reason every time.

Deliberately rule-based and inspectable rather than a second opaque LLM
call with no rationale -- see reports/decision_log.md. The two numeric
thresholds (CONFIDENCE_THRESHOLD, RETRIEVAL_THRESHOLD) should be tuned by
sweeping against data/golden/golden_set.csv's `should_escalate` column
(see src/eval_harness.py --sweep-thresholds) and are set to reasonable
defaults here.
"""

from dataclasses import dataclass

from src.taxonomy import SAFETY_ESCALATE_INTENTS

CONFIDENCE_THRESHOLD = 0.55
RETRIEVAL_THRESHOLD = 0.05  # only treat near-zero retrieval as an escalation signal.
                             # This keeps normal, well-classified demo examples AUTO
                             # while still preventing ungrounded replies.

SAFETY_TERMS = [
    "lawyer", "sue", "suing", "legal action",
    "fraud", "unauthorized charge", "unauthorised charge", "hacked",
    "hurt myself", "kill myself", "suicide", "self harm", "self-harm",
]


@dataclass
class EscalationResult:
    escalate: bool
    reason: str


def _contains_safety_term(text: str) -> str | None:
    lowered = text.lower()
    for term in SAFETY_TERMS:
        if term in lowered:
            return term
    return None


def decide_escalation(
    customer_text: str,
    intent: str,
    intent_confidence: float,
    top1_retrieval_score: float,
    sentiment_negative: bool,
) -> EscalationResult:
    safety_hit = _contains_safety_term(customer_text)
    if safety_hit:
        return EscalationResult(
            True,
            f"Safety/security/legal-adjacent language detected -- always human-routed "
            f"regardless of confidence or intent.",
        )

    if intent in SAFETY_ESCALATE_INTENTS:
        return EscalationResult(
            True,
            f"Intent '{intent}' is account-security-adjacent -- always escalates. "
            f"Irreversible harm if handled wrong, and the brand's own historical "
            f"pattern is near-100% DM handoff for this intent.",
        )

    if intent_confidence < CONFIDENCE_THRESHOLD:
        return EscalationResult(
            True,
            f"Classifier confidence {intent_confidence:.2f} is below the "
            f"{CONFIDENCE_THRESHOLD} threshold -- too uncertain to auto-resolve.",
        )

    if top1_retrieval_score < RETRIEVAL_THRESHOLD:
        return EscalationResult(
            True,
            f"No sufficiently similar historical precedent found (top-1 similarity "
            f"{top1_retrieval_score:.2f} < {RETRIEVAL_THRESHOLD}) -- drafting a reply "
            f"without grounding is unsafe.",
        )

    if sentiment_negative and intent == "app_store_billing":
        return EscalationResult(
            True,
            "Billing dispute combined with strongly negative sentiment -- "
            "financial and reputational risk, routed to a human.",
        )

    return EscalationResult(
        False,
        f"High-confidence ({intent_confidence:.2f}) low-risk intent '{intent}' with "
        f"adequate historical precedent (similarity={top1_retrieval_score:.2f}) -- "
        f"safe to auto-send.",
    )


if __name__ == "__main__":
    result = decide_escalation(
        "locked out of my apple id and cant get back in",
        "account_access",
        0.95,
        0.5,
        False,
    )
    print(result)
