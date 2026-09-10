"""
A handful of unit tests covering the parts of the pipeline that are cheap
to test without an LLM API key: cleaning, escalation rules, and the
thread-reconstruction join. Run with:

    pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import clean_tweet, normalise_mentions, normalise_urls
from src.escalate import decide_escalation
from src.classify_baseline import TrivialBaseline, weak_label_from_keywords


def test_clean_removes_brand_mention():
    out = clean_tweet("@AppleSupport my phone is broken")
    assert "@AppleSupport" not in out
    assert "@applesupport" not in out.lower()


def test_clean_normalises_other_mentions():
    out = normalise_mentions("@105835 hello")
    assert out == "@USER hello"


def test_clean_normalises_urls():
    out = normalise_urls("check this https://t.co/abc123 out")
    assert "<URL>" in out
    assert "t.co" not in out


def test_escalation_always_fires_on_safety_keyword():
    result = decide_escalation(
        "if this isn't fixed i'm calling my lawyer",
        "software_bug_update",
        0.99,
        0.9,
        False,
    )
    assert result.escalate is True
    assert "safety" in result.reason.lower() or "legal" in result.reason.lower()


def test_escalation_always_fires_on_account_access():
    result = decide_escalation(
        "locked out of my apple id",
        "account_access",
        0.99,
        0.9,
        False,
    )
    assert result.escalate is True


def test_escalation_low_confidence_escalates():
    result = decide_escalation(
        "my phone is slow",
        "software_bug_update",
        0.2,
        0.9,
        False,
    )
    assert result.escalate is True
    assert "confidence" in result.reason.lower()


def test_escalation_low_retrieval_escalates():
    result = decide_escalation(
        "my phone is slow",
        "software_bug_update",
        0.9,
        0.01,
        False,
    )
    assert result.escalate is True
    assert "precedent" in result.reason.lower() or "similarity" in result.reason.lower()


def test_escalation_safe_case_does_not_escalate():
    result = decide_escalation(
        "my battery drains fast since the update",
        "battery_power",
        0.9,
        0.9,
        False,
    )
    assert result.escalate is False


def test_trivial_baseline_predicts_majority_class():
    labels = ["a", "a", "a", "b"]
    baseline = TrivialBaseline().fit(labels)
    preds = baseline.predict(["x", "y", "z"])
    assert all(p == "a" for p in preds)


def test_weak_label_keywords_are_sane():
    assert weak_label_from_keywords("my battery drains so fast") == "battery_power"
    assert weak_label_from_keywords("locked out of my apple id") == "account_access"
    assert weak_label_from_keywords("charged twice for this app") == "app_store_billing"
