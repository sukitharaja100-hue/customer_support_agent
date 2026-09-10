"""
Text cleaning for CS-Twitter data.

One function, clean_tweet(), used identically at ingest time, classifier
inference time, and eval time -- see reports/decision_log.md item on why
this consistency matters (train/serve skew is a real failure mode here).

Deliberate choices (see decision log for the full reasoning):
  - Casing and punctuation are PRESERVED for the shared cleaner. The LLM
    classifier/generator use casing as signal and reply drafts need to be
    able to quote the customer's own wording back accurately.
  - Lowercasing/stripping punctuation happens ONLY inside the TF-IDF
    baseline's own sklearn pipeline (classify_baseline.py), not here.
  - Brand mentions (e.g. @AppleSupport) are removed -- not signal.
  - Other @handles are replaced with a placeholder token.
  - t.co links are replaced with a placeholder token.
  - PII-looking substrings (long digit runs, emails) are redacted before
    anything is sent to a third-party LLM API.
"""

import re

try:
    import ftfy
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

BRAND_HANDLES = {"applesupport"}

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@(\w+)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_DIGIT_RE = re.compile(r"\b\d{6,}\b")  # order numbers, IMEIs, phone-like runs


def fix_encoding(text: str) -> str:
    if _HAS_FTFY:
        return ftfy.fix_text(text)
    return text


def redact_pii(text: str) -> str:
    """Scrub things that look like PII before the text leaves the machine
    via an LLM API call. Not a general-purpose PII scrubber -- scoped to
    what's actually likely to appear in these tweets."""
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _LONG_DIGIT_RE.sub("<ID>", text)
    return text


def normalise_mentions(text: str) -> str:
    def _sub(m):
        handle = m.group(1)
        if handle.lower() in BRAND_HANDLES:
            return ""
        return "@USER"
    return _MENTION_RE.sub(_sub, text)


def normalise_urls(text: str) -> str:
    return _URL_RE.sub("<URL>", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_tweet(text: str, scrub_pii: bool = True) -> str:
    if text is None:
        return ""
    text = fix_encoding(text)
    text = normalise_mentions(text)
    text = normalise_urls(text)
    if scrub_pii:
        text = redact_pii(text)
    text = collapse_whitespace(text)
    return text


def has_negative_emoji(text: str) -> bool:
    negative = {"😡", "😠", "😤", "😢", "😭", "🙄", "💢", "😞", "😔"}
    return any(ch in text for ch in negative)


def has_positive_emoji(text: str) -> bool:
    positive = {"😊", "🙂", "😀", "❤️", "👍", "🎉", "😍"}
    return any(ch in text for ch in positive)


def mentions_dm_handoff(brand_text: str) -> bool:
    lowered = brand_text.lower()
    return "dm" in lowered or "direct message" in lowered


if __name__ == "__main__":
    samples = [
        "@AppleSupport causing the reply to be disregarded 😡😡😡",
        "@105835 Your business means a lot to us. Please DM your name, zip code https://t.co/znUu1VJn9r",
    ]
    for s in samples:
        print(repr(s), "->", repr(clean_tweet(s)))
