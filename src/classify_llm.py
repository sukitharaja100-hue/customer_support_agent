"""Groq-backed few-shot intent classifier."""
import os, re
from groq import Groq
from src.taxonomy import INTENT_NAMES, fewshot_block, intent_definitions_block

_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
_client = None

SYSTEM_TEMPLATE = """You classify a customer support message sent to Apple's support account into exactly one of these intents:\n{intent_defs}\n\nReturn ONLY the intent label. If ambiguous, return other_uncategorised."""

def available():
    return bool(os.getenv("GROQ_API_KEY"))

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

def classify(text: str):
    client = _get_client()
    system = SYSTEM_TEMPLATE.format(intent_defs=intent_definitions_block())
    user = f"Few-shot examples:\n{fewshot_block()}\nCustomer message: \"{text}\"\nIntent:"
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0,
        max_completion_tokens=256,
    )
    raw = (response.choices[0].message.content or "").strip()
    cleaned = re.sub(r"[^a-z_]", "", raw.lower())
    if cleaned in INTENT_NAMES:
        return cleaned, 1.0
    for name in INTENT_NAMES:
        if name in cleaned:
            return name, 0.5
    return "other_uncategorised", 0.3
