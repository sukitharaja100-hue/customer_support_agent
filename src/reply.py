"""Groq-backed grounded reply generation."""
import os
from groq import Groq
_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
_client = None
REPLY_SYSTEM = """You draft a reply from Apple's official support account. Be brief (1-3 sentences), empathetic and action-oriented. Direct the customer to DM for account/order/device-specific details. Do not invent policy, refund amounts, repair costs, or timelines. Use the historical examples only as style/grounding references. Output only the reply."""

def available(): return bool(os.getenv("GROQ_API_KEY"))
def _get_client():
    global _client
    if _client is None: _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

def draft_reply(customer_text, intent, retrieved):
    ref_block = "\n".join(f'- Similar case: "{c}" -> Brand reply: "{r}" (similarity={s:.2f})' for c,r,s in retrieved) or "(no sufficiently similar historical case found)"
    user = f"Customer intent: {intent}\nCustomer message: \"{customer_text}\"\nHistorical references:\n{ref_block}\n\nDraft the reply."
    response = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role":"system","content":REPLY_SYSTEM},{"role":"user","content":user}],
        temperature=0.2,
        max_completion_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()
