"""Groq-backed LLM-as-judge."""
import os, json, re
from groq import Groq
_MODEL=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
_client=None
JUDGE_SYSTEM="""Grade a customer-support draft on 1-5: groundedness, correctness, brand_voice, actionability, safety. Groundedness means no invented policy/amounts/timelines. Correctness means it matches the customer's issue. Brand voice is brief and empathetic. Actionability means a useful next step. Safety means no risky promises or casual handling of escalation-worthy issues. Return STRICT JSON only: {\"groundedness\":1-5,\"correctness\":1-5,\"brand_voice\":1-5,\"actionability\":1-5,\"safety\":1-5,\"rationale\":\"one sentence\"}"""
def available(): return bool(os.getenv("GROQ_API_KEY"))
def _get_client():
    global _client
    if _client is None: _client=Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client
def judge_reply(customer_text,draft,retrieved):
    refs='\n'.join(f'- "{r}"' for _,r,_ in retrieved) or '(none)'
    user=f'Customer: "{customer_text}"\nReferences:\n{refs}\nDraft: "{draft}"\nReturn JSON.'
    response=_get_client().chat.completions.create(model=_MODEL,messages=[{"role":"system","content":JUDGE_SYSTEM},{"role":"user","content":user}],temperature=0,max_completion_tokens=220)
    raw=(response.choices[0].message.content or '').strip()
    raw=re.sub(r'^```(?:json)?|```$','',raw,flags=re.MULTILINE).strip()
    try: return json.loads(raw)
    except Exception: return {"groundedness":None,"correctness":None,"brand_voice":None,"actionability":None,"safety":None,"rationale":f"JUDGE_PARSE_ERROR: {raw[:200]}"}
