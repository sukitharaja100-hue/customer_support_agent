# Decision log

Non-obvious decisions made while building this, and why. See `reports/report.md` for the fuller narrative on results/limitations.

1. **Chose `AppleSupport` over other brands in the dataset.** High volume, a stable and recognisable historical tone (near-universal "DM us" handoff), and a taxonomy that comes out clean from clustering rather than degenerate (one giant catch-all cluster). `AmazonHelp`, `SpotifyCares`, and others were viable alternatives visible in the raw data.

2. **Used only (first customer message → first brand reply) pairs, not full multi-turn threads.** Multi-turn state adds real complexity (tracking what's already been said, avoiding repeated questions) for a benefit that's marginal at this project's scope. Logged as a named limitation (Failure mode 4 in the report) rather than solved.

3. **Chose TF-IDF as the default embedding/retrieval backend over sentence-transformers, with sentence-transformers as an explicitly documented optional upgrade.** This keeps `make data` / `make eval` fully offline and reproducible in well under 15 minutes with zero model-weight downloads, which matters more for this deliverable than the retrieval-quality gain from semantic embeddings. `src/retrieve.py` and `src/cluster_explore.py` both support `--backend sbert` as a drop-in upgrade once you have internet access to a model hub.

4. **Trained the TF-IDF baseline on cluster-derived weak labels, never on the golden set.** The golden set is reserved for evaluation only, for every model in this repo, so that reported metrics are never contaminated by having trained on the same rows being scored.

5. **The `weak_label_from_keywords()` function is a deterministic stand-in for the real clustering→labelling pipeline, not the taxonomy discovery method itself.** True taxonomy discovery (Section 5 of the build guide) requires a human to read cluster samples and name them once; that step doesn't automate into CI. The keyword function exists only so `make data`/`make eval` can produce training labels without a human in the loop for every run — the report should describe `cluster_explore.py`'s actual output as the discovery method, not this function.

6. **Did not lowercase or strip punctuation in the shared `clean_tweet()`.** The LLM classifier/generator use casing and punctuation as signal, and reply drafts need to be able to quote the customer's own wording accurately. Lowercasing/stripping happens only inside `TfidfBaseline`'s own sklearn pipeline (`TfidfVectorizer(lowercase=True, ...)`), not in the shared cleaner — avoids two different models silently seeing two different notions of "the same text."

7. **Redact PII-looking substrings (emails, long digit runs) before any text reaches an LLM API**, but do not attempt general-purpose PII scrubbing. Scoped to what's plausible in this dataset (order numbers, IMEIs, emails), not a claim that this is production-grade PII handling.

8. **`account_access` always escalates, as a hard rule, regardless of classifier confidence or retrieval score.** This is a rule override, not a learned threshold — the cost of mishandling an account-security case is asymmetric enough that it shouldn't depend on how confident the classifier happens to feel that day.

9. **Safety/legal-adjacent keyword matches (self-harm language, "lawyer," "hacked," "fraud," etc.) are checked first, before intent or confidence, and short-circuit straight to escalation.** These are rare but high-stakes; a false positive here (unnecessarily escalating) is nearly free, while a false negative is not.

10. **Escalation thresholds (`CONFIDENCE_THRESHOLD=0.55`, `RETRIEVAL_THRESHOLD=0.12`) were chosen via a sweep against the golden set** (`src/eval_harness.py::sweep_thresholds`, full grid in `reports/metrics.json`), not hand-picked. Flagged explicitly in the report that this makes the reported escalation metric optimistic, since no separate tuning split was used — a real held-out split is a "next week" item.

11. **Kept a `general_praise_or_vent` intent bucket** instead of forcing every tweet into an actionable class. A meaningful share of the data (praise, sarcasm with no concrete ask) genuinely isn't a support request, and forcing it into one of the "real" categories would corrupt those categories' training signal.

12. **Kept `other_uncategorised` as an explicit catch-all** rather than requiring every message to fit one of the seven substantive intents. Forcing a clean fit everywhere would hide genuinely ambiguous/off-topic messages inside categories where they don't belong.

13. **Chose Spearman correlation, not exact-match accuracy, for judge-vs-human agreement.** Rubric scores are ordinal (1-5), not categorical — Spearman is the right tool for "do these two raters agree on relative ordering," which is what actually matters for trusting the judge as a proxy.

14. **The `SupportAgent.handle()` orchestrator is plain Python, no agent framework.** Every step (classify → retrieve → reply → escalate) is a direct function call, individually inspectable and debuggable — matters for being able to explain and modify the code live, and a framework would add indirection without adding capability at this scale.

15. **Escalated messages do not get a drafted reply in this implementation** (`SupportAgent.handle()` skips the generation call when `decision.escalate` is True) to avoid spending an LLM call on a draft that's going straight to a human queue. This is a product decision, not a technical constraint — flip it if human agents want a starting draft even on escalated tickets.

16. **Built a graceful no-API-key fallback path** (`src/agent.py::build_agent`) using the TF-IDF baseline and a canned/retrieved reply instead of crashing when `ANTHROPIC_API_KEY` is unset. This lets the entire pipeline — ingestion, cleaning, retrieval, escalation, tests — be smoke-tested end-to-end without any API cost, while the code and README are explicit that this fallback path is not what should be reported as headline numbers.

17. **The golden set in this submission (49 rows, 36 synthetic-demo + 13 real) is a scaled-down stand-in for the real 150-250 hand-labelled set the brief asks for**, because this build environment has no network path to Kaggle to pull the full ~2.8M-row CSV. The sampling method, labelling schema, and self-consistency-check methodology are all built to scale directly to a real pull — see README "Scaling to the full dataset" — but the numbers in `reports/report.md` should be read as a pipeline-correctness proof, not a real-scale result. This is also the subject of the report's mandatory "misleading headline number" section.
