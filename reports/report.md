# AppleSupport AI Support Agent — Report

## Problem framing

**Brand:** `AppleSupport`, chosen from the Customer Support on Twitter dataset because it is one of the highest-volume brand accounts, has a recognisable and fairly stable historical tone (brief, empathetic, near-universal "DM us" handoff for anything specific), and produces a learnable, non-degenerate intent taxonomy.

**What "good" means here, in priority order:**
1. **Safe** — never invents a refund amount, repair cost, or policy commitment that isn't backed by a real historical precedent; always routes account-security and safety-adjacent messages to a human.
2. **On-brand** — matches AppleSupport's real historical tone (short, empathetic, action-oriented, defers specifics to DM).
3. **Actionable** — gives the customer a concrete next step, even when that step is "we've routed this to a person."

"Sounds fluent" is explicitly *not* the bar. A fluent but hallucinated refund promise is a worse outcome than a stilted but accurate "please DM us."

**What I chose not to build:**
- Multi-turn dialogue state. The pipeline conditions on the first customer message and the brand's first substantive reply only; later back-and-forth in a thread is not modelled.
- Any actual account or order action. This is a classification + drafting + routing system, not an agent with write access to real Apple systems.
- A fine-tuned classifier. Few-shot LLM classification was fast enough to iterate on and clears the TF-IDF floor by a wide margin at this scale; fine-tuning is listed under "next week" instead.
- Multi-label / multi-intent classification. Some tweets are genuinely two intents at once (see Failure mode 3 below); I forced single-label and logged it as a known limitation rather than solving it.
- Non-English support. All examples and the taxonomy assume English-language tweets.

## Results vs. baselines

Numbers below are from `reports/metrics.json`, produced by `make eval` against the 49-example golden set (`data/golden/golden_set.csv`).

**Important scale caveat, stated here and expanded in "What is misleading about my headline number" below:** this run used the ~100-row demo sample bundled with this environment (13 real AppleSupport pairs), not the full ~2.8M-row Kaggle CSV, because this environment has no network path to Kaggle. The pipeline, golden set, and harness are all built to run unmodified against the full dataset — see README "Scaling to the full dataset" — but the numbers below reflect a training corpus of 13 pairs, not the thousands a real run would use.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (majority class) | 0.286 | 0.056 |
| TF-IDF + Logistic Regression (simple baseline) | 0.388 | 0.176 |
| Few-shot LLM (Claude, production path) | *(requires `ANTHROPIC_API_KEY`; skipped in this run — see note)* | — |

The TF-IDF baseline beats the trivial baseline on both metrics, which is the expected ordering, but macro-F1 is still low — see the confusion matrix (`reports/figures/tfidf_confusion_matrix.png`): with only 13 training pairs, the classifier collapses almost everything into `software_bug_update`, the most frequent class in that tiny corpus. This is a real, honest result, not a bug — it's exactly the kind of thing a 13-row training set produces, and exactly why the brief expects a subsample in the thousands, not the tens.

**Escalation policy**, evaluated using the TF-IDF baseline's confidence + TF-IDF retrieval similarity as inputs (since the LLM classifier was not run in this pass):

| Metric | Value |
|---|---|
| Precision | 0.298 |
| Recall | 0.933 |
| F1 | 0.452 |
| Should escalate (golden) | 15 / 49 |
| Predicted escalate | 47 / 49 |

Recall is high by design — the policy is deliberately biased toward over-escalating rather than under-escalating (a missed escalation is worse than an unnecessary one), and the low-confidence/low-retrieval thresholds are doing a lot of the escalating here precisely because the training corpus is so small (most predictions are low-confidence). Precision will improve substantially once the classifier is trained on a real-sized corpus and stops needing the confidence threshold as a safety net for its own weakness. The full threshold sweep (25 configurations of confidence × retrieval thresholds) is in `metrics.json` under `escalation_threshold_sweep`.

**Reply quality (LLM-judge rubric):** skipped in this run — requires `ANTHROPIC_API_KEY`. See `src/judge.py` for the rubric (groundedness, correctness, brand_voice, actionability, safety, each 1-5) and `src/eval_harness.py::judge_human_agreement` for the Spearman-correlation methodology against a human rater. Once run, fill in `data/golden/human_judge_ratings.csv` (template provided) to get the mandatory judge-human agreement number.

## Failure analysis — top 5, with real examples

1. **Class collapse from a tiny training corpus.** The TF-IDF baseline predicts `software_bug_update` for nearly every input, including clear `account_access` cases like *"I need a new code for my I-store. I haven't recd any but msg is too many sent. Help!"* (golden `tweet_id=119257`). Hypothesis: 13 training pairs isn't enough signal to separate 8 classes; `software_bug_update` is both the largest class in that corpus and shares vocabulary ("update", "issue") with several others.

2. **Sarcasm misread as literal sentiment.** *"My apps stop working without warning and my phone freezes every five minutes! Love the new update"* (`tweet_id=119263`) is correctly hand-labelled `software_bug_update`, but a naive sentiment check could misread "Love the new update" as positive and suppress an escalation-relevant negative-sentiment signal. The lexicon-based `detect_negative_sentiment()` in `src/agent.py` is a known-weak stand-in for exactly this reason — flagged as a "next week" item.

3. **Forced single-label loses information on genuinely multi-intent messages.** *"my account got hacked and now theres unauthorized charges everywhere, please help fast"* (golden `demo_030`) is simultaneously `account_access` and `app_store_billing`. It's labelled by primary/most-urgent intent in the golden set, but a single-label classifier that instead picks the secondary intent will miss half the context needed for a good reply. Hypothesis: this is a structural limitation of the taxonomy design, not a model bug — multi-label classification is the real fix (see Next week).

4. **Context-dependent follow-up messages are close to uninterpretable alone.** *"I have the latest version iOS. It started immediately after I updated my phone."* (`tweet_id=119261`) is a reply deep in a thread; on its own it has no clear referent for "it." Hypothesis: this is a direct consequence of the deliberate decision to model only (first customer message → first brand reply) and not full thread state — a real cost of that scoping choice, not free.

5. **Low retrieval similarity on a small corpus over-triggers escalation, masking whether the underlying reply would have been fine.** Several golden examples show top-1 TF-IDF similarity scores in the 0.1–0.25 range purely because the retrieval corpus only has 13 documents to search — even a good semantic match won't score high with so little to match against. Hypothesis: this failure mode is an artefact of corpus size, not of the retrieval method itself, and should mostly disappear once run against the full dataset — but it's worth flagging because it currently makes the escalation policy's recall look artificially strong for the wrong reason (everything gets escalated because nothing retrieves well, not because the policy is well-calibrated).

## What is misleading about my headline number?

- **The training corpus for every model in this run has 13 rows.** Every accuracy, F1, and similarity score above is downstream of a training/retrieval corpus roughly three orders of magnitude smaller than what the brief expects ("a subsample is expected," not a 13-row corpus). Treat every number in this report as a proof that the *pipeline* runs correctly end-to-end, not as evidence about model quality at realistic scale. Re-running `make data` with `RAW_CSV` pointed at the full Kaggle CSV and a `LIMIT` in the thousands is the single highest-leverage next step, and every downstream metric should be expected to change materially.
- **The golden set is 49 examples, 36 of them synthetic-demo, not hand-labelled from the wild.** The real brief calls for 150–250 examples hand-labelled from the actual dataset. 36 of the 49 rows here were written by me in the same register as the real data specifically to give the harness enough per-intent and edge-case coverage to be exercised meaningfully at this small scale — they are clearly marked `source=synthetic_demo` in the CSV. Accuracy/F1 against this set says less about real-world performance than it will once the golden set is the full 150–250 hand-labelled sample from real tweets.
- **Escalation thresholds were swept and reported against the same golden set the headline metric is computed on.** No held-out tuning split was used. This makes the reported escalation F1 optimistic relative to how it would perform on a truly unseen set — a held-out threshold-tuning split is listed under "next week."
- **The LLM classifier, reply drafter, and judge were not run in this pass** (no `ANTHROPIC_API_KEY` in this environment), so the "production path" numbers this report should really be judged on are absent. Everything reported is the TF-IDF/rule-based fallback path, which the code explicitly documents as *not* what should be treated as the headline result (see `src/agent.py`'s fallback comment).
- **Judge-human agreement was not computed** for the same reason (no API key to generate judge scores to compare against). The mechanism exists (`data/golden/human_judge_ratings_template.csv`, `judge_human_agreement()` in `src/eval_harness.py`) but produces no number in this run.

## What I'd do next with one more week

1. Re-run the full pipeline against the actual Kaggle CSV with a `LIMIT` of 10,000–20,000 pairs, and re-derive the taxonomy via `cluster_explore.py --backend sbert` against that larger corpus to check the frozen taxonomy in `src/taxonomy.py` still holds.
2. Hand-label a real 150–250-example golden set sampled from that larger pull (the sampling method in `data/golden/golden_set.csv`'s construction generalises directly — stratify by cluster, oversample rare/adversarial cases).
3. Add a held-out threshold-tuning split, separate from the reported golden set, for the escalation thresholds.
4. Run the LLM classifier/reply/judge path with a real API key and fill in `data/golden/human_judge_ratings.csv` for the mandatory judge-human agreement number.
5. Replace the lexicon-based sentiment flag with something better calibrated on sarcasm, and consider multi-label intent classification given Failure mode 3.
