# AppleSupport AI Support Agent

An AI support agent for Apple's Twitter support account (`@AppleSupport`), built from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. Given an incoming customer message, it:

1. **Classifies** it into one of 8 intents defined from the data itself (see `src/taxonomy.py` and "Taxonomy discovery" below).
2. **Drafts a reply**, grounded in retrieved examples of how AppleSupport has actually resolved similar issues historically (retrieval-augmented generation, `src/reply.py`).
3. **Decides** whether to auto-send that draft or escalate to a human, with an explicit, stated reason every time (`src/escalate.py`).

Everything is evaluated against a hand-built golden set (`data/golden/golden_set.csv`) using an automated metrics + LLM-as-judge harness (`src/eval_harness.py`).

**No Banking77 or any secondary dataset is used anywhere in this repo** — the taxonomy and every labelled example come from the Customer Support on Twitter data.

---

## ⚠️ Important scale note, read this first

This repo ships with a **~100-row demo sample** (`data/raw/twcs_sample.csv`) instead of the full ~2.8M-row Kaggle CSV, because the environment this was built in has no network path to Kaggle. The entire pipeline — ingestion, taxonomy discovery, baselines, retrieval, escalation, the eval harness — runs correctly end-to-end against this sample (that's what "reproduce in under 15 minutes" below verifies), but **every metric you'll see from a default run is a proof of pipeline correctness, not a real-scale result.** See `reports/report.md`'s "What is misleading about my headline number" section for the full accounting, and "Scaling to the full dataset" below for how to get real numbers.

---

## Setup (2 minutes)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY -- optional, see below
export $(cat .env | xargs)        # or `source .env` depending on your shell
```

**Running without an API key:** everything below still runs. Intent classification falls back to the TF-IDF baseline, and reply drafting falls back to returning the most similar historical reply verbatim instead of generating a new one. This is enough to smoke-test the whole pipeline for free, but it is **not** the path the report's headline numbers should come from — set `ANTHROPIC_API_KEY` for the real few-shot LLM classifier, grounded generation, and LLM-judge scoring.

## Reproduce the headline results (under 15 minutes, including a fresh pip install)

```bash
make data       # ~5 sec  -- ingest + clean + reconstruct pairs from the bundled sample
make golden     # instant -- confirms data/golden/golden_set.csv is present, prints a summary
make baseline   # ~1 sec  -- trivial + TF-IDF baselines vs. golden set
make eval       # ~1 sec  without an API key; a few minutes with one (LLM classify+reply+judge per golden row)
make test       # ~1 sec  -- unit tests for cleaning, escalation rules, baselines
make report     # prints headline metrics from reports/metrics.json
```

What each target produces:
- `reports/metrics.json` — every metric described below, in one file
- `reports/baseline_metrics.json` — the two intent baselines in isolation
- `reports/figures/tfidf_confusion_matrix.png` (and `llm_confusion_matrix.png` if an API key was set)
- `reports/drafted_replies.csv`, `reports/judge_scores.csv` — only produced with an API key set

## Try the agent directly

```bash
python3 -m src.agent --text "my battery drains so fast since the update"
python3 -m src.agent --text "locked out of my apple id and cant get back in"
python3 -m src.agent --text "if this isnt fixed by tomorrow im calling my lawyer"
```

Each call prints the classified intent + confidence, the escalation decision + stated reason, the top-1 retrieval similarity score, and (if an API key is set) a Groq-drafted reply. Escalated cases are held for human review; AUTO cases can be auto-sent.

## Repository structure

```text
apple-support-agent/
├── README.md                     # you are here
├── requirements.txt
├── .env.example
├── Makefile
├── data/
│   ├── raw/twcs_sample.csv       # bundled demo sample (~100 rows) -- see scale note above
│   ├── interim/                  # generated: cleaned/joined pairs, trained baseline model
│   └── golden/
│       ├── golden_set.csv                        # 49-example golden evaluation set (see below)
│       └── human_judge_ratings_template.csv      # copy -> human_judge_ratings.csv and fill in
├── src/
│   ├── ingest.py                 # DuckDB filter + thread reconstruction for one brand
│   ├── clean.py                  # text cleaning / PII scrub, shared everywhere
│   ├── taxonomy.py               # frozen 8-intent taxonomy + few-shot exemplars
│   ├── cluster_explore.py        # taxonomy discovery (TF-IDF+KMeans default, sbert+HDBSCAN optional)
│   ├── classify_baseline.py      # trivial + TF-IDF/LogReg baselines
│   ├── classify_llm.py           # few-shot LLM intent classifier (needs API key)
│   ├── retrieve.py               # grounding index (TF-IDF default, sbert+FAISS optional)
│   ├── reply.py                  # grounded reply drafting (needs API key)
│   ├── escalate.py               # rule-based escalation policy with stated reasons
│   ├── agent.py                  # SupportAgent orchestrator (classify -> retrieve -> reply -> escalate)
│   ├── judge.py                  # LLM-as-judge rubric scorer (needs API key)
│   └── eval_harness.py           # runs everything above over the golden set -> reports/metrics.json
├── reports/
│   ├── report.md                 # problem framing, results, failure analysis, next steps
│   ├── decision_log.md           # 17 non-obvious decisions and why
│   └── figures/                  # confusion matrices etc., generated by eval_harness
└── tests/
    └── test_pipeline.py          # unit tests for cleaning, escalation rules, baselines
```

## Taxonomy discovery

The 8 intents in `src/taxonomy.py` were not invented top-down. They came from:
1. Embedding a sample of customer messages and clustering them (`src/cluster_explore.py`, TF-IDF+KMeans by default, sentence-transformers+HDBSCAN optional).
2. Reading 15-20 examples per cluster and writing a one-line description (open coding).
3. Freezing the result, with real example tweets per label as few-shot exemplars.

Re-run the discovery step yourself:
```bash
python3 -m src.cluster_explore --pairs data/interim/pairs.parquet --k 8
```
With only 13 real rows in the bundled sample, this mostly demonstrates the mechanism rather than producing a trustworthy taxonomy — re-run against the full dataset (see below) to actually validate or revise `src/taxonomy.py`.

## The golden evaluation set

`data/golden/golden_set.csv` has 49 rows: 13 drawn from real tweets in the bundled sample (`source=real_sample`), and 36 written in the same register to give per-intent and edge-case coverage at a scale the tiny bundled sample can't provide on its own (`source=synthetic_demo`) — **clearly marked as such in the `source` column.** This stands in for the brief's real 150-250 hand-labelled example requirement; see `reports/report.md` for the full disclosure.

Columns: `tweet_id, cust_text, true_intent, should_escalate, escalate_reason, ideal_reply_notes, labeler_notes, source`. Deliberately does **not** include a scripted "gold reply" string — grading a model against exact phrasing would defeat the point of the LLM-judge rubric. Instead, `ideal_reply_notes` states what a good reply must/must not contain.

Sampling method, self-consistency check, and how to scale this to a real 150-250-example set sampled from the full dataset: see `reports/report.md` → "Problem framing" and the build guide.

## Judge-human agreement (mandatory evidence for reply quality)

1. Run `make eval` with `ANTHROPIC_API_KEY` set — this produces `reports/judge_scores.csv`.
2. Copy `data/golden/human_judge_ratings_template.csv` to `data/golden/human_judge_ratings.csv`.
3. Score each drafted reply yourself, **blind to the judge's scores**, using the identical rubric in `src/judge.py` (groundedness, correctness, brand_voice, actionability, safety — 1 to 5).
4. Re-run `make eval`. `reports/metrics.json` → `judge_human_agreement` will now contain Spearman correlation per axis.

## Scaling to the full dataset

1. Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`) to `data/raw/twcs.csv`.
2. `make data RAW_CSV=data/raw/twcs.csv LIMIT=20000` (or higher).
3. Re-run `python3 -m src.cluster_explore --pairs data/interim/pairs.parquet --backend sbert` to validate/revise the taxonomy against a real-sized corpus.
4. Hand-label a real 150-250-row golden set sampled from that pull, following the method in `reports/report.md`, and replace `data/golden/golden_set.csv`.
5. `make eval` again — every number in `reports/metrics.json` will now reflect a real run.

## Tests

```bash
make test
```
Covers text cleaning, every branch of the escalation policy, and the baseline classifiers — the parts of the pipeline that are cheap to test without an LLM API key.


## Streamlit + Groq UI

This version adds `app.py` and replaces the Anthropic calls with Groq. Groq exposes an OpenAI-compatible API and the project defaults to `openai/gpt-oss-20b`; check Groq's current model/rate-limit page before changing the model.

### Run locally

```bash
pip install -r requirements.txt
# Windows
copy .env.example .env
# put your GROQ_API_KEY in .env
streamlit run app.py
```

For Streamlit Community Cloud, put `GROQ_API_KEY` in App Settings -> Secrets.

### Evaluation leakage fix

`data/golden/golden_set.csv` is held out for final evaluation. Threshold-tuning examples are in `data/tuning/threshold_tuning.csv` and are not copied from the golden set. Keep these datasets separate when reporting final results.
"# customer_support_agent" 
