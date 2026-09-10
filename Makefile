.PHONY: setup data explore golden baseline eval test report clean

RAW_CSV ?= data/raw/twcs_sample.csv
BRAND ?= AppleSupport
LIMIT ?= 20000

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Ingest + clean + reconstruct (customer -> brand reply) pairs for one brand.
# Defaults to the tiny bundled sample so this always runs with zero setup;
# override RAW_CSV to point at the full Kaggle download for a real run:
#   make data RAW_CSV=data/raw/twcs.csv LIMIT=20000
data:
	python3 -m src.ingest --brand $(BRAND) --raw $(RAW_CSV) --out data/interim/pairs.parquet --limit $(LIMIT)

# Optional: re-derive/validate the intent taxonomy from the data (Section 5
# of the build guide). Not required for `make eval` -- the taxonomy is
# already frozen in src/taxonomy.py.
explore:
	python3 -m src.cluster_explore --pairs data/interim/pairs.parquet --k 8

# Golden set is committed at data/golden/golden_set.csv -- this target just
# confirms it's present and prints a summary.
golden:
	python3 -c "import pandas as pd; df = pd.read_csv('data/golden/golden_set.csv'); \
	print(df.shape); print(df['true_intent'].value_counts()); print(df['source'].value_counts())"

baseline:
	python3 -m src.classify_baseline --pairs data/interim/pairs.parquet --golden data/golden/golden_set.csv

eval:
	python3 -m src.eval_harness

test:
	python3 -m pytest tests/ -v

report:
	@echo "See reports/report.md and reports/decision_log.md"
	@echo "Headline metrics:"
	@python3 -c "import json; m = json.load(open('reports/metrics.json')); print(json.dumps(m['intent_classification'], indent=2)); print('escalation_f1:', m['escalation']['f1'])"

clean:
	rm -rf data/interim/*.parquet data/interim/*.joblib reports/metrics.json reports/baseline_metrics.json reports/figures/*.png reports/drafted_replies.csv reports/judge_scores.csv
