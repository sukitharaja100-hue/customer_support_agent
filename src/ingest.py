"""
Ingest the CS-Twitter CSV and reconstruct (customer message -> brand's
historical reply) pairs for one brand, using DuckDB so the full ~2.8M-row
Kaggle CSV never has to be loaded into pandas at once.

Usage:
    python -m src.ingest --brand AppleSupport --raw data/raw/twcs.csv \
        --out data/interim/pairs.parquet --limit 20000

For this repo's committed demo run, --raw defaults to the small bundled
sample (data/raw/twcs_sample.csv) so `make data` works with zero setup.
Point --raw at the full Kaggle download (thoughtvector/customer-support-on-twitter)
for real headline numbers -- see README "Scaling to the full dataset".
"""

import argparse
import sys

import duckdb
import pandas as pd

from src.clean import clean_tweet, has_negative_emoji, has_positive_emoji, mentions_dm_handoff


def build_pairs(raw_csv: str, brand: str, limit: int | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW twcs AS
        SELECT * FROM read_csv_auto('{raw_csv}', SAMPLE_SIZE=-1, ALL_VARCHAR=TRUE)
        """
    )

    limit_clause = f"ORDER BY random() LIMIT {limit}" if limit else ""
    query = f"""
        SELECT
            c.tweet_id      AS cust_tweet_id,
            c.text          AS cust_text,
            c.created_at    AS cust_created_at,
            b.tweet_id      AS brand_tweet_id,
            b.text          AS brand_text,
            b.created_at    AS brand_created_at
        FROM twcs b
        JOIN twcs c ON b.in_response_to_tweet_id = c.tweet_id
        WHERE b.inbound = 'False' AND b.author_id = '{brand}'
          AND c.inbound = 'True'
        {limit_clause}
    """
    df = con.execute(query).df()
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cust_text_clean"] = df["cust_text"].apply(clean_tweet)
    df["brand_text_clean"] = df["brand_text"].apply(clean_tweet)
    df["has_negative_emoji"] = df["cust_text"].apply(has_negative_emoji)
    df["has_positive_emoji"] = df["cust_text"].apply(has_positive_emoji)
    df["has_dm_handoff"] = df["brand_text"].apply(mentions_dm_handoff)

    before = len(df)
    df = df.drop_duplicates(subset=["cust_text_clean", "brand_text_clean"])
    df = df[df["brand_text_clean"].str.replace("<URL>", "").str.strip().str.len() > 0]
    dropped = before - len(df)
    if dropped:
        print(f"[ingest] dropped {dropped} duplicate/boilerplate-only rows", file=sys.stderr)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--raw", default="data/raw/twcs_sample.csv")
    ap.add_argument("--out", default="data/interim/pairs.parquet")
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()

    df = build_pairs(args.raw, args.brand, args.limit)
    print(f"[ingest] {len(df)} raw pairs for brand={args.brand}", file=sys.stderr)
    df = enrich(df)
    print(f"[ingest] {len(df)} pairs after cleaning/dedup", file=sys.stderr)

    if len(df) == 0:
        print(
            f"[ingest] WARNING: 0 pairs found for brand={args.brand} in {args.raw}. "
            "The bundled sample CSV is tiny (~100 rows) and only intended to prove "
            "the pipeline runs end to end -- point --raw at the full Kaggle CSV "
            "for a real run.",
            file=sys.stderr,
        )

    df.to_parquet(args.out, index=False)
    print(f"[ingest] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
