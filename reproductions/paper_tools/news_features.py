"""HS300 news count features, optionally joined with independently scored sentiment."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from prepare import LOCK, sha256, verify


def article_id(ticker, timestamp, summary):
    payload = json.dumps([ticker, timestamp, summary], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_daily(rows, ticker, scores=None):
    records = []
    for row in rows:
        text = row.get("summary")
        if not isinstance(text, str) or not text.strip():
            continue
        timestamp = row["datetime"]
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("Asia/Shanghai")
        else:
            stamp = stamp.tz_convert("Asia/Shanghai")
        # Collect news through local midnight. First usable forecast is subsequent market close.
        available = stamp.normalize() + pd.Timedelta(days=1)
        records.append({"article_id": article_id(ticker, timestamp, text),
                        "available_at": available.tz_convert("UTC"), "summary": text})
    articles = pd.DataFrame(records, columns=["article_id", "available_at", "summary"]).drop_duplicates("article_id")
    if articles.empty:
        raise ValueError("No usable news")
    if scores is not None:
        if scores.article_id.duplicated().any() or not np.isfinite(scores.sentiment).all() or not scores.sentiment.between(-1, 1).all():
            raise ValueError("Require unique article IDs and finite sentiment in [-1, 1]")
        articles = articles.merge(scores[["article_id", "sentiment"]], on="article_id", how="left", validate="one_to_one")
        if articles.sentiment.isna().any():
            raise ValueError("Sentiment scores must cover every exported article")
    grouped = articles.groupby("available_at")
    daily = grouped.size().rename("news_count").to_frame()
    if scores is not None:
        daily["sentiment_mean"] = grouped.sentiment.mean()
    # Missing calendar days are zero-count; final zero day prevents indefinite stale carry.
    calendar = pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(days=1), freq="D")
    daily = daily.reindex(calendar).fillna(0.0).rename_axis("available_at")
    return articles, daily.reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--published", type=Path, required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scores", type=Path, help="CSV article_id,sentiment; use exported articles for scoring")
    p.add_argument("--score-provenance", type=Path, help="Required JSON with model and revision when scores are supplied")
    args = p.parse_args()
    if bool(args.scores) != bool(args.score_provenance):
        p.error("Scores require their model/revision provenance")
    spec = json.loads(LOCK.read_text())["finmultitime"]
    item = next(i for i in spec["files"] if i["path"] == "text/hs300news_summary.zip")
    archive = args.published / item["path"]
    verify(archive, item)
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if Path(n).name.startswith(args.ticker + '_') and n.endswith('.jsonl')]
        if len(matches) != 1:
            raise ValueError(f"Need one news member, found {len(matches)}")
        with z.open(matches[0]) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    scores = pd.read_csv(args.scores) if args.scores else None
    provenance = json.loads(args.score_provenance.read_text()) if args.score_provenance else None
    if provenance is not None and not all(provenance.get(k) for k in ("model", "revision")):
        raise ValueError("Missing scorer identity")
    articles, daily = build_daily(rows, args.ticker, scores)
    args.output.mkdir(parents=True, exist_ok=False)
    articles.to_csv(args.output / "articles.csv", index=False)
    daily.to_csv(args.output / "features.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({"protocol": "reconstructed-news-features", "ticker": args.ticker,
        "archive_sha256": sha256(archive), "scorer": provenance, "scores_sha256": sha256(args.scores) if args.scores else None,
        "availability": "Next local midnight after news timestamp; join uses HS300 15:00 close",
        "limitations": ["Counts alone do not reproduce paper sentiment", "Sentiment scorer identity in paper unavailable",
                        "Availability of summaries versus original news timestamps is not proven"]}, indent=2))


if __name__ == "__main__":
    main()
