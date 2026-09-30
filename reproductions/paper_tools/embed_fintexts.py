"""Reconstructed SBERT preprocessing, NOT recovered author embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prepare import LOCK, sha256, verify

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GROUPS = {
    "macro": [f"macro_category{i}" for i in range(1, 6)],
    "sector": [f"sector_category{i}" for i in range(1, 6)],
    "targetCompany": [f"targetCompany_category{i}" for i in range(1, 4)],
    "relatedCompany": [f"relatedCompany_category{i}" for i in range(1, 4)],
    "filing": ["filing_financialStatement", "filing_governanceRisks",
               "filing_overviewProduct", "filing_recentEventCatalyst", "filing_strategyMarketOps"],
}


def clean_text(value):
    return value.strip() if isinstance(value, str) else ""


def pool_categories(frame, columns, encoder, batch_size=32):
    """Encode each nonempty category; mean only present categories; missing level=NaN."""
    texts = [[clean_text(v) for v in row] for row in frame[columns].itertuples(index=False, name=None)]
    unique = sorted({v for row in texts for v in row if v})
    vectors = encoder.encode(unique, batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=False, show_progress_bar=False) if unique else np.empty((0, 384))
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(unique), 384) or not np.isfinite(vectors).all():
        raise ValueError("Encoder must return finite 384-dimensional vectors")
    cache = dict(zip(unique, vectors))
    result = np.full((len(frame), 384), np.nan, dtype=np.float32)
    for i, row in enumerate(texts):
        present = [cache[v] for v in row if v]
        if present:
            result[i] = np.mean(present, axis=0)
    return result


def prepare_tickers(published, output, tickers, encoder, groups, batch_size=32):
    spec = json.loads(LOCK.read_text())["fintexts"]
    paths = [Path(published) / item["path"] for item in spec["files"]]
    for path, item in zip(paths, spec["files"]):
        verify(path, item)
    selected = set(tickers)
    if not selected or any(not t or not all(c.isalnum() or c in '.-_' for c in t) for t in selected):
        raise ValueError("Supply valid explicit ticker names")
    if Path(output).exists():
        raise ValueError("Output already exists; select a new directory")
    collected = {t: [] for t in selected}
    cols = ["date", "ticker", "open", "high", "low", "close"] + [c for g in groups for c in GROUPS[g]]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=cols):
            frame = batch.to_pandas()
            frame = frame[frame.ticker.isin(selected)]
            for ticker, subset in frame.groupby("ticker"):
                collected[ticker].append(subset)
    if any(not frames for frames in collected.values()):
        raise ValueError("Requested ticker not found")
    Path(output).mkdir(parents=True)
    outputs = []
    for ticker in sorted(selected):
        frame = pd.concat(collected[ticker], ignore_index=True).sort_values("date")
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        if frame.date.duplicated().any():
            raise ValueError(f"Duplicate dates: {ticker}")
        prepared = frame[["date", "ticker", "open", "high", "low", "close"]].copy()
        for group in groups:
            vectors = pool_categories(frame, GROUPS[group], encoder, batch_size)
            prepared = pd.concat([prepared.reset_index(drop=True), pd.DataFrame(
                vectors, columns=[f"{group}_emb{i}" for i in range(384)])], axis=1)
        path = Path(output) / f"{ticker}.parquet"
        prepared.to_parquet(path, index=False)
        outputs.append({"ticker": ticker, "rows": len(prepared), "file": path.name, "sha256": sha256(path)})
        print("PREPARED =", ticker, len(prepared), flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL, revision=REVISION, device=args.device,
                                  local_files_only=args.offline, trust_remote_code=False)
    outputs = prepare_tickers(args.published, args.output, args.tickers, encoder,
                              args.groups, args.batch_size)
    manifest = {"protocol": "reconstructed-encoder", "model": MODEL, "revision": REVISION,
                "dimension": 384, "normalize_embeddings": False,
                "pooling": "mean nonempty category vectors per level; author averages levels",
                "max_seq_length": encoder.max_seq_length, "source_revision":
                json.loads(LOCK.read_text())["fintexts"]["revision"], "outputs": outputs,
                "limitations": ["Exact paper SBERT identity unavailable", "Published text may contain future information",
                                "Truncation follows encoder max_seq_length; no new LLM pairing performed"]}
    (args.output / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
