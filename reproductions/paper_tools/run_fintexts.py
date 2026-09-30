"""Validate prepared embeddings, then plan or run the unchanged FinTexTS entrypoint."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare import LOCK, sha256, verify_source

PREFIXES = ("macro", "sector", "targetCompany", "relatedCompany", "filing")
# Only choices whose constructors are actually imported by the pinned runner.
MODELS = ("DLinear", "PatchTST", "Informer", "Autoformer", "iTransformer",
          "Reformer", "Crossformer", "Transformer", "FiLM",
          "Nonstationary_Transformer", "TSMixer", "TiDE")


def validate_rows(rows, columns, prefixes, seq_len=64, pred_len=3, batch_size=32):
    required = ["date", "ticker", "open", "high", "low", "close"]
    embeddings = [f"{p}_emb{i}" for p in prefixes for i in range(384)]
    missing = sorted(set(required + embeddings) - set(columns))
    if missing:
        raise ValueError(f"Missing {len(missing)} columns; first: {missing[:8]}. "
                         "Published raw text is not a forecasting embedding parquet.")
    counts = dict(train=0, val=0, test=0)
    previous = None
    ticker = None
    missing_embeddings = 0
    for row in rows:
        date = row["date"]
        if not isinstance(date, str):
            raise ValueError("Author-compatible date must be an ISO YYYY-MM-DD string")
        if dt.date.fromisoformat(date).isoformat() != date:
            raise ValueError("Noncanonical date")
        if previous is not None and date <= previous:
            raise ValueError("Dates must be unique and strictly increasing; no automatic sorting")
        previous = date
        value = row["ticker"]
        if not isinstance(value, str) or not value:
            raise ValueError("Missing ticker")
        ticker = value if ticker is None else ticker
        if value != ticker:
            raise ValueError("Expected exactly one ticker per parquet")
        if not "2019-01-01" <= date < "2023-12-19":
            continue
        for name in ("open", "high", "low", "close"):
            if row[name] is None or not math.isfinite(float(row[name])):
                raise ValueError(f"Invalid price: {date} {name}")
        for name in embeddings:
            value = row[name]
            if value is None or math.isnan(float(value)):
                missing_embeddings += 1  # Author loader fills missing pooled values.
            elif not math.isfinite(float(value)):
                raise ValueError(f"Infinite embedding: {date} {name}")
        split = "train" if date < "2022-01-01" else "val" if date < "2023-01-01" else "test"
        counts[split] += 1
    if counts["val"] < seq_len:
        raise ValueError("Validation history too short for the author's test overlap")
    windows = {"train": counts["train"] - seq_len - pred_len + 1,
               "val": counts["val"] - pred_len + 1,
               "test": counts["test"] - pred_len + 1}
    if windows["train"] < batch_size or min(windows.values()) < 1:
        raise ValueError(f"Empty/incomplete author batches: {windows}")
    return {"ticker": ticker, "rows": counts, "windows": windows,
            "missing_embedding_cells": missing_embeddings}


def inspect_parquet(path, prefixes, seq_len, pred_len, batch_size):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet preflight requires pyarrow; install in the reproduction environment") from exc
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    selected = ["date", "ticker", "open", "high", "low", "close"]
    selected += [f"{p}_emb{i}" for p in prefixes for i in range(384)]
    missing = set(selected) - set(columns)
    if missing:
        validate_rows([], columns, prefixes, seq_len, pred_len, batch_size)
    rows = (row for batch in parquet.iter_batches(batch_size=128, columns=selected)
            for row in batch.to_pylist())
    return validate_rows(rows, columns, prefixes, seq_len, pred_len, batch_size)


def launch_command(args, author, output):
    data = args.data.resolve()
    return [sys.executable, "-u", "-m", "forecasting_task.run",
            "--root_path", str(data.parent), "--data_path", data.name,
            "--logdir", str(output), "--model_type", args.model,
            "--used_col_prefixes", ",".join(args.prefixes), "--seed", str(args.seed),
            "--num_epoch", str(args.epochs), "--seq_len", str(args.seq_len),
            "--label_len", str(args.label_len), "--pred_len", str(args.pred_len),
            "--batch_size", str(args.batch_size), "--text_weight", str(args.text_weight)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", choices=MODELS, default="PatchTST")
    p.add_argument("--prefixes", nargs="+", choices=PREFIXES, default=list(PREFIXES))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--label-len", type=int, default=16)
    p.add_argument("--pred-len", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--text-weight", type=float, default=0.1)
    p.add_argument("--execute", action="store_true", help="Default is validation and command preview only")
    args = p.parse_args()
    if min(args.epochs, args.seq_len, args.label_len, args.pred_len, args.batch_size) < 1:
        p.error("Lengths, epochs and batch size must be positive")
    if args.label_len > args.seq_len or not 0 <= args.text_weight <= 1:
        p.error("Invalid label length or text weight")
    if len(set(args.prefixes)) != len(args.prefixes):
        p.error("Duplicate prefixes change the author's pooling weights")
    spec = json.loads(LOCK.read_text())["fintexts"]
    author = verify_source(args.author, spec)
    output = args.output.resolve()
    if output == author or author in output.parents:
        p.error("Output must be outside the pristine author checkout")
    if output.exists():
        p.error("Use a new output directory; existing results are never overwritten")
    report = inspect_parquet(args.data, args.prefixes, args.seq_len, args.pred_len, args.batch_size)
    command = launch_command(args, author, output)
    metadata = {"protocol": "author-code; not validated paper-number replication",
                "author_commit": spec["author_commit"], "input_sha256": sha256(args.data),
                "data": report, "command": command,
                "known_limitations": ["Scaler fits all filtered train/val/test rows",
                                      "Exact published forecasting encoder is not recovered",
                                      "Runner saves predictions, not trained model weights"],
                "status": "PLANNED", "training_started": False}
    if not args.execute:
        print(json.dumps(metadata, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "runner_manifest.json"
    metadata.update(status="STARTING", training_started=False)
    manifest.write_text(json.dumps(metadata, indent=2))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(author), str(author / "forecasting_task")])
    try:
        with (output / "console.log").open("wb") as log:
            proc = subprocess.Popen(command, cwd=author, env=env, stdout=log, stderr=subprocess.STDOUT)
            metadata.update(status="RUNNING", training_started="UNKNOWN_UNTIL_LOG_VERIFIED")
            manifest.write_text(json.dumps(metadata, indent=2))
            rc = proc.wait()
        metadata.update(returncode=rc, status="ENTRYPOINT_EXITED_OK" if rc == 0 else "FAILED")
        if rc:
            raise RuntimeError(f"Author entrypoint failed ({rc}); see {output / 'console.log'}")
        if not all((output / x).is_file() for x in ("result.txt", "test_io.pt")):
            metadata["status"] = "MISSING_OUTPUTS"
            raise RuntimeError("Author entrypoint did not produce nonempty-run outputs")
        metadata.update(status="OUTPUTS_PRESENT_NOT_METRIC_ACCEPTANCE", training_started=True)
    finally:
        manifest.write_text(json.dumps(metadata, indent=2))
    print("AUTHOR_OUTPUTS =", output)
    print("PAPER_RESULTS_REPRODUCED = NOT_VERIFIED")


if __name__ == "__main__":
    main()
