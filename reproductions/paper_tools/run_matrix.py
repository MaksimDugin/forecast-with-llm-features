"""Create/review and sequentially run explicit reproducible experiment matrices."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

from prepare import sha256
from run_fintexts import MODELS


def build_commands(config, output):
    paper = config["paper"]
    seeds = config["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Require explicit unique seeds (paper's three values are unpublished)")
    tickers = config["tickers"]
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Require explicit unique tickers")
    models = config["models"]
    if not models or len(models) != len(set(models)):
        raise ValueError("Require unique models")
    cells = []
    for ticker, model, seed in itertools.product(tickers, models, seeds):
        horizons = [3] if paper == "fintexts" else config["horizons"]
        for horizon in horizons:
            cell_id = f"{ticker}-{model}-{seed}-h{horizon}"
            if any(not all(c.isalnum() or c in '._-' for c in str(v)) for v in (ticker, model, seed, horizon)):
                raise ValueError("Unsafe cell identifier")
            cell = Path(output).resolve() / cell_id
            if paper == "fintexts":
                if model not in MODELS:
                    raise ValueError(model)
                cmd = [sys.executable, str(Path(__file__).with_name("train_fintexts.py")),
                       "--author", config["author"], "--data", str(Path(config["prepared"]) / f"{ticker}.parquet"),
                       "--prefixes", *config["prefixes"], "--scaler", config.get("scaler", "author-full"),
                       "--text-weight", str(config.get("text_weight", 0.1))]
            elif paper == "finmultitime":
                cmd = [sys.executable, str(Path(__file__).with_name("finmultitime.py")),
                       "--published", config["published"], "--ticker", ticker,
                       "--train-end", config["train_end"], "--val-end", config["val_end"],
                       "--horizon", str(horizon)]
                if config.get("modality", "price") != "price":
                    cmd += ["--modality", config["modality"], "--features", config["feature_files"][ticker],
                            "--feature-columns", *config["feature_columns"]]
            else:
                raise ValueError(paper)
            if paper == "fintexts" and config.get("portable_autoformer"):
                cmd += ["--portable-autoformer"]
            cmd += ["--model", model, "--seed", str(seed), "--epochs", str(config["epochs"]),
                    "--output", str(cell)]
            if config.get("smoke_batches"):
                cmd += ["--smoke-batches", str(config["smoke_batches"])]
            cells.append({"id": cell_id, "ticker": ticker, "model": model, "seed": seed,
                          "horizon": horizon, "output": str(cell), "command": cmd})
    if not cells or len({c['id'] for c in cells}) != len(cells):
        raise ValueError("Empty or duplicate matrix")
    return cells


def aggregate(cells):
    rows = []
    for cell in cells:
        success = Path(cell["output"]) / "SUCCESS.json"
        if success.is_file():
            rows.append({"id": cell["id"], **json.loads(success.read_text())["metrics"]})
    from statistics import fmean, pstdev
    grouped = {}
    for cell in cells:
        row = next((r for r in rows if r["id"] == cell["id"]), None)
        if row is not None:
            grouped.setdefault((cell["model"], cell["horizon"]), {}).setdefault(cell["ticker"], []).append(row)
    summaries = []
    for (model, horizon), tickers in sorted(grouped.items()):
        metric_names = ["scaled_mse", "scaled_mae", "persistence_scaled_mse"]
        values = {}
        for name in metric_names:
            per_ticker = [fmean(r[name] for r in runs) for runs in tickers.values()]
            values[name] = {"mean_across_tickers_after_seed_average": fmean(per_ticker),
                            "population_std_across_tickers": pstdev(per_ticker)}
        summaries.append({"model": model, "horizon": horizon, "tickers": len(tickers), "metrics": values})
    return {"requested": len(cells), "completed": len(rows), "complete": len(rows) == len(cells),
            "smoke_only": any(r.get("smoke_only", False) for r in rows),
            "averaging": "Equal seed average within ticker, then equal ticker average; incomplete matrices remain labeled incomplete",
            "summaries": summaries, "results": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    cells = build_commands(config, args.output)
    plan = {"config_sha256": sha256(args.config), "cells": cells, "full_training_requested": args.execute}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "matrix.json").write_text(json.dumps(plan, indent=2))
    for cell in cells:
        with (args.output / (cell["id"] + ".log")).open("w") as log:
            result = subprocess.run(cell["command"] + ["--execute"], stdout=log, stderr=subprocess.STDOUT)
        (args.output / "summary.json").write_text(json.dumps(aggregate(cells), indent=2))
        result.check_returncode()
        print("COMPLETED =", cell["id"], flush=True)


if __name__ == "__main__":
    main()
