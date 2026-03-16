import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_events(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "events" in payload:
            return pd.DataFrame(payload["events"])
    raise ValueError(f"Unsupported input format: {path}")


def _bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def prompt_bucket(x):
        if pd.isna(x):
            return "unknown"
        x = int(x)
        if x < 256:
            return "short"
        if x < 1024:
            return "medium"
        return "long"

    def output_bucket(x):
        if pd.isna(x):
            return "unknown"
        x = int(x)
        if x < 128:
            return "short"
        if x < 512:
            return "medium"
        return "long"

    def qps_bucket(x):
        if pd.isna(x):
            return "unknown"
        x = float(x)
        if x < 1.0:
            return "low"
        if x < 4.0:
            return "medium"
        return "high"

    out["prompt_bucket"] = out.get("prompt_len", pd.Series(dtype=float)).map(prompt_bucket)
    out["output_bucket"] = out.get("output_len", pd.Series(dtype=float)).map(output_bucket)
    out["qps_bucket"] = out.get("request_rate", pd.Series(dtype=float)).map(qps_bucket)
    out["temp_bucket"] = out.get("temperature", pd.Series(dtype=float)).fillna(0.0).map(
        lambda v: "deterministic" if float(v) < 0.01 else "sampled"
    )
    return out


def _safe_ratio(numer, denom):
    denom = denom.replace({0: pd.NA})
    return numer / denom


def _plot_bucket(df: pd.DataFrame, bucket_col: str, out_path: Path):
    agg = df.groupby(bucket_col, dropna=False, as_index=False).agg(
        proposed_tokens=("proposed_tokens", "sum"),
        accepted_tokens=("accepted_tokens", "sum"),
        rejected_tokens=("rejected_tokens", "sum"),
    )
    if agg.empty:
        return
    agg["acceptance_rate"] = _safe_ratio(agg["accepted_tokens"], agg["proposed_tokens"]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(agg[bucket_col].astype(str), agg["acceptance_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("acceptance_rate")
    ax.set_xlabel(bucket_col)
    ax.set_title(f"Acceptance by {bucket_col}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="acceptance trace json/jsonl")
    parser.add_argument("--bench-summary", default="", help="optional parsed bench summary csv")
    parser.add_argument("--output-dir", default="results/acceptance")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_events(Path(args.trace))
    if df.empty:
        raise SystemExit("empty acceptance trace")

    required = ["request_id", "step_id", "proposed_tokens", "accepted_tokens"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"missing required column: {c}")

    if "rejected_tokens" not in df.columns:
        df["rejected_tokens"] = (df["proposed_tokens"] - df["accepted_tokens"]).clip(lower=0)

    df = _bucketize(df)

    step_df = df.copy()
    step_df["acceptance_rate"] = _safe_ratio(step_df["accepted_tokens"], step_df["proposed_tokens"]).fillna(0.0)

    req_df = (
        df.groupby("request_id", as_index=False)
        .agg(
            prompt_len=("prompt_len", "max"),
            output_len=("output_len", "max"),
            request_rate=("request_rate", "max"),
            temperature=("temperature", "max"),
            proposed_tokens=("proposed_tokens", "sum"),
            accepted_tokens=("accepted_tokens", "sum"),
            rejected_tokens=("rejected_tokens", "sum"),
        )
        .fillna({"temperature": 0.0})
    )
    req_df["acceptance_rate"] = _safe_ratio(req_df["accepted_tokens"], req_df["proposed_tokens"]).fillna(0.0)
    req_df = _bucketize(req_df)

    bucket_df = (
        req_df.groupby(["prompt_bucket", "output_bucket", "qps_bucket", "temp_bucket"], as_index=False)
        .agg(
            requests=("request_id", "nunique"),
            proposed_tokens=("proposed_tokens", "sum"),
            accepted_tokens=("accepted_tokens", "sum"),
            rejected_tokens=("rejected_tokens", "sum"),
            acceptance_rate=("acceptance_rate", "mean"),
        )
        .sort_values("requests", ascending=False)
    )

    step_out = output_dir / "step_acceptance.parquet"
    req_out = output_dir / "request_acceptance.parquet"
    bucket_out = output_dir / "bucket_acceptance.parquet"
    step_df.to_parquet(step_out, index=False)
    req_df.to_parquet(req_out, index=False)
    bucket_df.to_parquet(bucket_out, index=False)

    if args.bench_summary:
        bench_df = pd.read_csv(args.bench_summary)
        if "request_id" in bench_df.columns:
            aligned = req_df.merge(bench_df, on="request_id", how="left")
            aligned.to_parquet(output_dir / "request_bench_aligned.parquet", index=False)

    _plot_bucket(req_df, "prompt_bucket", output_dir / "acceptance_by_prompt_bucket.png")
    _plot_bucket(req_df, "output_bucket", output_dir / "acceptance_by_output_bucket.png")
    _plot_bucket(req_df, "qps_bucket", output_dir / "acceptance_by_qps_bucket.png")

    print(f"wrote {step_out}")
    print(f"wrote {req_out}")
    print(f"wrote {bucket_out}")


if __name__ == "__main__":
    main()
