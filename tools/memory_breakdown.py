import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COMPONENTS = [
    "target_mb",
    "draft_mb",
    "kv_mb",
    "temp_buffers_mb",
    "spec_metadata_mb",
]


def _read_snapshots(path: Path) -> pd.DataFrame:
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
        if isinstance(payload, dict) and "snapshots" in payload:
            return pd.DataFrame(payload["snapshots"])
    raise ValueError(f"Unsupported snapshot format: {path}")


def _plot_stacked(avg_df: pd.DataFrame, out_path: Path):
    if avg_df.empty:
        return
    idx = avg_df["method"] + "|K=" + avg_df["k"].astype(str)
    base = pd.DataFrame(index=idx)
    for c in COMPONENTS:
        base[c] = avg_df.get(c, 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = pd.Series([0] * len(base), index=base.index)
    for c in COMPONENTS:
        ax.bar(base.index, base[c], bottom=bottom, label=c)
        bottom = bottom + base[c]
    ax.set_ylabel("MB")
    ax.set_title("Average memory breakdown by method and K")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True, help="memory snapshots json/jsonl")
    parser.add_argument("--output-dir", default="results/memory_breakdown")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_snapshots(Path(args.snapshots))
    if df.empty:
        raise SystemExit("empty memory snapshots")

    for c in ["method", "k", "workload_profile"] + COMPONENTS:
        if c not in df.columns:
            if c in COMPONENTS:
                df[c] = 0
            else:
                raise SystemExit(f"missing required column: {c}")

    if "total_mb" not in df.columns:
        df["total_mb"] = df[COMPONENTS].sum(axis=1)

    avg = (
        df.groupby(["method", "k", "workload_profile"], as_index=False)[COMPONENTS + ["total_mb"]]
        .mean(numeric_only=True)
        .sort_values(["method", "k", "workload_profile"])
    )

    compare = (
        avg.groupby(["method", "k"], as_index=False)[COMPONENTS + ["total_mb"]]
        .mean(numeric_only=True)
        .sort_values(["method", "k"])
    )

    no_sd = compare[compare["method"] == "no_sd"][["k", "total_mb"]].rename(columns={"total_mb": "total_mb_no_sd"})
    eagle3 = compare[compare["method"] == "eagle3"][["k", "total_mb"]].rename(columns={"total_mb": "total_mb_eagle3"})
    delta = no_sd.merge(eagle3, on="k", how="outer")
    if not delta.empty:
        delta["delta_mb_eagle3_minus_no_sd"] = delta["total_mb_eagle3"] - delta["total_mb_no_sd"]

    avg.to_csv(out_dir / "memory_breakdown_by_workload.csv", index=False)
    compare.to_csv(out_dir / "memory_breakdown_by_method.csv", index=False)
    delta.to_csv(out_dir / "memory_delta_no_sd_vs_eagle3.csv", index=False)

    _plot_stacked(compare, out_dir / "memory_breakdown_stacked.png")

    print(f"wrote {out_dir / 'memory_breakdown_by_workload.csv'}")
    print(f"wrote {out_dir / 'memory_breakdown_by_method.csv'}")
    print(f"wrote {out_dir / 'memory_delta_no_sd_vs_eagle3.csv'}")


if __name__ == "__main__":
    main()
