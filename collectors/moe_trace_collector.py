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


def _as_expert_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        parts = [x.strip() for x in v.split(",") if x.strip()]
        return [int(x) for x in parts]
    return []


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _compute_overlap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (req, layer), group in df.groupby(["request_id", "layer_id"]):
        g = group.sort_values("token_idx")
        prev = None
        for _, row in g.iterrows():
            cur = row["experts"]
            if prev is not None:
                rows.append(
                    {
                        "request_id": req,
                        "layer_id": layer,
                        "token_idx": int(row["token_idx"]),
                        "overlap_jaccard": _jaccard(prev, cur),
                        "phase": row.get("phase", "unknown"),
                    }
                )
            prev = cur
    return pd.DataFrame(rows)


def _compute_reuse_distance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (req, layer), group in df.groupby(["request_id", "layer_id"]):
        g = group.sort_values("token_idx")
        last_seen = {}
        for _, row in g.iterrows():
            token_idx = int(row["token_idx"])
            for expert in row["experts"]:
                if expert in last_seen:
                    rows.append(
                        {
                            "request_id": req,
                            "layer_id": layer,
                            "expert_id": int(expert),
                            "token_idx": token_idx,
                            "reuse_distance": token_idx - last_seen[expert],
                            "phase": row.get("phase", "unknown"),
                        }
                    )
                last_seen[expert] = token_idx
    return pd.DataFrame(rows)


def _plot_heat(heat_df: pd.DataFrame, out_path: Path):
    pivot = heat_df.pivot_table(index="layer_id", columns="expert_id", values="count", fill_value=0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xlabel("expert_id")
    ax.set_ylabel("layer_id")
    ax.set_title("Expert Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_hist(df: pd.DataFrame, col: str, out_path: Path, title: str):
    if df.empty or col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(title)
    ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="moe trace json/jsonl")
    parser.add_argument("--output-dir", default="results/moe_trace")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_events(Path(args.trace))
    if df.empty:
        raise SystemExit("empty moe trace")

    required = ["request_id", "token_idx", "layer_id", "experts"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"missing required column: {c}")

    if "phase" not in df.columns:
        df["phase"] = "unknown"
    df["experts"] = df["experts"].map(_as_expert_list)

    exploded = df.explode("experts").dropna(subset=["experts"])
    exploded["expert_id"] = exploded["experts"].astype(int)

    heat = (
        exploded.groupby(["layer_id", "expert_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    overlap = _compute_overlap(df)
    reuse = _compute_reuse_distance(df)

    heat.to_parquet(out_dir / "expert_heat.parquet", index=False)
    overlap.to_parquet(out_dir / "overlap.parquet", index=False)
    reuse.to_parquet(out_dir / "reuse_distance.parquet", index=False)

    _plot_heat(heat, out_dir / "expert_heatmap.png")
    _plot_hist(overlap, "overlap_jaccard", out_dir / "overlap_distribution.png", "Token-to-token overlap")
    _plot_hist(reuse, "reuse_distance", out_dir / "reuse_distance_distribution.png", "Expert reuse distance")

    print(f"wrote {out_dir / 'expert_heat.parquet'}")
    print(f"wrote {out_dir / 'overlap.parquet'}")
    print(f"wrote {out_dir / 'reuse_distance.parquet'}")


if __name__ == "__main__":
    main()
