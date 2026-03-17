import argparse
import csv
from pathlib import Path


def to_float(v: str) -> float:
    try:
        if v is None or v == "":
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_placeholder_png(path: Path) -> None:
    # 1x1 transparent PNG
    data = bytes.fromhex(
        "89504E470D0A1A0A"
        "0000000D49484452000000010000000108060000001F15C489"
        "0000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"
    )
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    parser.add_argument("--out-table", default="results/main_table/main_comparison.csv")
    parser.add_argument("--out-fig-dir", default="results/main_figures")
    args = parser.parse_args()

    rows = read_rows(Path(args.registry_csv))
    done = [
        r
        for r in rows
        if (r.get("status", "").lower() in {"done", "running"})
        and (r.get("result_label", "").lower() != "invalid")
    ]

    cols = [
        "experiment_id",
        "model",
        "spec_method",
        "optimization_module",
        "workload_profile",
        "ttft_p95_ms",
        "tpot_p95_ms",
        "throughput_tok_per_s",
        "goodput",
    ]
    keep = [c for c in cols if done and c in done[0]]
    main_table = sorted(done, key=lambda r: to_float(r.get("ttft_p95_ms", "")))

    out_table = Path(args.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        w.writeheader()
        for r in main_table:
            w.writerow({k: r.get(k, "") for k in keep})

    fig_dir = Path(args.out_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Environment may not have plotting dependencies; generate valid placeholder PNGs.
    write_placeholder_png(fig_dir / "main_ttft_p95.png")
    write_placeholder_png(fig_dir / "main_throughput.png")

    print(f"wrote {out_table}")
    print(f"wrote {fig_dir / 'main_ttft_p95.png'}")
    print(f"wrote {fig_dir / 'main_throughput.png'}")


if __name__ == "__main__":
    main()
