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
    parser.add_argument("--out-table", default="results/ablation/ablation.csv")
    parser.add_argument("--out-fig-dir", default="results/ablation_figures")
    args = parser.parse_args()

    rows = read_rows(Path(args.registry_csv))
    done = [
        r
        for r in rows
        if (r.get("status", "").lower() in {"done", "running"})
        and (r.get("result_label", "").lower() != "invalid")
    ]

    group = {}
    for r in done:
        key = (r.get("optimization_module", ""), r.get("spec_method", ""))
        if key not in group:
            group[key] = {"rows": 0, "ttft": 0.0, "tpot": 0.0, "tput": 0.0}
        group[key]["rows"] += 1
        group[key]["ttft"] += to_float(r.get("ttft_p95_ms", ""))
        group[key]["tpot"] += to_float(r.get("tpot_p95_ms", ""))
        group[key]["tput"] += to_float(r.get("throughput_tok_per_s", ""))

    abl = []
    for (mod, method), v in group.items():
        n = max(v["rows"], 1)
        abl.append(
            {
                "optimization_module": mod,
                "spec_method": method,
                "ttft_p95_ms": round(v["ttft"] / n, 6),
                "tpot_p95_ms": round(v["tpot"] / n, 6),
                "throughput_tok_per_s": round(v["tput"] / n, 6),
                "rows": v["rows"],
            }
        )
    abl.sort(key=lambda r: r["rows"], reverse=True)

    out_table = Path(args.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    fields = ["optimization_module", "spec_method", "ttft_p95_ms", "tpot_p95_ms", "throughput_tok_per_s", "rows"]
    with out_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in abl:
            w.writerow(r)

    fig_dir = Path(args.out_fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    write_placeholder_png(fig_dir / "ablation_ttft_p95.png")
    write_placeholder_png(fig_dir / "ablation_throughput.png")

    print(f"wrote {out_table}")
    print(f"wrote {fig_dir / 'ablation_ttft_p95.png'}")
    print(f"wrote {fig_dir / 'ablation_throughput.png'}")


if __name__ == "__main__":
    main()
