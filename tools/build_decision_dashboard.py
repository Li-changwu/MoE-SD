import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def to_float(v: Any):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(registry_csv: Path) -> List[Dict[str, str]]:
    if not registry_csv.exists():
        return []
    with registry_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def valid_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        status = (r.get("status") or "").lower()
        label = (r.get("result_label") or "").lower()
        has_signal = any((to_float(r.get(k)) or 0) > 0 for k in ["ttft_p95_ms", "tpot_p95_ms", "throughput_tok_per_s", "goodput"])
        if status in {"done", "running"} and label != "invalid" and has_signal:
            out.append(r)
    return out


def best_row(rows: List[Dict[str, str]], metric: str, minimize: bool) -> Dict[str, str]:
    cands = [r for r in rows if to_float(r.get(metric)) is not None]
    if not cands:
        return {}
    key_fn = (lambda r: to_float(r.get(metric)))
    return min(cands, key=key_fn) if minimize else max(cands, key=key_fn)


def build_best_cards(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    defs = [
        ("Best TTFT", "ttft_p95_ms", True, "delta_ttft_p95_pct"),
        ("Best TPOT", "tpot_p95_ms", True, "delta_tpot_p95_pct"),
        ("Best Throughput", "throughput_tok_per_s", False, "delta_throughput_pct"),
        ("Best Goodput", "goodput", False, "delta_goodput_pct"),
    ]
    cards = []
    for title, metric, minimize, delta_metric in defs:
        row = best_row(rows, metric, minimize)
        if not row:
            cards.append({"title": title, "empty": True})
            continue
        cards.append(
            {
                "title": title,
                "method": row.get("spec_method", ""),
                "config": f"policy={row.get('policy_name', '')}; module={row.get('optimization_module', '')}",
                "value": row.get(metric, ""),
                "delta": row.get(delta_metric, ""),
                "updated": row.get("last_updated", row.get("date", "")),
                "workload": row.get("workload_profile", ""),
                "exp_id": row.get("experiment_id", ""),
            }
        )
    return cards


def recommend_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    cands = [r for r in rows if to_float(r.get("score_main")) is not None]
    if not cands:
        return {}
    return max(cands, key=lambda r: to_float(r.get("score_main")) or -1e9)


def top_issues(rows: List[Dict[str, str]]) -> List[str]:
    issues = []
    high_ttft = [r for r in rows if (to_float(r.get("ttft_p95_ms")) or 0) > 2000]
    if high_ttft:
        issues.append("Long prompt workload 下 TTFT p95 仍偏高")

    unstable_prefetch = [
        r
        for r in rows
        if (to_float(r.get("acceptance_rate_mean")) or 1) < 0.5 and (to_float(r.get("wasted_prefetched_bytes")) or 0) > 0
    ]
    if unstable_prefetch:
        issues.append("低 acceptance 时 prefetch 收益不稳定")

    high_qps_reg = [
        r
        for r in rows
        if "high_qps" in (r.get("workload_profile") or "").lower() and (r.get("result_label") or "") in {"regression", "partial_win"}
    ]
    if high_qps_reg:
        issues.append("高 QPS 场景收益稳定性不足")

    if not issues:
        issues = ["暂无显著阻塞项，请继续扩充 workload 覆盖"]
    return issues[:3]


def dump_regression_table(rows: List[Dict[str, str]], out_csv: Path) -> None:
    keys = rows[0].keys() if rows else []
    filtered = [
        r
        for r in rows
        if (r.get("result_label") in {"regression", "invalid"})
        or (to_float(r.get("fallback_count")) or 0) > 0
        or (to_float(r.get("oom_count")) or 0) > 0
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            writer.writerows(filtered)


def html_table(rows: List[Dict[str, str]], columns: List[str], limit: int = 12) -> str:
    rows = rows[:limit]
    th = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    tr = []
    for r in rows:
        td = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns)
        tr.append(f"<tr>{td}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(tr)}</tbody></table>"


def build_html(rows: List[Dict[str, str]]) -> str:
    valid = valid_rows(rows)
    cards = build_best_cards(valid)
    result_dist = Counter((r.get("result_label") or "unknown") for r in valid)
    module_dist = Counter((r.get("optimization_module") or "unknown") for r in valid)
    rec = recommend_row(valid)
    problems = top_issues(valid)

    latest = sorted(valid, key=lambda r: r.get("date", ""), reverse=True)

    trend_payload = [
        {
            "experiment_id": r.get("experiment_id", ""),
            "date": r.get("date", ""),
            "ttft_p50_ms": to_float(r.get("ttft_p50_ms")),
            "ttft_p95_ms": to_float(r.get("ttft_p95_ms")),
            "tpot_p50_ms": to_float(r.get("tpot_p50_ms")),
            "tpot_p95_ms": to_float(r.get("tpot_p95_ms")),
            "throughput_tok_per_s": to_float(r.get("throughput_tok_per_s")),
            "goodput": to_float(r.get("goodput")),
        }
        for r in sorted(valid, key=lambda x: (x.get("date", ""), x.get("experiment_id", "")))
    ]

    cards_html = []
    for c in cards:
        if c.get("empty"):
            cards_html.append(f"<div class='card'><h3>{html.escape(c['title'])}</h3><p>No data</p></div>")
            continue
        cards_html.append(
            "<div class='card'>"
            f"<h3>{html.escape(c['title'])}</h3>"
            f"<p><b>Method:</b> {html.escape(c['method'])}</p>"
            f"<p><b>Config:</b> {html.escape(c['config'])}</p>"
            f"<p><b>Value:</b> {html.escape(str(c['value']))}</p>"
            f"<p><b>Delta vs baseline:</b> {html.escape(str(c['delta']))}%</p>"
            f"<p><b>Workload:</b> {html.escape(c['workload'])}</p>"
            f"<p><b>Updated:</b> {html.escape(c['updated'])}</p>"
            "</div>"
        )

    result_items = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in result_dist.items())
    module_items = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in module_dist.items())
    problem_items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)

    rec_block = "No recommendation yet"
    if rec:
        rec_block = (
            f"<p><b>配置:</b> {html.escape(rec.get('spec_method', ''))} + {html.escape(rec.get('policy_name', ''))}</p>"
            f"<p><b>Scope:</b> {html.escape(rec.get('workload_profile', ''))}</p>"
            f"<p><b>Risk:</b> {html.escape(rec.get('primary_cost', ''))}</p>"
            f"<p><b>score_main:</b> {html.escape(str(rec.get('score_main', '')))}</p>"
            f"<p><b>merge candidate:</b> {html.escape(rec.get('is_merge_candidate', ''))}</p>"
        )

    latest_table = html_table(
        latest,
        [
            "experiment_id",
            "date",
            "optimization_module",
            "workload_profile",
            "ttft_p95_ms",
            "tpot_p95_ms",
            "throughput_tok_per_s",
            "goodput",
            "result_label",
            "score_main",
            "is_merge_candidate",
        ],
        limit=15,
    )

    trend_json = json.dumps(trend_payload, ensure_ascii=True)

    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Optimization Dashboard</title>
  <style>
    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 24px; color: #1f2937; background: linear-gradient(135deg, #f4f7fb, #eef7f2); }}
    h1 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ background: #f9fafb; }}
    .muted {{ color: #6b7280; font-size: 12px; }}
    #trend {{ width: 100%; height: 360px; }}
    .legend span {{ margin-right: 10px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Optimization Dashboard</h1>
  <p class=\"muted\">Generated at: {html.escape(datetime.now().isoformat(timespec='seconds'))}</p>

  <div class=\"panel\">
    <h2>A. 当前最优结果卡片</h2>
    <div class=\"grid\">{''.join(cards_html)}</div>
  </div>

  <div class=\"panel\">
    <h2>B. 主指标趋势图</h2>
    <canvas id=\"trend\"></canvas>
    <div class=\"legend\">
      <span style=\"color:#1d4ed8\">TTFT p50</span>
      <span style=\"color:#1e40af\">TTFT p95</span>
      <span style=\"color:#b91c1c\">TPOT p50</span>
      <span style=\"color:#7f1d1d\">TPOT p95</span>
      <span style=\"color:#047857\">throughput</span>
      <span style=\"color:#065f46\">goodput</span>
    </div>
  </div>

  <div class=\"panel\">
    <h2>C. 实验结论分布</h2>
    <ul>{result_items or '<li>No data</li>'}</ul>
  </div>

  <div class=\"panel\">
    <h2>D. 模块贡献分布</h2>
    <ul>{module_items or '<li>No data</li>'}</ul>
  </div>

  <div class=\"panel\">
    <h2>E. 当前推荐配置</h2>
    {rec_block}
  </div>

  <div class=\"panel\">
    <h2>F. 当前主要问题</h2>
    <ol>{problem_items}</ol>
  </div>

  <div class=\"panel\">
    <h2>实验台账（最近）</h2>
    {latest_table}
  </div>

  <script>
    const trendData = {trend_json};
    const canvas = document.getElementById('trend');
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.clientWidth * devicePixelRatio;
    const H = canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const pad = 36;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;

    function series(name) {{
      return trendData.map((d, i) => ({{x: i, y: d[name]}})).filter(p => p.y !== null && !Number.isNaN(p.y));
    }}

    const metrics = [
      ['ttft_p50_ms', '#1d4ed8'],
      ['ttft_p95_ms', '#1e40af'],
      ['tpot_p50_ms', '#b91c1c'],
      ['tpot_p95_ms', '#7f1d1d'],
      ['throughput_tok_per_s', '#047857'],
      ['goodput', '#065f46'],
    ];

    let vals = [];
    for (const [m] of metrics) {{
      vals = vals.concat(series(m).map(p => p.y));
    }}
    const minY = vals.length ? Math.min(...vals) : 0;
    const maxY = vals.length ? Math.max(...vals) : 1;

    function tx(i) {{
      if (trendData.length <= 1) return pad;
      return pad + (i / (trendData.length - 1)) * (cw - 2 * pad);
    }}
    function ty(v) {{
      if (maxY === minY) return ch / 2;
      return ch - pad - ((v - minY) / (maxY - minY)) * (ch - 2 * pad);
    }}

    ctx.strokeStyle = '#d1d5db';
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, ch - pad);
    ctx.lineTo(cw - pad, ch - pad);
    ctx.stroke();

    for (const [m, color] of metrics) {{
      const s = series(m);
      if (!s.length) continue;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      s.forEach((p, idx) => {{
        const x = tx(p.x);
        const y = ty(p.y);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optimization decision dashboard")
    parser.add_argument("--registry-csv", default="results/registry/experiment_registry.csv")
    parser.add_argument("--output-html", default="results/figures/optimization_dashboard.html")
    parser.add_argument("--regression-csv", default="results/parsed/regression_stability.csv")
    args = parser.parse_args()

    registry_csv = Path(args.registry_csv)
    rows = load_rows(registry_csv)

    out_html = Path(args.output_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(build_html(rows), encoding="utf-8")

    dump_regression_table(rows, Path(args.regression_csv))

    print(f"dashboard generated: {out_html}")
    print(f"regression table: {args.regression_csv}")


if __name__ == "__main__":
    main()
