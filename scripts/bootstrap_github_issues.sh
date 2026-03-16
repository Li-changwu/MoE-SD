#!/usr/bin/env bash
set -euo pipefail

# Idempotent GitHub bootstrap for delta issues after ISSUE-001..018.
# Requires a GitHub token discoverable from git credential helper.

python3 - <<'PY'
import json
import subprocess
import urllib.request
import urllib.error


def read_token() -> str:
    cred = subprocess.check_output(
        "printf 'protocol=https\\nhost=github.com\\n\\n' | git credential fill",
        shell=True,
        text=True,
    )
    for line in cred.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("No GitHub token found from git credential helper")


def req(method, url, headers, data=None):
    body = None
    h = dict(headers)
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        h["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            txt = r.read().decode("utf-8")
            return r.status, json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8")
        try:
            payload = json.loads(txt) if txt else {}
        except Exception:
            payload = {"raw": txt}
        return e.code, payload


def ensure_label(base, headers, name, color, desc):
    c, p = req("POST", f"{base}/labels", headers, {"name": name, "color": color, "description": desc})
    if c in (200, 201, 422):
        return
    raise RuntimeError(f"ensure_label failed for {name}: {c} {p}")


def issue_exists(base, headers, title):
    c, p = req("GET", f"{base}/issues?state=all&per_page=100", headers)
    if c != 200:
        raise RuntimeError(f"list issues failed: {c} {p}")
    for it in p:
        if it.get("title") == title:
            return it["number"], it["html_url"]
    return None


def ensure_issue(base, headers, title, body, labels):
    ex = issue_exists(base, headers, title)
    if ex:
        return ex[0], ex[1], True
    c, p = req("POST", f"{base}/issues", headers, {
        "title": title,
        "body": body,
        "labels": labels,
    })
    if c not in (200, 201):
        raise RuntimeError(f"create issue failed for {title}: {c} {p}")
    return p["number"], p["html_url"], False


token = read_token()
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Resolve owner dynamically; default repo name aligns with current project.
c, me = req("GET", "https://api.github.com/user", headers)
if c != 200:
    raise RuntimeError(f"Cannot query current user: {c} {me}")
owner = me["login"]
repo = "MoE-SD"
base = f"https://api.github.com/repos/{owner}/{repo}"

# Verify repo exists.
c, _ = req("GET", base, headers)
if c != 200:
    raise RuntimeError(f"Repository {owner}/{repo} not reachable, code={c}")

# Ensure labels used by these delta issues.
ensure_label(base, headers, "p1", "D93F0B", "主线核心，但可依赖 p0 结果稍后开始")
ensure_label(base, headers, "p2", "0E8A16", "锦上添花，或偏论文打磨/工程收尾")
ensure_label(base, headers, "tooling", "7057FF", "工具链")
ensure_label(base, headers, "artifact", "CFD3D7", "artifact产出")
ensure_label(base, headers, "reproducibility", "0E8A16", "可复现性")

issue_19_title = "ISSUE-019：Makefile 与脚本入口统一"
issue_19_body = """## Summary
统一 Makefile 与脚本入口，保证常用研发/实验动作都可通过 `make` 触发。

## Why
工程入口分散会导致命令漂移和复现成本上升；统一入口可显著降低协作和排错成本。

## Scope
包含 make target 规范、脚本命名规范、README 对齐。
不包含策略算法实现。

## Deliverables
- Makefile target 对齐与补全
- scripts/ 下入口脚本命名规范
- docs/make_targets.md（可选）
- README 命令入口更新

## Tasks
- [ ] 盘点现有脚本入口
- [ ] 统一 target 命名（run- / bench- / parse- / clean-）
- [ ] 将常用命令收口到 make
- [ ] `make help` 输出可读说明
- [ ] 更新 README 快速启动段落

## Definition of Done
- [ ] 不需要翻 README 找原始命令
- [ ] 任一成员能通过 `make help` 找到入口
- [ ] baseline 两条链路可由 make 一键触发

## Labels
p1, tooling
"""

issue_20_title = "ISSUE-020：结果目录与命名规范冻结"
issue_20_body = """## Summary
冻结实验结果目录与命名规范，确保 raw/parsed/figures 可自动归档和可追溯。

## Why
若命名口径不统一，后处理和论文出图将依赖人工猜测，极易产生错配。

## Scope
包含路径规范、命名模板、metadata 落盘字段。
不包含新增 benchmark 逻辑。

## Deliverables
- docs/result_naming_convention.md
- results 目录规则（raw/parsed/figures）
- parse 脚本对规范路径的支持

## Naming Contract
路径至少包含：
- model
- method（no_sd / eagle3 / ...）
- workload profile
- timestamp 或 config hash

## Tasks
- [ ] 定义目录层级与文件名模板
- [ ] 在 bench 执行入口中注入规范路径
- [ ] 在解析脚本中读取并保留关键 metadata
- [ ] 补充示例与反例

## Definition of Done
- [ ] 结果路径包含 model / method / workload / timestamp 或 config hash
- [ ] 后处理脚本不需要靠猜文件名
- [ ] 任一结果可回溯到 config 与 commit

## Labels
p2, artifact, reproducibility
"""

for title, body, labels in [
    (issue_19_title, issue_19_body, ["p1", "tooling"]),
    (issue_20_title, issue_20_body, ["p2", "artifact", "reproducibility"]),
]:
    num, url, existed = ensure_issue(base, headers, title, body, labels)
    print(("EXISTS" if existed else "CREATED"), f"#{num}", url)
PY
