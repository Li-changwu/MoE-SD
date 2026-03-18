#!/usr/bin/env python3
"""
Patch vLLM's eagle.py for dynamic K via /dev/shm/moe_sd_k.

Strategy: Read effective_k at propose() start. Run fewer loop iterations.
Pad output to original K with repeated first token (verifier will reject them).
"""

import sys
import shutil
from pathlib import Path

EAGLE_PATH = Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "vllm" / "v1" / "spec_decode" / "eagle.py"
BACKUP_PATH = EAGLE_PATH.with_suffix(".py.orig")

MARKER = "# ── MoE-SD"


def get_lines():
    return EAGLE_PATH.read_text().splitlines(keepends=True)


def is_patched(lines):
    return any(MARKER in l for l in lines)


def find_line(lines, needle, start=0):
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return None


def apply_patch():
    if not EAGLE_PATH.exists():
        print(f"ERROR: {EAGLE_PATH} not found"); sys.exit(1)

    lines = get_lines()
    if is_patched(lines):
        print("Already patched. Use 'revert' first."); return

    if not BACKUP_PATH.exists():
        shutil.copy2(EAGLE_PATH, BACKUP_PATH)
        print(f"Backup: {BACKUP_PATH}")

    # --- PATCH 1: Read effective_k at start of propose() ---
    propose_def = find_line(lines, "def propose(")
    batch_line = find_line(lines, "batch_size = common_attn_metadata.batch_size()", propose_def)
    assert batch_line, "Could not find batch_size line"

    p1 = [
        "        # ── MoE-SD: Read dynamic effective K ──────────────────────\n",
        "        _moe_sd_effective_k = self.num_speculative_tokens\n",
        "        try:\n",
        "            with open('/dev/shm/moe_sd_k', 'r') as _f:\n",
        "                _ek = int(_f.read().strip())\n",
        "                if 1 <= _ek <= self.num_speculative_tokens:\n",
        "                    _moe_sd_effective_k = _ek\n",
        "        except (FileNotFoundError, ValueError, OSError):\n",
        "            pass\n",
        "        # ── End MoE-SD: read ──────────────────────────────────────\n",
    ]
    lines[batch_line:batch_line] = p1
    offset = len(p1)
    print(f"P1: +{offset} lines at {batch_line+1}")

    # --- PATCH 2: Early exit for effective_k==1 ---
    # Insert BEFORE the existing K==1 check
    early_check = find_line(lines, "if self.num_speculative_tokens == 1 or self.parallel_drafting:")
    assert early_check, "Could not find K==1 early exit"

    p2 = [
        "        # ── MoE-SD: Early exit for effective K=1 (pad to full shape) ──\n",
        "        if _moe_sd_effective_k == 1 and not self.parallel_drafting:\n",
        "            _one = self._greedy_sample(sample_hidden_states)\n",
        "            return _one.unsqueeze(1).expand(-1, self.num_speculative_tokens).contiguous()\n",
        "        # ── End MoE-SD: early exit ────────────────────────────────────\n",
    ]
    lines[early_check:early_check] = p2
    offset += len(p2)
    print(f"P2: +{len(p2)} lines at {early_check+1}")

    # --- PATCH 3: Replace loop range ---
    loop_line = find_line(lines, "for token_index in range(self.num_speculative_tokens - 1):")
    assert loop_line, "Could not find draft loop"
    old = lines[loop_line]
    indent = old[:len(old) - len(old.lstrip())]
    lines[loop_line] = f"{indent}for token_index in range(_moe_sd_effective_k - 1):  {MARKER}: effective K loop\n"
    print(f"P3: loop at {loop_line+1}")

    # --- PATCH 3b: Replace the condition before loop ---
    # Original: "if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:"
    cond_line = find_line(lines, "if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:", loop_line - 20)
    if cond_line:
        old_cond = lines[cond_line]
        new_cond = old_cond.replace(
            "self.num_speculative_tokens > 1",
            "_moe_sd_effective_k > 1"
        )
        # Add marker as comment at end (but preserve the colon)
        lines[cond_line] = new_cond
        print(f"P3b: condition at {cond_line+1}")

    # --- PATCH 4: Pad output after stack ---
    stack_line = find_line(lines, "draft_token_ids = torch.stack(draft_token_ids_list, dim=1)", loop_line)
    assert stack_line, "Could not find stack line"
    ret_line = find_line(lines, "return draft_token_ids", stack_line)
    assert ret_line, "Could not find return line"

    p4 = [
        f"        {MARKER}: Pad to original K if effective_k < num_speculative_tokens\n",
        "        if _moe_sd_effective_k < self.num_speculative_tokens:\n",
        "            _pad_n = self.num_speculative_tokens - _moe_sd_effective_k\n",
        "            _pad = draft_token_ids_list[0].unsqueeze(1).expand(-1, _pad_n)\n",
        "            draft_token_ids = torch.cat([draft_token_ids, _pad], dim=1)\n",
        f"        {MARKER}: End pad\n",
    ]
    lines[ret_line:ret_line] = p4
    print(f"P4: pad at {ret_line+1}")

    EAGLE_PATH.write_text("".join(lines))
    print(f"\nPatch applied to {EAGLE_PATH}")


def revert_patch():
    if BACKUP_PATH.exists():
        shutil.copy2(BACKUP_PATH, EAGLE_PATH)
        print(f"Reverted from {BACKUP_PATH}")
    else:
        print("No backup."); sys.exit(1)


def verify_patch():
    lines = get_lines()
    if is_patched(lines):
        print("APPLIED")
        for i, l in enumerate(lines):
            if MARKER in l:
                print(f"  L{i+1}: {l.strip()[:70]}")
    else:
        print("NOT applied")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "apply": apply_patch(); verify_patch()
    elif cmd == "revert": revert_patch()
    elif cmd == "verify": verify_patch()
    else: print("Usage: {apply|revert|verify}")
