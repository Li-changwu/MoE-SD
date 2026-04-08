#!/usr/bin/env python3
"""
Figure 4: BriskMoE System Architecture — Redesign v4.

Key change from v3: Clear three-COLUMN layout.
  - Each phase (Enable / Unblock / Protect) gets its own colored zone
  - SD pipeline sits on top as a separate horizontal band
  - CPU zone at bottom
  - Main data-flow arrows are clear and non-overlapping
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})

# ── Palette ───────────────────────────────────────────────────────────
C_EN  = '#2E5C8A'   # Enable  (blue)
C_UB  = '#548235'   # Unblock (green)
C_PR  = '#C00000'   # Protect (red)
C_PC  = '#D4812A'   # PCIe    (orange)
C_DK  = '#333333'   # general dark


# ── Helpers ───────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, label, fc='white', ec='black',
         fs=7.5, fw='normal', lw=1.0, tc='black', zorder=3, ls=1.15):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1,
            linespacing=ls)


def badge(ax, x, y, label, color, fs=9):
    ax.text(x, y, label, fontsize=fs, fontweight='bold', ha='center',
            va='center', color='white', zorder=7,
            bbox=dict(boxstyle='round,pad=0.22', facecolor=color,
                      edgecolor='none', alpha=0.92))


def arr(ax, x0, y0, x1, y1, color=C_DK, lw=1.3, style='->',
        ls='-', conn=None, zorder=5):
    kw = dict(arrowstyle=style, color=color, lw=lw, linestyle=ls)
    if conn:
        kw['connectionstyle'] = conn
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=kw, zorder=zorder)


# ── Canvas ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))
ax.set_xlim(0, 7.5)
ax.set_ylim(-0.1, 5.45)
ax.set_aspect('equal')
ax.axis('off')


# ══════════════════════════════════════════════════════════════════════
# GPU zone background
# ══════════════════════════════════════════════════════════════════════
gpu = FancyBboxPatch((0.05, 0.78), 7.40, 4.55, boxstyle="round,pad=0.05",
                     facecolor='#F0F5FA', edgecolor=C_EN,
                     lw=1.8, alpha=0.35, zorder=0)
ax.add_patch(gpu)
ax.text(0.15, 5.22, 'GPU  (RTX A6000, 48 GB HBM)',
        fontsize=9, fontweight='bold', color=C_EN, va='top', zorder=1)


# ══════════════════════════════════════════════════════════════════════
# TOP BAND: SD Pipeline    (y ≈ 4.50 – 5.05)
# ══════════════════════════════════════════════════════════════════════
# Draft
rbox(ax, 0.30, 4.52, 1.60, 0.48, 'Draft Model\n(EAGLE-3)',
     fc='#E2EFDA', ec=C_UB, fs=7.5, fw='bold', lw=1.3)
arr(ax, 1.90, 4.76, 2.55, 4.76, lw=1.5)
ax.text(2.23, 4.88, '$K$=3 tokens', fontsize=6.5, ha='center', color=C_DK)

# Target
rbox(ax, 2.55, 4.52, 2.05, 0.48, 'Target Model\n(Qwen3-30B-A3B)',
     fc='#D6E4F0', ec='#2F5496', fs=7.5, fw='bold', lw=1.3)
arr(ax, 4.60, 4.76, 5.20, 4.76, lw=1.5)
ax.text(4.90, 4.88, 'verify', fontsize=6.5, ha='center', color=C_DK)

# Output
rbox(ax, 5.20, 4.56, 1.10, 0.40, '~2.5 tok/step',
     fc='#F2F2F2', ec='#666', fs=7.5, fw='bold', lw=0.8)

# Loop-back
ax.annotate('', xy=(0.30, 5.18), xytext=(6.30, 5.18),
            arrowprops=dict(arrowstyle='->', color='#888',
                            connectionstyle='arc3,rad=-0.15',
                            lw=0.8, linestyle='--'))
ax.text(3.30, 5.32, 'next draft round', fontsize=5.5,
        ha='center', color='#888', style='italic')


# ── Thin separator ────────────────────────────────────────────────────
ax.plot([0.18, 7.32], [4.35, 4.35], color='#bbb', lw=0.5, ls=':', zorder=1)


# ══════════════════════════════════════════════════════════════════════
# Arrow:  Target → "needs MoE experts" → three zones
# ══════════════════════════════════════════════════════════════════════
arr(ax, 3.57, 4.52, 3.57, 4.10, color='#666', lw=1.0)
ax.text(3.92, 4.23, 'MoE layer\nexecution', fontsize=5.5,
        ha='left', va='center', color='#666', style='italic')


# ══════════════════════════════════════════════════════════════════════
# COLUMN 1  —  (1) Enable                (x 0.15 – 2.35)
# ══════════════════════════════════════════════════════════════════════
c1x, c1w, c1y, c1h = 0.15, 2.20, 0.88, 3.10
col1 = FancyBboxPatch((c1x, c1y), c1w, c1h, boxstyle="round,pad=0.04",
                       facecolor='#DAE8FC', edgecolor=C_EN,
                       lw=1.4, alpha=0.50, zorder=1)
ax.add_patch(col1)

badge(ax, c1x + c1w/2, c1y + c1h - 0.12, '(1) Enable', C_EN)
ax.text(c1x + c1w/2, c1y + c1h - 0.42, 'Expert Cache Pool',
        fontsize=8.5, fontweight='bold', ha='center', color=C_EN, zorder=3)

# ── Cache slot grid  (3 rows × 5 cols) ──
hit, pf, emp = '#4472C4', '#70AD47', '#CCCCCC'
slots = ([hit]*4 + [pf]
       + [hit]*2 + [pf, emp, hit]
       + [hit, pf] + [hit]*3)
sw, sh = 0.27, 0.17
gx0, gy0 = c1x + 0.22, c1y + 1.60
for i, sc in enumerate(slots):
    r, c = divmod(i, 5)
    sx = gx0 + c * (sw + 0.06)
    sy = gy0 + (2 - r) * (sh + 0.07)
    ax.add_patch(Rectangle((sx, sy), sw, sh,
                            facecolor=sc, edgecolor='white', lw=0.6,
                            alpha=0.85 if sc != emp else 0.35, zorder=3))

# Legend
for ci, (clr, lbl) in enumerate([(hit, 'hit'), (pf, 'prefetch'), (emp, 'empty')]):
    lx = c1x + 0.25 + ci * 0.65
    ax.add_patch(Rectangle((lx, c1y + 1.42), 0.10, 0.10,
                            facecolor=clr, edgecolor='#666', lw=0.3,
                            alpha=0.8, zorder=3))
    ax.text(lx + 0.14, c1y + 1.47, lbl, fontsize=5.5, va='center', zorder=4)

# Formula
ax.text(c1x + c1w/2, c1y + 1.10,
        r'$S = \lfloor C/(L_{off}\!\cdot\! m_e)\rfloor \approx 17$ slots/layer',
        fontsize=6.5, ha='center', color=C_EN, style='italic', zorder=4)

# HBM badge
ax.text(c1x + c1w/2, c1y + 0.65, 'HBM  768 GB/s',
        fontsize=7, ha='center', fontweight='bold', color=C_EN, zorder=4,
        bbox=dict(boxstyle='round,pad=0.10', facecolor='white',
                  edgecolor=C_EN, lw=0.6))

# Description
ax.text(c1x + c1w/2, c1y + 0.25,
        'PCIe $\\rightarrow$ HBM regime shift',
        fontsize=6.5, ha='center', color=C_EN, zorder=4)


# ══════════════════════════════════════════════════════════════════════
# COLUMN 2  —  (2) Unblock               (x 2.55 – 5.05)
# ══════════════════════════════════════════════════════════════════════
c2x, c2w, c2y, c2h = 2.55, 2.50, 0.88, 3.10
col2 = FancyBboxPatch((c2x, c2y), c2w, c2h, boxstyle="round,pad=0.04",
                       facecolor='#EDF2E5', edgecolor=C_UB,
                       lw=1.4, alpha=0.50, zorder=1)
ax.add_patch(col2)

badge(ax, c2x + c2w/2, c2y + c2h - 0.12, '(2) Unblock', C_UB)
ax.text(c2x + c2w/2, c2y + c2h - 0.42, 'Cache-Path Optimizations',
        fontsize=8, fontweight='bold', ha='center', color=C_UB, zorder=3)

# Four boxes — vertical chain
opts = [
    ('Pool-Direct',      r'elim. $T_{copy}$'),
    ('TASER',            r'elim. $T_{sync}$'),
    ('Oracle Prefetch',  r'overlap $T_{load}$'),
    ('Fused MoE Kernel', 'tuned kernel'),
]

bx   = c2x + 0.15         # box left
bw   = c2w - 0.30         # box width
bh   = 0.38               # box height
# Vertical positions (top → bottom):
#   first top = c2y + c2h - 0.60 = 3.38
#   gap = 0.24
#   box 1: y=3.00, top=3.38
#   box 2: y=2.38, top=2.76
#   box 3: y=1.76, top=2.14
#   box 4: y=1.14, top=1.52
bys = [3.00, 2.38, 1.76, 1.14]

for (name, desc), by in zip(opts, bys):
    rbox(ax, bx, by, bw, bh, f'{name}\n{desc}',
         fc='#E2EFDA', ec=C_UB, fs=6.8, fw='bold', lw=1.1, tc='#2D5016')

# Chain arrows between consecutive boxes
mid_x = bx + bw / 2
for i in range(3):
    arr(ax, mid_x, bys[i], mid_x, bys[i+1] + bh,
        color=C_UB, lw=1.0)


# ══════════════════════════════════════════════════════════════════════
# COLUMN 3  —  (3) Protect               (x 5.25 – 7.35)
# ══════════════════════════════════════════════════════════════════════
c3x, c3w, c3y, c3h = 5.25, 2.10, 0.88, 3.10
col3 = FancyBboxPatch((c3x, c3y), c3w, c3h, boxstyle="round,pad=0.04",
                       facecolor='#FDE8E0', edgecolor=C_PR,
                       lw=1.4, alpha=0.50, zorder=1)
ax.add_patch(col3)

badge(ax, c3x + c3w/2, c3y + c3h - 0.12, '(3) Protect', C_PR)
ax.text(c3x + c3w/2, c3y + c3h - 0.50, 'Draft-Guided\nPreloading',
        fontsize=8, fontweight='bold', ha='center', color=C_PR, zorder=3,
        linespacing=1.1)

# -- Upper box: mechanism --
rbox(ax, c3x + 0.10, 2.65, c3w - 0.20, 0.65,
     'Use target router\nweights on draft\nhidden states',
     fc='#FBE5D6', ec=C_PR, fs=7, fw='bold', lw=1.1, tc='#6B1010')

# -- Lower box: goal --
rbox(ax, c3x + 0.10, 1.60, c3w - 0.20, 0.65,
     r'Maintain $\eta_{SD}$' + '\nunder tight\nmemory budget',
     fc='#FBE5D6', ec=C_PR, fs=7, fw='bold', lw=1.1, tc='#6B1010')

# Arrow between the two Protect boxes
arr(ax, c3x + c3w/2, 2.65, c3x + c3w/2, 2.25, color=C_PR, lw=0.9)

# Cost / accuracy note
ax.text(c3x + c3w/2, c3y + 0.30,
        '~72 $\\mu$s/step\n70--85 % accuracy',
        fontsize=6, ha='center', color=C_PR, style='italic', zorder=4)


# ══════════════════════════════════════════════════════════════════════
# BOTTOM BAND:  CPU zone                  (y -0.02 – 0.68)
# ══════════════════════════════════════════════════════════════════════
cpu = FancyBboxPatch((0.05, -0.02), 7.40, 0.70, boxstyle="round,pad=0.04",
                      facecolor='#FFF3E0', edgecolor='#8B5E3C',
                      lw=1.6, alpha=0.45, zorder=0)
ax.add_patch(cpu)
ax.text(0.15, 0.60, 'CPU  (Host DRAM)',
        fontsize=9, fontweight='bold', color='#8B5E3C', va='top', zorder=1)

# Expert-layer blocks
n = 13
bwidth, gapx = 0.46, 0.06
total_w = n * bwidth + (n - 1) * gapx
sx0 = (7.50 - total_w) / 2
for i in range(n):
    ex = sx0 + i * (bwidth + gapx)
    ax.add_patch(FancyBboxPatch((ex, 0.05), bwidth, 0.32,
                                boxstyle="round,pad=0.02",
                                facecolor='#F4B183', edgecolor='#8B5E3C',
                                linewidth=0.5, alpha=0.7, zorder=2))
    ax.text(ex + bwidth/2, 0.21,
            f'L{i*2}-{min(i*2+1, 25)}',
            fontsize=5, ha='center', va='center',
            color='#4A3520', fontweight='bold', zorder=3)

ax.text(7.50/2, 0.47,
        '26 offloaded MoE layers  (128 experts/layer, 9.44 MB each)',
        fontsize=6.5, ha='center', va='bottom',
        color='#6B4226', fontweight='bold', zorder=3)


# ══════════════════════════════════════════════════════════════════════
# CROSS-ZONE ARROWS  (the 5 key data-flow connections)
# ══════════════════════════════════════════════════════════════════════

# ① CPU → Cache Pool  (PCIe)
arr(ax, c1x + c1w/2, 0.68, c1x + c1w/2, 0.88,
    color=C_PC, lw=2.5, style='-|>')
ax.text(c1x + c1w/2 + 0.55, 0.75, 'PCIe Gen4\n25 GB/s',
        fontsize=6.5, ha='left', color=C_PC, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                  edgecolor=C_PC, lw=0.5), zorder=6)

# ② Cache Pool → first Unblock box  (HBM read)
pool_mid_y = bys[0] + bh / 2          # midpoint of Pool-Direct box
arr(ax, c1x + c1w, pool_mid_y, bx, pool_mid_y,
    color=C_EN, lw=1.6)
ax.text((c1x + c1w + bx) / 2, pool_mid_y + 0.12,
        'HBM read', fontsize=6, ha='center', color=C_EN, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.05', facecolor='white',
                  edgecolor='none', alpha=0.85), zorder=6)

# ③ Unblock chain end → back to Target  (MoE output)
#    A small upward arrow on the right side of Unblock column
ret_x = c2x + c2w - 0.15
arr(ax, ret_x, bys[0] + bh, 4.10, 4.52,
    color='#555', lw=1.0, conn='arc3,rad=-0.15')
ax.text(c2x + c2w + 0.08, 3.52, 'MoE\noutput',
        fontsize=5.5, ha='left', va='center', color='#555',
        bbox=dict(boxstyle='round,pad=0.04', facecolor='white',
                  edgecolor='none', alpha=0.8), zorder=6)

# ④ Draft Model → Protect  (draft hidden states → target router)
#    Dashed red arc from Draft bottom → to Protect column top area
arr(ax, 1.10, 4.52, c3x + 0.30, c3y + c3h - 0.55,
    color=C_PR, lw=1.1, ls='--', conn='arc3,rad=-0.15')
ax.text(2.50, 4.38,
        r'draft hidden states $\rightarrow$ target router',
        fontsize=5.5, ha='center', color=C_PR, style='italic',
        bbox=dict(boxstyle='round,pad=0.06', facecolor='white',
                  edgecolor=C_PR, alpha=0.90, lw=0.4), zorder=8)

# ⑤ Protect → Cache Pool  (preload)
#    Red arc going BELOW the Unblock column so it doesn't cross the boxes
arr(ax, c3x, 1.25, c1x + c1w, 1.20,
    color=C_PR, lw=1.4, conn='arc3,rad=0.22')
ax.text((c1x + c1w + c3x) / 2, 0.82,
        'preload experts into cache',
        fontsize=6.5, ha='center', color=C_PR, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.06', facecolor='white',
                  edgecolor=C_PR, alpha=0.90, lw=0.5), zorder=8)


# ══════════════════════════════════════════════════════════════════════
plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, 'fig_architecture.pdf'))
fig.savefig(os.path.join(out_dir, 'fig_architecture_preview.png'), dpi=150)
print('  OK fig_architecture.pdf')
plt.close(fig)
