#!/usr/bin/env python3
"""
Figure 4: BriskMoE System Architecture — v5 (two-phase design).

Changes from v4:
  - Remove "(1) Enable" column — cache pool is now a precondition (M* threshold)
  - Two-column layout: (1) Unblock,  (2) Protect
  - Expert cache pool shown as a shared substrate band between design modules and CPU
  - Language aligned with §3 (Obs 1 / Obs 2, no "regime" terminology)
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
C_UB  = '#548235'   # Unblock (green)
C_PR  = '#C00000'   # Protect (red)
C_PC  = '#D4812A'   # PCIe    (orange)
C_CA  = '#2E5C8A'   # Cache   (blue)
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
fig, ax = plt.subplots(figsize=(7.5, 5.8))
ax.set_xlim(0, 7.5)
ax.set_ylim(-0.1, 5.85)
ax.set_aspect('equal')
ax.axis('off')


# ══════════════════════════════════════════════════════════════════════
# GPU zone background
# ══════════════════════════════════════════════════════════════════════
gpu = FancyBboxPatch((0.05, 0.78), 7.40, 4.95, boxstyle="round,pad=0.05",
                     facecolor='#F0F5FA', edgecolor=C_CA,
                     lw=1.8, alpha=0.35, zorder=0)
ax.add_patch(gpu)
ax.text(0.15, 5.62, 'GPU  (RTX A6000, 48 GB HBM)',
        fontsize=9, fontweight='bold', color=C_CA, va='top', zorder=1)


# ══════════════════════════════════════════════════════════════════════
# TOP BAND: SD Pipeline    (y ≈ 4.90 – 5.45)
# ══════════════════════════════════════════════════════════════════════
# Draft
rbox(ax, 0.30, 4.92, 1.60, 0.48, 'Draft Model\n(EAGLE-3)',
     fc='#E2EFDA', ec=C_UB, fs=7.5, fw='bold', lw=1.3)
arr(ax, 1.90, 5.16, 2.55, 5.16, lw=1.5)
ax.text(2.23, 5.28, '$K$=3 tokens', fontsize=6.5, ha='center', color=C_DK)

# Target
rbox(ax, 2.55, 4.92, 2.05, 0.48, 'Target Model\n(Qwen3-30B-A3B)',
     fc='#D6E4F0', ec='#2F5496', fs=7.5, fw='bold', lw=1.3)
arr(ax, 4.60, 5.16, 5.20, 5.16, lw=1.5)
ax.text(4.90, 5.28, 'verify', fontsize=6.5, ha='center', color=C_DK)

# Output
rbox(ax, 5.20, 4.96, 1.10, 0.40, '~2.5 tok/step',
     fc='#F2F2F2', ec='#666', fs=7.5, fw='bold', lw=0.8)

# Loop-back
ax.annotate('', xy=(0.30, 5.58), xytext=(6.30, 5.58),
            arrowprops=dict(arrowstyle='->', color='#888',
                            connectionstyle='arc3,rad=-0.15',
                            lw=0.8, linestyle='--'))
ax.text(3.30, 5.72, 'next draft round', fontsize=5.5,
        ha='center', color='#888', style='italic')


# ── Thin separator ────────────────────────────────────────────────────
ax.plot([0.18, 7.32], [4.75, 4.75], color='#bbb', lw=0.5, ls=':', zorder=1)


# ══════════════════════════════════════════════════════════════════════
# Arrow:  Target → "needs MoE experts" → two design zones
# ══════════════════════════════════════════════════════════════════════
arr(ax, 3.57, 4.92, 3.57, 4.50, color='#666', lw=1.0)
ax.text(3.92, 4.63, 'MoE layer\nexecution', fontsize=5.5,
        ha='left', va='center', color='#666', style='italic')


# ══════════════════════════════════════════════════════════════════════
# COLUMN 1  —  (1) Unblock               (x 0.15 – 3.85)
# ══════════════════════════════════════════════════════════════════════
c1x, c1w, c1y, c1h = 0.15, 3.70, 2.10, 2.30
col1 = FancyBboxPatch((c1x, c1y), c1w, c1h, boxstyle="round,pad=0.04",
                       facecolor='#EDF2E5', edgecolor=C_UB,
                       lw=1.4, alpha=0.50, zorder=1)
ax.add_patch(col1)

badge(ax, c1x + c1w/2, c1y + c1h - 0.12, '(1) Unblock', C_UB)
ax.text(c1x + c1w/2, c1y + c1h - 0.42, 'Cache-Path Optimizations  (Obs. 1)',
        fontsize=7.5, fontweight='bold', ha='center', color=C_UB, zorder=3)

# Four boxes — 2×2 grid layout
opts = [
    ('Pool-Direct',      r'elim. $T_{copy}$'),
    ('TASER',            r'elim. $T_{sync}$'),
    ('Oracle Prefetch',  r'overlap $T_{load}$'),
    ('Fused MoE Kernel', 'tuned kernel'),
]

bw_ub = 1.50
bh_ub = 0.52
# Row 1 (top): Pool-Direct, TASER
# Row 2 (bottom): Oracle Prefetch, Fused MoE Kernel
row1_y = c1y + c1h - 1.20
row2_y = c1y + 0.22
col1_bx = c1x + 0.20
col2_bx = c1x + c1w/2 + 0.10

positions = [
    (col1_bx, row1_y),   # Pool-Direct
    (col2_bx, row1_y),   # TASER
    (col1_bx, row2_y),   # Oracle Prefetch
    (col2_bx, row2_y),   # Fused MoE Kernel
]

for (name, desc), (bx, by) in zip(opts, positions):
    rbox(ax, bx, by, bw_ub, bh_ub, f'{name}\n{desc}',
         fc='#E2EFDA', ec=C_UB, fs=7, fw='bold', lw=1.1, tc='#2D5016')

# Arrows: row 1 → row 2 (Pool-Direct → Oracle, TASER → Kernel)
for bx_col in [col1_bx, col2_bx]:
    arr(ax, bx_col + bw_ub/2, row1_y, bx_col + bw_ub/2, row2_y + bh_ub,
        color=C_UB, lw=1.0)


# ══════════════════════════════════════════════════════════════════════
# COLUMN 2  —  (2) Protect               (x 4.10 – 7.35)
# ══════════════════════════════════════════════════════════════════════
c2x, c2w, c2y, c2h = 4.10, 3.25, 2.10, 2.30
col2bg = FancyBboxPatch((c2x, c2y), c2w, c2h, boxstyle="round,pad=0.04",
                       facecolor='#FDE8E0', edgecolor=C_PR,
                       lw=1.4, alpha=0.50, zorder=1)
ax.add_patch(col2bg)

badge(ax, c2x + c2w/2, c2y + c2h - 0.12, '(2) Protect', C_PR)
ax.text(c2x + c2w/2, c2y + c2h - 0.50, 'Draft-Guided Preloading\n(Obs. 2)',
        fontsize=7.5, fontweight='bold', ha='center', color=C_PR, zorder=3,
        linespacing=1.1)

# -- Upper box: mechanism --
rbox(ax, c2x + 0.15, 2.90, c2w - 0.30, 0.65,
     'Use target router\nweights on draft\nhidden states',
     fc='#FBE5D6', ec=C_PR, fs=7, fw='bold', lw=1.1, tc='#6B1010')

# -- Lower box: goal --
rbox(ax, c2x + 0.15, 2.15, c2w - 0.30, 0.55,
     'Prevent cascade eviction\n' + r'maintain $\eta_{SD} \approx \eta_{AR}$',
     fc='#FBE5D6', ec=C_PR, fs=7, fw='bold', lw=1.1, tc='#6B1010')

# Arrow between the two Protect boxes
arr(ax, c2x + c2w/2, 2.90, c2x + c2w/2, 2.70, color=C_PR, lw=0.9)

# Cost / accuracy note
ax.text(c2x + c2w - 0.15, c2y + 0.08,
        '~72 $\\mu$s/step, 70--85% accuracy',
        fontsize=5.5, ha='right', color=C_PR, style='italic', zorder=4)


# ══════════════════════════════════════════════════════════════════════
# CACHE POOL BAND  (shared substrate, y ≈ 1.20 – 2.00)
# ══════════════════════════════════════════════════════════════════════
cache_y, cache_h = 1.20, 0.78
cache_bg = FancyBboxPatch((0.15, cache_y), 7.20, cache_h,
                           boxstyle="round,pad=0.04",
                           facecolor='#DAE8FC', edgecolor=C_CA,
                           lw=1.4, alpha=0.50, zorder=1)
ax.add_patch(cache_bg)

ax.text(0.30, cache_y + cache_h - 0.10,
        'Expert Cache Pool  (precondition: $M_{GPU} \\geq M^* \\approx 40$ GiB)',
        fontsize=7.5, fontweight='bold', color=C_CA, va='top', zorder=3)

# ── Cache slot grid (2 rows × 8 cols, compact) ──
hit, pf, emp = '#4472C4', '#70AD47', '#CCCCCC'
slots = ([hit]*6 + [pf, pf]
       + [hit]*5 + [pf, emp, hit])
sw, sh = 0.26, 0.16
gx0, gy0 = 0.40, cache_y + 0.10
for i, sc in enumerate(slots):
    r, c = divmod(i, 8)
    sx = gx0 + c * (sw + 0.05)
    sy = gy0 + (1 - r) * (sh + 0.06)
    ax.add_patch(Rectangle((sx, sy), sw, sh,
                            facecolor=sc, edgecolor='white', lw=0.6,
                            alpha=0.85 if sc != emp else 0.35, zorder=3))

# Legend
for ci, (clr, lbl) in enumerate([(hit, 'hit'), (pf, 'prefetch'), (emp, 'empty')]):
    lx = 3.30 + ci * 0.65
    ax.add_patch(Rectangle((lx, cache_y + 0.12), 0.10, 0.10,
                            facecolor=clr, edgecolor='#666', lw=0.3,
                            alpha=0.8, zorder=3))
    ax.text(lx + 0.14, cache_y + 0.17, lbl, fontsize=5.5, va='center', zorder=4)

# Formula + HBM badge (right side)
ax.text(5.60, cache_y + 0.45,
        r'$S = \lfloor C/(L_{off}\!\cdot\! m_e)\rfloor \approx 17$ slots/layer',
        fontsize=6.5, ha='center', color=C_CA, style='italic', zorder=4)
ax.text(5.60, cache_y + 0.18, 'LRU replacement, HBM 768 GB/s',
        fontsize=5.5, ha='center', color=C_CA, zorder=4)


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
# CROSS-ZONE ARROWS
# ══════════════════════════════════════════════════════════════════════

# ① CPU → Cache Pool  (PCIe)
arr(ax, 3.75, 0.68, 3.75, cache_y,
    color=C_PC, lw=2.5, style='-|>')
ax.text(4.45, 0.88, 'PCIe Gen4\n25 GB/s',
        fontsize=6.5, ha='left', color=C_PC, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                  edgecolor=C_PC, lw=0.5), zorder=6)

# ② Cache Pool → Unblock  (HBM read)
arr(ax, 1.80, cache_y + cache_h, 1.80, c1y,
    color=C_CA, lw=1.6)
ax.text(1.80, cache_y + cache_h + 0.03,
        'HBM read', fontsize=6, ha='center', va='bottom',
        color=C_CA, fontweight='bold', zorder=6)

# ③ Unblock → back to Target  (MoE output)
arr(ax, 2.00, c1y + c1h, 3.10, 4.92,
    color='#555', lw=1.0, conn='arc3,rad=0.15')
ax.text(1.60, 4.60, 'MoE\noutput',
        fontsize=5.5, ha='center', va='center', color='#555',
        bbox=dict(boxstyle='round,pad=0.04', facecolor='white',
                  edgecolor='none', alpha=0.8), zorder=6)

# ④ Draft Model → Protect  (draft hidden states → target router)
arr(ax, 1.10, 4.92, c2x + 0.50, c2y + c2h,
    color=C_PR, lw=1.1, ls='--', conn='arc3,rad=-0.22')
ax.text(4.00, 4.60,
        r'draft hidden states $\rightarrow$ target router',
        fontsize=5.5, ha='center', color=C_PR, style='italic',
        bbox=dict(boxstyle='round,pad=0.06', facecolor='white',
                  edgecolor=C_PR, alpha=0.90, lw=0.4), zorder=8)

# ⑤ Protect → Cache Pool  (preload experts)
arr(ax, c2x + c2w/2, c2y, c2x + c2w/2, cache_y + cache_h,
    color=C_PR, lw=1.6)
ax.text(c2x + c2w/2 + 0.15, (c2y + cache_y + cache_h) / 2,
        'preload\nexperts', fontsize=5.5, ha='left', va='center',
        color=C_PR, fontweight='bold', zorder=6)


# ══════════════════════════════════════════════════════════════════════
plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, 'fig_architecture.pdf'))
fig.savefig(os.path.join(out_dir, 'fig_architecture_preview.png'), dpi=150)
print('  OK fig_architecture.pdf + preview')
plt.close(fig)
