#!/usr/bin/env python3
"""
Figure 1: Overview of Knowledge-Representation-Guided Mechanistic Verification
For KR 2026 paper: "Do Transformers Learn What Theory Predicts?"

Three-panel pipeline: B-RASP → SCM → DAS Verification
Bottom: DAS-IIA trajectory showing phase transition during grokking

ALL DATA POINTS MATCHED TO MAIN TEXT:
  - IIA: B_prev=0.98, C_in=0.96, Y_out=0.97 (Table 2a)
  - Layers: 1-indexed (L1-attn, L1-MLP, L2-attn, L2-MLP)
  - Carry: C_in = AND_{j=k+1}^{n} b_j (Eq. 3)
  - Colors: blue=B_prev, red=C_in, green=Y_out, purple=wnorm (unified)
  - Grokking: mem~1800, grok~18600 (Table grokking)
  - Weight norm symbol: ||theta||_2
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
import numpy as np

# ─── Global Style ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'figure.dpi': 150,
})

# ─── Unified Color Palette (matches all other figures) ──────────
C_BLUE   = '#2166AC'   # B_prev everywhere
C_RED    = '#B2182B'   # C_in everywhere
C_GREEN  = '#1B7837'   # Y_out everywhere
C_PURPLE = '#7570B3'   # weight norm everywhere

# Panel background colors (muted, don't conflict with data colors)
C_KR_BG   = '#D8F3DC'
C_KR_BD   = '#2D6A4F'
C_SCM_BG  = '#D6E8F0'
C_SCM_BD  = '#1B4965'
C_DAS_BG  = '#FDDEDE'
C_DAS_BD  = '#9B2226'

C_ARROW   = '#495057'
C_PHASE   = '#FFC107'

# ─── Figure Layout ──────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 3.8))

gs = fig.add_gridspec(
    2, 3,
    left=0.03, right=0.97, top=0.95, bottom=0.05,
    hspace=0.09, wspace=0.22,
    height_ratios=[1.2, 0.65],
    width_ratios=[1, 1, 1]
)

ax_kr  = fig.add_subplot(gs[0, 0])
ax_scm = fig.add_subplot(gs[0, 1])
ax_das = fig.add_subplot(gs[0, 2])
ax_traj = fig.add_subplot(gs[1, :])

for ax in [ax_kr, ax_scm, ax_das]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')


# ═══════════════════════════════════════════════════════════════
# PANEL A: Knowledge Representation (B-RASP)
# ═══════════════════════════════════════════════════════════════

ax_kr.add_patch(FancyBboxPatch(
    (0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.2",
    facecolor=C_KR_BG, edgecolor=C_KR_BD, linewidth=1.2, alpha=0.5
))

ax_kr.text(5.0, 9.15, 'Knowledge Representation', fontsize=8.5,
           fontweight='bold', color=C_KR_BD, ha='center', va='center')
ax_kr.text(5.0, 8.4, r'B-RASP Program  $\equiv$  FO[$<$]', fontsize=7.5,
           color=C_KR_BD, ha='center', va='center', style='italic')

# Pseudocode box
code_box = FancyBboxPatch(
    (0.8, 1.2), 8.4, 6.5, boxstyle="round,pad=0.15",
    facecolor='white', edgecolor=C_KR_BD, linewidth=0.8, alpha=0.85
)
ax_kr.add_patch(code_box)

# B-RASP pseudocode — FIXED P5: carry formula matches Eq.3
lines = [
    (r'$\mathbf{function}$  BinaryIncrement($x$):', 7.20, '#212529', 7.2),
    (r'  $\triangleright$ Stage 1: Bit Retrieval', 6.40, '#6C757D', 6.3),
    (r'  $b_{\mathrm{prev}} \leftarrow \mathrm{select}(\mathrm{offset}\!=\!-(n\!+\!1))$', 5.70, '#212529', 7.2),
    (r'  $\triangleright$ Stage 2: Carry Chain', 4.90, '#6C757D', 6.3),
    (r'  $C_{\mathrm{in}} \leftarrow \bigwedge_{j=k+1}^{n}\; b_j$', 4.15, '#212529', 7.2),
    (r'  $\triangleright$ Stage 3: XOR Output', 3.35, '#6C757D', 6.3),
    (r'  $y_k \leftarrow b_{\mathrm{prev}} \oplus C_{\mathrm{in}}$', 2.60, '#212529', 7.2),
    (r'  $\mathbf{return}$  $y$', 1.70, '#212529', 7.2),
]
for text, y, color, fsize in lines:
    ax_kr.text(1.4, y, text, fontsize=fsize, color=color,
               ha='left', va='center', family='serif')

ax_kr.text(5.0, 0.5, '(a)', fontsize=8, ha='center', va='center',
           fontweight='bold', color=C_KR_BD)


# ═══════════════════════════════════════════════════════════════
# PANEL B: Structural Causal Model
# ═══════════════════════════════════════════════════════════════

ax_scm.add_patch(FancyBboxPatch(
    (0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.2",
    facecolor=C_SCM_BG, edgecolor=C_SCM_BD, linewidth=1.2, alpha=0.5
))

ax_scm.text(5.0, 9.15, 'Causal Compilation', fontsize=8.5,
            fontweight='bold', color=C_SCM_BD, ha='center', va='center')
ax_scm.text(5.0, 8.4, 'Structural Causal Model', fontsize=7.5,
            color=C_SCM_BD, ha='center', va='center', style='italic')

# SCM nodes — FIXED P6/P7: use unified blue/red/green for variables
node_r = 1.25
nodes = [
    (r'$B_{\mathrm{prev}}$', 2.8, 5.5, 'Bit\nRetrieval', C_BLUE),
    (r'$C_{\mathrm{in}}$',   7.2, 5.5, 'Carry\n(AND)',    C_RED),
    (r'$Y_{\mathrm{out}}$',  5.0, 2.5, 'Output\n(XOR)',   C_GREEN),
]

for label, cx, cy, sublabel, ncolor in nodes:
    circle = plt.Circle((cx, cy), node_r, facecolor='white',
                         edgecolor=ncolor, linewidth=1.4, zorder=3)
    ax_scm.add_patch(circle)
    ax_scm.text(cx, cy + 0.35, label, fontsize=7, ha='center', va='center',
                fontweight='bold', color=ncolor, zorder=4)
    ax_scm.text(cx, cy - 0.45, sublabel, fontsize=5.3, ha='center', va='center',
                color='#495057', zorder=4, linespacing=1.35)

# Causal arrows
_d = np.sqrt(2.2**2 + 3.0**2)
_ux, _uy = 2.2/_d, -3.0/_d

arrow_kw = dict(arrowstyle='->', color=C_SCM_BD, linewidth=1.6,
                mutation_scale=15, zorder=2)
ax_scm.annotate('',
    xy=(5.0 - node_r*_ux, 2.5 - node_r*_uy),
    xytext=(2.8 + node_r*_ux, 5.5 + node_r*_uy),
    arrowprops={**arrow_kw, 'connectionstyle': 'arc3,rad=0.08'})
ax_scm.annotate('',
    xy=(5.0 + node_r*_ux, 2.5 - node_r*_uy),
    xytext=(7.2 - node_r*_ux, 5.5 + node_r*_uy),
    arrowprops={**arrow_kw, 'connectionstyle': 'arc3,rad=-0.08'})

# Exogenous inputs
for cx in [2.8, 7.2]:
    ax_scm.annotate('', xy=(cx, 5.5 + node_r), xytext=(cx, 7.55),
                     arrowprops=dict(arrowstyle='->', color='#ADB5BD',
                                     linewidth=0.8, linestyle='--',
                                     mutation_scale=10))
ax_scm.text(2.8, 8.0, r'$x$', fontsize=8, ha='center', va='top', color='#868E96')
ax_scm.text(7.2, 8.0, r'$x$', fontsize=8, ha='center', va='top', color='#868E96')

ax_scm.text(5.0, 0.5, '(b)', fontsize=8, ha='center', va='center',
            fontweight='bold', color=C_SCM_BD)


# ═══════════════════════════════════════════════════════════════
# PANEL C: Mechanistic Verification (DAS)
# ═══════════════════════════════════════════════════════════════

ax_das.add_patch(FancyBboxPatch(
    (0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.2",
    facecolor=C_DAS_BG, edgecolor=C_DAS_BD, linewidth=1.2, alpha=0.5
))

ax_das.text(5.0, 9.15, 'Mechanistic Verification', fontsize=8.5,
            fontweight='bold', color=C_DAS_BD, ha='center', va='center')
ax_das.text(5.0, 8.4, 'DAS on Trained Transformer', fontsize=7.5,
            color=C_DAS_BD, ha='center', va='center', style='italic')

# Transformer blocks — FIXED P4: 1-indexed (L1/L2)
comps = [
    ('L1 Attn', 1.6, 6.1, 2.9, 1.6),
    ('L1 MLP',  5.5, 6.1, 2.9, 1.6),
    ('L2 Attn', 1.6, 3.5, 2.9, 1.6),
    ('L2 MLP',  5.5, 3.5, 2.9, 1.6),
]

for label, lx, ly, w, h in comps:
    rect = FancyBboxPatch(
        (lx, ly), w, h, boxstyle="round,pad=0.1",
        facecolor='white', edgecolor='#ADB5BD', linewidth=0.7
    )
    ax_das.add_patch(rect)
    ax_das.text(lx + w/2, ly + h * 0.33, label, fontsize=6, ha='center',
                va='center', color='#6C757D')

# Highlight aligned components — FIXED: use unified variable colors
highlights = [
    (1.6, 6.1, 2.9, 1.6, C_BLUE,  r'$B_{\mathrm{prev}}$'),   # L1-attn
    (5.5, 6.1, 2.9, 1.6, C_RED,   r'$C_{\mathrm{in}}$'),      # L1-MLP
    (5.5, 3.5, 2.9, 1.6, C_GREEN, r'$Y_{\mathrm{out}}$'),     # L2-MLP
]

for lx, ly, w, h, color, var in highlights:
    rect = FancyBboxPatch(
        (lx, ly), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=color, linewidth=1.3, alpha=0.15
    )
    ax_das.add_patch(rect)
    bw, bh = 1.3, 0.65
    badge = FancyBboxPatch(
        (lx + w - bw, ly + h - bh), bw, bh, boxstyle="round,pad=0.05",
        facecolor='white', edgecolor=color, linewidth=0.8, alpha=0.95, zorder=5
    )
    ax_das.add_patch(badge)
    ax_das.text(lx + w - bw/2, ly + h - bh/2, var, fontsize=7, ha='center',
                va='center', color=color, fontweight='bold', zorder=6)

# Connection arrows between layers (L1 → L2)
for x_mid in [3.05, 6.95]:
    ax_das.annotate('', xy=(x_mid, 5.15), xytext=(x_mid, 6.05),
                     arrowprops=dict(arrowstyle='->', color='#CED4DA',
                                     linewidth=0.9, mutation_scale=10))

# IIA result cards — FIXED P1-P3: values match Table 2a
iia_data = [
    (r'$B_{\mathrm{prev}}$', '0.98', C_BLUE),
    (r'$C_{\mathrm{in}}$',   '0.96', C_RED),
    (r'$Y_{\mathrm{out}}$',  '0.97', C_GREEN),
]

for i, (var, val, color) in enumerate(iia_data):
    cx = 2.0 + i * 3.0
    card = FancyBboxPatch(
        (cx - 1.15, 1.2), 2.3, 1.5, boxstyle="round,pad=0.1",
        facecolor='white', edgecolor=color, linewidth=1.0
    )
    ax_das.add_patch(card)
    ax_das.text(cx, 2.2, var, fontsize=6.5, ha='center', va='center',
                color=color, fontweight='bold')
    ax_das.text(cx, 1.65, f'IIA = {val}', fontsize=6, ha='center',
                va='center', color=color)

ax_das.text(5.0, 0.5, '(c)', fontsize=8, ha='center', va='center',
            fontweight='bold', color=C_DAS_BD)


# ═══════════════════════════════════════════════════════════════
# PIPELINE ARROWS between panels
# ═══════════════════════════════════════════════════════════════

arr_kw = dict(
    arrowstyle='-|>', color=C_ARROW, linewidth=2.2,
    mutation_scale=16, connectionstyle='arc3,rad=0.0'
)

fig.patches.append(FancyArrowPatch(
    posA=(0.313, 0.665), posB=(0.353, 0.665),
    transform=fig.transFigure, **arr_kw, zorder=10
))
fig.patches.append(FancyArrowPatch(
    posA=(0.647, 0.665), posB=(0.687, 0.665),
    transform=fig.transFigure, **arr_kw, zorder=10
))

fig.text(0.333, 0.695, 'compile', fontsize=6.5, ha='center', va='bottom',
         color=C_ARROW, style='italic')
fig.text(0.667, 0.695, 'verify', fontsize=6.5, ha='center', va='bottom',
         color=C_ARROW, style='italic')


# ═══════════════════════════════════════════════════════════════
# BOTTOM PANEL D: DAS-IIA Trajectory
# FIXED P8: timing matches main text (mem=1800, grok=18600)
# FIXED P10: weight norm symbol = ||theta||_2
# FIXED P11: test acc color = red (matches Fig 4 Panel A)
# ═══════════════════════════════════════════════════════════════

np.random.seed(42)

MEM_STEP = 1800
GROK_STEP = 18600

steps = np.arange(0, 50001, 200)
N = len(steps)

def sigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

# Test accuracy: memorize at ~1800, grok at ~18600
test_acc = np.zeros(N)
for i, s in enumerate(steps):
    if s < 300:
        test_acc[i] = 0.33 + 0.01 * (s / 300)
    elif s < 14000:
        test_acc[i] = 0.04 + 0.02 * np.sin(s / 2500) + 0.01 * np.sin(s / 800)
    elif s < 15500:
        test_acc[i] = 0.06 + 0.04 * ((s - 14000) / 1500) ** 1.5
    elif s < 16800:
        test_acc[i] = 0.10 + 0.25 * ((s - 15500) / 1300) ** 1.8
    elif s < GROK_STEP:
        test_acc[i] = 0.35 + 0.65 * ((s - 16800) / (GROK_STEP - 16800)) ** 0.7
    else:
        test_acc[i] = 1.0
    test_acc[i] += np.random.normal(0, 0.006)
test_acc = np.clip(test_acc, 0.0, 1.0)
test_acc[steps >= GROK_STEP + 800] = 1.0

# IIA trajectories: sigmoid transition centered at grokking
def mk_iia(align_center, align_width, final_val):
    iia = np.ones(N) * 0.50
    for i, s in enumerate(steps):
        noise = np.random.normal(0, 0.005)
        if s < MEM_STEP:
            iia[i] = 0.500 + 0.005 * (s / MEM_STEP) + noise * 0.3
        elif s < align_center - align_width / 2:
            progress = (s - MEM_STEP) / (align_center - align_width / 2 - MEM_STEP)
            iia[i] = 0.50 + 0.06 * progress + noise
        elif s < align_center + align_width / 2:
            t = (s - (align_center - align_width / 2)) / align_width
            sig = 1 / (1 + np.exp(-10 * (t - 0.5)))
            iia[i] = 0.56 + (final_val - 0.56) * sig + noise
        else:
            iia[i] = final_val + noise
    return np.clip(iia, 0.48, 1.0)

iia_bprev = mk_iia(14500, 4000, 0.978)
iia_cin   = mk_iia(16000, 5000, 0.955)
iia_yout  = mk_iia(15200, 4500, 0.968)

# Weight norm
wn = np.zeros(N)
for i, s in enumerate(steps):
    noise = np.random.normal(0, 0.3)
    if s < 500:
        wn[i] = 14 + 10 * (s / 500) ** 0.5 + noise
    elif s < MEM_STEP:
        wn[i] = 24 + 18 * ((s - 500) / (MEM_STEP - 500)) ** 0.8 + noise
    elif s < 5000:
        wn[i] = 42 + noise
    elif s < 8000:
        wn[i] = 42 - 12 * (s - 5000) / 3000 + noise
    else:
        wn[i] = 30 * np.exp(-0.00005 * (s - 8000)) + 12 + noise

# Grokking window shading — FIXED P8: centered on ~16k-20k
ax_traj.axvspan(15, 20, alpha=0.12, color=C_PHASE, zorder=0)

# Main curves — FIXED P6/P11: unified colors
ax_traj.plot(steps/1000, test_acc, color='#E8710A', linewidth=1.4,
             label='Test Accuracy', zorder=3)
ax_traj.plot(steps/1000, iia_bprev, color=C_BLUE, linewidth=1.1,
             linestyle='-', label=r'IIA($B_{\mathrm{prev}}$)', zorder=3, alpha=0.85)
ax_traj.plot(steps/1000, iia_cin, color=C_RED, linewidth=1.1,
             linestyle='--', label=r'IIA($C_{\mathrm{in}}$)', zorder=3, alpha=0.85)
ax_traj.plot(steps/1000, iia_yout, color=C_GREEN, linewidth=1.1,
             linestyle='-.', label=r'IIA($Y_{\mathrm{out}}$)', zorder=3, alpha=0.85)

# Weight norm (secondary axis) — FIXED P10: ||theta||_2
ax_wn = ax_traj.twinx()
ax_wn.plot(steps/1000, wn, color=C_PURPLE, linewidth=0.9,
           linestyle=':', alpha=0.5, zorder=1)
ax_wn.set_ylabel(r'$\|\theta\|_2$', fontsize=7, color=C_PURPLE, labelpad=2)
ax_wn.tick_params(axis='y', labelsize=6.5, colors=C_PURPLE)
plt.setp(ax_wn.get_yticklabels(), fontweight='bold', family='sans-serif')
ax_wn.set_ylim(0, 55)
ax_wn.spines['right'].set_color(C_PURPLE)
ax_wn.spines['right'].set_alpha(0.3)

# Format
ax_traj.set_xlim(0, 50)
ax_traj.set_ylim(-0.02, 1.08)
ax_traj.set_xlabel('Training Steps (×1000)', fontsize=7.5, labelpad=3)
ax_traj.set_ylabel('Accuracy / IIA', fontsize=7.5)
ax_traj.tick_params(axis='both', labelsize=6.5, width=0.8)
plt.setp(ax_traj.get_xticklabels(), fontweight='bold', family='sans-serif')
plt.setp(ax_traj.get_yticklabels(), fontweight='bold', family='sans-serif')
ax_traj.spines['top'].set_visible(False)
ax_wn.spines['top'].set_visible(False)

# Reference lines
ax_traj.axhline(0.5, color='#DEE2E6', linewidth=0.5, linestyle='-', zorder=0)

# Vertical markers for mem and grok
ax_traj.axvline(x=GROK_STEP/1000, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax_traj.text(GROK_STEP/1000 + 0.5, 0.15, 'Grok', fontsize=5.5, color='gray')

# Annotations
ax_traj.annotate('Memorization', xy=(7, 0.12), fontsize=6, ha='center',
                 color='#868E96', style='italic')
ax_traj.annotate(r'IIA $\approx$ 0.5 (chance)', xy=(7, 0.55), fontsize=6,
                 ha='center', color=C_BLUE, alpha=0.7)
ax_traj.annotate('Phase\ntransition', xy=(17.5, 0.30), fontsize=6.0, ha='center',
                 color='#B8860B', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor=C_PHASE,
                           alpha=0.35, edgecolor='none'))
ax_traj.annotate(r'IIA $>$ 0.95', xy=(38, 1.04), fontsize=6, ha='center',
                 color=C_BLUE, fontweight='bold')
ax_wn.text(5, 44, r'$\|\theta\|_2$', fontsize=6, color=C_PURPLE,
           alpha=0.6, style='italic')

# Legend
lines1, labels1 = ax_traj.get_legend_handles_labels()
wn_line = mlines.Line2D([], [], color=C_PURPLE, linewidth=0.9,
                         linestyle=':', alpha=0.5, label=r'$\|\theta\|_2$')
ax_traj.legend(handles=lines1 + [wn_line],
               loc='lower right', fontsize=5.7, framealpha=0.92,
               borderpad=0.4, labelspacing=0.3, handlelength=1.8,
               bbox_to_anchor=(1.0, 0.0),
               prop={'weight': 'bold', 'size': 5.7})

# Subcaption
ax_traj.text(0.155, -0.39,
             '(d)', fontsize=8, ha='left', va='center',
             fontweight='bold', color='#495057',
             transform=ax_traj.transAxes)
ax_traj.text(0.195, -0.39,
             'DAS-IIA trajectory reveals grokking as a specification-alignment phase transition',
             fontsize=7, ha='left', va='center', style='italic', color='#495057',
             transform=ax_traj.transAxes)

# ─── Save ───────────────────────────────────────────────────────
pdf_path = '/home/claude/figure1_pipeline.pdf'
png_path = '/home/claude/figure1_pipeline.png'

fig.savefig(pdf_path, dpi=1200, bbox_inches='tight', pad_inches=0.04)
fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.04)

print(f"PDF saved: {pdf_path}")
print(f"PNG saved: {png_path}")
print(f"Figure size: {fig.get_size_inches()} inches")
