#!/usr/bin/env python3
"""
Experiment 2: Proximity vs. Causal Plausibility
Does enforcing causal constraints force counterfactuals to be further
from the original molecule, or can we have both closeness AND plausibility?


Project: CS 6460 - Causally-Constrained Counterfactual Explanations

Run from project root:
    PYTHONPATH=. python counterfactual/experiment2.py
"""

import warnings
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = 'counterfactual/output/figures'
EVAL_CSV = 'counterfactual/output/evaluation_results.csv'

# Colors
C_CON = '#E07B54'   # constrained
C_UNCON = '#5B8DB8'   # unconstrained
C_VALID = '#4CAF82'   # valid / high plausibility
BG = '#F8F8FB'
GRID = '#E4E4ED'
THRESHOLD = 0.85       # causal score threshold used in your pipeline


def setup_style():
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.facecolor':    BG,
        'figure.facecolor':  'white',
        'axes.grid':         True,
        'grid.color':        GRID,
        'grid.linewidth':    0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.labelsize':    11,
        'axes.titlesize':    12,
        'axes.titleweight':  'bold',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   9,
        'figure.dpi':        150,
    })


def load_data():
    if not os.path.exists(EVAL_CSV):
        print(
            f"ERROR: {EVAL_CSV} not found.\nRun counterfactual/evaluate.py first.")
        sys.exit(1)
    df = pd.read_csv(EVAL_CSV)
    print(f"  Loaded: {len(df)} molecules from {EVAL_CSV}")
    print(f"  Columns: {list(df.columns)}\n")
    return df


def find_columns(df):
    """
    Flexibly find proximity and causal score columns regardless of exact naming.
    Returns (prox_uncon, prox_con, score_uncon, score_con) column names.
    """
    cols = df.columns.tolist()

    def find(keywords, exclude=None):
        for c in cols:
            cl = c.lower()
            if all(k in cl for k in keywords):
                if exclude is None or not any(e in cl for e in exclude):
                    return c
        return None

    prox_uncon = find(['proximity', 'uncon']) or find(
        ['proximity', 'unconstrained'])
    prox_con = find(['proximity', 'con'], exclude=['uncon']) or find(
        ['proximity', 'constrained'], exclude=['unconstrained'])
    score_uncon = find(['causal', 'uncon']) or find(['score', 'uncon'])
    score_con = find(['causal', 'con'], exclude=['uncon']) or find(
        ['score', 'con'], exclude=['uncon'])

    # Fallback: try common alternative names
    if prox_uncon is None:
        prox_uncon = find(['l1', 'uncon']) or find(['dist', 'uncon'])
    if prox_con is None:
        prox_con = find(['l1', 'con'], exclude=['uncon']) or find(
            ['dist', 'con'], exclude=['uncon'])

    missing = []
    for name, val in [('proximity_unconstrained', prox_uncon),
                      ('proximity_constrained',   prox_con),
                      ('causal_score_unconstrained', score_uncon),
                      ('causal_score_constrained',   score_con)]:
        if val is None:
            missing.append(name)

    if missing:
        print("Could not auto-detect these columns:")
        for m in missing:
            print(f"  - {m}")
        print("\nAvailable columns in your CSV:")
        for c in cols:
            print(f"  {c}")
        print("\nPlease update the column names in find_columns() to match.")
        sys.exit(1)

    print(f"  Column mapping:")
    print(f"    proximity_unconstrained  -> '{prox_uncon}'")
    print(f"    proximity_constrained    -> '{prox_con}'")
    print(f"    causal_score_uncon       -> '{score_uncon}'")
    print(f"    causal_score_con         -> '{score_con}'")
    return prox_uncon, prox_con, score_uncon, score_con


def plot_experiment2(df, prox_uncon, prox_con, score_uncon, score_con):
    setup_style()

    ok = df.dropna(subset=[prox_uncon, prox_con,
                   score_uncon, score_con]).copy()
    if len(ok) == 0:
        print("ERROR: No rows with complete data for all 4 columns.")
        sys.exit(1)

    n = len(ok)
    pu = ok[prox_uncon].values
    pc = ok[prox_con].values
    su = ok[score_uncon].values
    sc = ok[score_con].values

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        'Experiment 2: Proximity vs. Causal Plausibility',
        fontsize=14, fontweight='bold', y=1.01
    )

    # ── Panel 1: Unconstrained scatter ───────────────────────────────────────
    ax1 = axes[0]
    colors_u = [C_VALID if s >= THRESHOLD else C_UNCON for s in su]
    ax1.scatter(pu, su, c=colors_u, s=70, alpha=0.85,
                edgecolors='white', linewidths=0.5)
    ax1.axhline(THRESHOLD, color='#999', lw=1.2,
                ls='--', label=f'threshold={THRESHOLD}')
    ax1.set_xlabel('Proximity (L1 distance)')
    ax1.set_ylabel('Causal Score')
    ax1.set_title('Unconstrained (DiCE only)')
    ax1.set_ylim(-0.05, 1.1)

    n_valid_u = sum(s >= THRESHOLD for s in su)
    ax1.text(0.97, 0.05,
             f'Valid: {n_valid_u}/{n} ({100*n_valid_u/n:.0f}%)',
             transform=ax1.transAxes, ha='right', fontsize=9,
             color=C_VALID, fontweight='bold')

    # Regression line
    if len(pu) > 2:
        slope, intercept, r, p, _ = stats.linregress(pu, su)
        x_line = np.linspace(pu.min(), pu.max(), 100)
        ax1.plot(x_line, slope * x_line + intercept,
                 color=C_UNCON, lw=1.5, ls=':', alpha=0.8,
                 label=f'r={r:.2f}, p={p:.3f}')
    ax1.legend(fontsize=8)

    # ── Panel 2: Constrained scatter ─────────────────────────────────────────
    ax2 = axes[1]
    colors_c = [C_VALID if s >= THRESHOLD else C_CON for s in sc]
    ax2.scatter(pc, sc, c=colors_c, s=70, alpha=0.85,
                edgecolors='white', linewidths=0.5)
    ax2.axhline(THRESHOLD, color='#999', lw=1.2,
                ls='--', label=f'threshold={THRESHOLD}')
    ax2.set_xlabel('Proximity (L1 distance)')
    ax2.set_ylabel('Causal Score')
    ax2.set_title('Constrained (DiCE + DoWhy)')
    ax2.set_ylim(-0.05, 1.1)

    n_valid_c = sum(s >= THRESHOLD for s in sc)
    ax2.text(0.97, 0.05,
             f'Valid: {n_valid_c}/{n} ({100*n_valid_c/n:.0f}%)',
             transform=ax2.transAxes, ha='right', fontsize=9,
             color=C_VALID, fontweight='bold')

    if len(pc) > 2:
        slope, intercept, r, p, _ = stats.linregress(pc, sc)
        x_line = np.linspace(pc.min(), pc.max(), 100)
        ax2.plot(x_line, slope * x_line + intercept,
                 color=C_CON, lw=1.5, ls=':', alpha=0.8,
                 label=f'r={r:.2f}, p={p:.3f}')
    ax2.legend(fontsize=8)

    # ── Panel 3: Overlay comparison ──────────────────────────────────────────
    ax3 = axes[2]
    ax3.scatter(pu, su, c=C_UNCON, s=55, alpha=0.55, label='Unconstrained',
                edgecolors='white', linewidths=0.4)
    ax3.scatter(pc, sc, c=C_CON,   s=55, alpha=0.75, label='Constrained',
                edgecolors='white', linewidths=0.4, marker='D')

    # Draw arrows showing the shift per molecule
    for i in range(n):
        ax3.annotate('', xy=(pc[i], sc[i]), xytext=(pu[i], su[i]),
                     arrowprops=dict(arrowstyle='->', color='#AAAAAA',
                                     lw=0.7, alpha=0.5))

    ax3.axhline(THRESHOLD, color='#999', lw=1.2,
                ls='--', label=f'threshold={THRESHOLD}')
    ax3.set_xlabel('Proximity (L1 distance)')
    ax3.set_ylabel('Causal Score')
    ax3.set_title('Overlay: Shift from Unconstrained → Constrained')
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR, 'experiment2_proximity_vs_plausibility.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"\n  Figure saved: {out_path}")
    plt.close()
    return n_valid_u, n_valid_c, n


def print_stats(df, prox_uncon, prox_con, score_uncon, score_con):
    ok = df.dropna(subset=[prox_uncon, prox_con, score_uncon, score_con])
    n = len(ok)

    pu = ok[prox_uncon]
    pc = ok[prox_con]
    su = ok[score_uncon]
    sc = ok[score_con]

    print(f"\n{'='*60}")
    print("  EXPERIMENT 2 SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Molecules analysed: {n}")
    print(f"\n  Proximity (L1 distance from original):")
    print(f"    Unconstrained  mean={pu.mean():.4f}  std={pu.std():.4f}")
    print(f"    Constrained    mean={pc.mean():.4f}  std={pc.std():.4f}")
    delta_prox = pc.mean() - pu.mean()
    print(
        f"    Δ proximity    {delta_prox:+.4f}  ({'closer' if delta_prox < 0 else 'further'} with constraints)")

    print(f"\n  Causal Score (0–1, higher = more causally valid):")
    print(f"    Unconstrained  mean={su.mean():.4f}  std={su.std():.4f}")
    print(f"    Constrained    mean={sc.mean():.4f}  std={sc.std():.4f}")
    delta_score = sc.mean() - su.mean()
    print(
        f"    Δ causal score {delta_score:+.4f}  ({'improved' if delta_score > 0 else 'decreased'} with constraints)")

    n_valid_u = (su >= THRESHOLD).sum()
    n_valid_c = (sc >= THRESHOLD).sum()
    print(f"\n  % above threshold ({THRESHOLD}):")
    print(f"    Unconstrained  {n_valid_u}/{n} ({100*n_valid_u/n:.1f}%)")
    print(f"    Constrained    {n_valid_c}/{n} ({100*n_valid_c/n:.1f}%)")

    # Correlation
    if len(pu) > 2:
        r_u, p_u = stats.pearsonr(pu, su)
        r_c, p_c = stats.pearsonr(pc, sc)
        print(f"\n  Correlation (proximity vs causal score):")
        print(f"    Unconstrained  r={r_u:.3f}  p={p_u:.4f}")
        print(f"    Constrained    r={r_c:.3f}  p={p_c:.4f}")

    # Paper sentence
    print(f"\n{'='*60}")
    print("  PAPER SENTENCE (copy-paste into Results section):")
    print(f"{'='*60}")
    print(f"""
  Causally-constrained counterfactuals achieved a mean causal
  score of {sc.mean():.3f} (SD={sc.std():.3f}), compared to {su.mean():.3f}
  (SD={su.std():.3f}) for unconstrained DiCE, a gain of
  {delta_score:+.3f} points. The mean L1 proximity shifted by
  {delta_prox:+.4f}, indicating that the causal filter
  {'did not substantially increase' if abs(delta_prox) < 0.5 else 'changed'} the
  distance from the original molecule while improving
  causal validity from {100*n_valid_u/n:.1f}% to {100*n_valid_c/n:.1f}%
  above the {THRESHOLD} threshold.
""")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("  Experiment 2: Proximity vs. Causal Plausibility")
    print(f"{'='*60}\n")

    df = load_data()
    prox_uncon, prox_con, score_uncon, score_con = find_columns(df)
    plot_experiment2(df, prox_uncon, prox_con, score_uncon, score_con)
    print_stats(df, prox_uncon, prox_con, score_uncon, score_con)
