#!/usr/bin/env python3
"""
Experiment 1: Causal Constraint Violation Analysis
Graphical report showing what DiCE gets wrong and what the causal filter catches.

Project: CS 6460 - Causally-Constrained Counterfactual Explanations
"""

import warnings
from matplotlib.gridspec import GridSpec
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
CHECKS_CSV = 'counterfactual/output/causal_checks.csv'

# Colors
C_UNCON = '#5B8DB8'   # steel blue  — unconstrained
C_CON = '#E07B54'   # burnt orange — constrained
C_VALID = '#4CAF82'   # green — pass
C_VIOL = '#E05454'   # red   — violation
C_ORPHAN = '#F0A500'   # amber — orphan change
BG = '#F8F8FB'
GRID = '#E4E4ED'


def setup_style():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'axes.facecolor':   BG,
        'figure.facecolor': 'white',
        'axes.grid':        True,
        'grid.color':       GRID,
        'grid.linewidth':   0.8,
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'axes.labelsize':   11,
        'axes.titlesize':   12,
        'axes.titleweight': 'bold',
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'legend.fontsize':  9,
        'figure.dpi':       150,
    })


def load_data():
    eval_df = pd.read_csv(EVAL_CSV)
    checks_df = pd.read_csv(CHECKS_CSV) if os.path.exists(CHECKS_CSV) else None
    return eval_df, checks_df


def compute_violation_summary(eval_df):
    """Derive violation counts from evaluation results."""
    ok = eval_df.dropna(subset=['causal_rate_unconstrained']).copy()

    total_cfs = ok['n_cfs_generated'].sum()
    total_valid = ok['n_cfs_causal_valid'].sum()
    total_violations = total_cfs - total_valid

    # Per-molecule violation count
    ok['n_violations'] = ok['n_cfs_generated'] - ok['n_cfs_causal_valid']

    # Classify violation severity by causal score
    # score < 0.85 = failed filter; score between 0.85-1.0 = passed
    ok['pct_valid'] = ok['causal_rate_unconstrained'] * 100

    return ok, total_cfs, total_valid, total_violations


def plot_experiment1(eval_df, checks_df):
    """Full graphical report for Experiment 1."""
    setup_style()

    ok, total_cfs, total_valid, total_violations = compute_violation_summary(
        eval_df)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Experiment 1: Causal Constraint Violation Analysis\n'
        'Unconstrained DiCE vs. DoWhy-Filtered Counterfactuals',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel 1: Big summary donut ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)

    sizes = [total_valid, total_violations]
    colors = [C_VALID, C_VIOL]
    labels = [f'Causal-valid\n({total_valid})',
              f'Violations caught\n({total_violations})']

    wedges, texts = ax1.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
    )
    pct = total_valid / total_cfs * 100
    ax1.text(0, 0, f'{pct:.1f}%\npass rate',
             ha='center', va='center', fontsize=13, fontweight='bold', color='#333')
    ax1.set_title(f'Total CFs: {int(total_cfs)}', pad=8)
    ax1.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18),
               fontsize=8, framealpha=0)

    # ── Panel 2: Per-molecule violations bar ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor(BG)

    ok_sorted = ok.sort_values(
        'n_violations', ascending=False).reset_index(drop=True)
    smiles_labels = [s[:14] + '…' if len(s) > 14 else s
                     for s in ok_sorted['smiles']]

    x = np.arange(len(ok_sorted))
    bar_colors = [C_VIOL if v >
                  0 else C_VALID for v in ok_sorted['n_violations']]

    bars = ax2.bar(x, ok_sorted['n_violations'], color=bar_colors,
                   alpha=0.85, zorder=3, edgecolor='white')

    # Annotate bars
    for bar, val in zip(bars, ok_sorted['n_violations']):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     str(int(val)), ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color=C_VIOL)

    ax2.set_xticks(x)
    ax2.set_xticklabels(smiles_labels, rotation=40, ha='right', fontsize=7.5)
    ax2.set_ylabel('CFs with causal violations')
    ax2.set_title('Violations per Query Molecule (caught by DoWhy filter)')
    ax2.set_ylim(0, ok_sorted['n_violations'].max() + 1)

    valid_patch = mpatches.Patch(
        color=C_VALID, label='0 violations (all CFs valid)')
    viol_patch = mpatches.Patch(
        color=C_VIOL,  label='≥1 violation (filter triggered)')
    ax2.legend(handles=[valid_patch, viol_patch], fontsize=8)

    # ── Panel 3: Causal score distribution comparison ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)

    scores_unc = ok['causal_score_unconstrained'].dropna()
    scores_con = ok['causal_score_constrained'].dropna()

    bins = np.linspace(0.7, 1.01, 16)
    ax3.hist(scores_unc, bins=bins, color=C_UNCON, alpha=0.7,
             label='Unconstrained', zorder=3, edgecolor='white')
    ax3.hist(scores_con, bins=bins, color=C_CON, alpha=0.7,
             label='Constrained', zorder=3, edgecolor='white')
    ax3.axvline(0.85, color='#333', lw=1.5, ls='--', alpha=0.7,
                label='Filter threshold (0.85)')
    ax3.set_xlabel('Causal Score')
    ax3.set_ylabel('Count (molecules)')
    ax3.set_title('Causal Score Distribution')
    ax3.legend(fontsize=8)

    # ── Panel 4: Before vs after filter — stacked bar per molecule ───────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)

    # Aggregate: molecules with 0, 1, 2, 3, 4, 5 valid CFs after filtering
    counts = ok['n_cfs_causal_valid'].value_counts().sort_index()
    all_vals = list(range(0, int(ok['n_cfs_generated'].max()) + 1))
    freq = [counts.get(v, 0) for v in all_vals]
    bar_c = [C_VIOL if v == 0 else (C_VALID if v == 5 else C_UNCON)
             for v in all_vals]

    ax4.bar(all_vals, freq, color=bar_c, alpha=0.85,
            edgecolor='white', zorder=3)
    ax4.set_xlabel('# CFs passing causal filter')
    ax4.set_ylabel('# Molecules')
    ax4.set_title('How Many CFs Survived per Molecule?')
    ax4.set_xticks(all_vals)

    for i, (v, f) in enumerate(zip(all_vals, freq)):
        if f > 0:
            ax4.text(v, f + 0.1, str(f), ha='center', va='bottom', fontsize=9)

    none_patch = mpatches.Patch(
        color=C_VIOL,  label='0 valid (molecule failed)')
    all_patch = mpatches.Patch(color=C_VALID, label='5 valid (all passed)')
    some_patch = mpatches.Patch(color=C_UNCON, label='1–4 valid (partial)')
    ax4.legend(handles=[all_patch, some_patch, none_patch], fontsize=8)

    # ── Panel 5: Key metrics table ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG)
    ax5.axis('off')

    metrics = [
        ('Total CFs generated',          f'{int(total_cfs)}'),
        ('Total violations caught',       f'{int(total_violations)}'),
        ('CFs passing causal filter',
         f'{int(total_valid)} ({total_valid/total_cfs*100:.1f}%)'),
        ('Molecules with 0 violations',
         f'{int((ok["n_violations"] == 0).sum())} / {len(ok)}'),
        ('Avg causal score (uncon.)',
         f'{ok["causal_score_unconstrained"].mean():.4f}'),
        ('Avg causal score (con.)',
         f'{ok["causal_score_constrained"].mean():.4f}'),
        ('Avg CFs valid per molecule',
         f'{ok["n_cfs_causal_valid"].mean():.1f} / {ok["n_cfs_generated"].mean():.0f}'),
    ]

    ax5.set_title('Experiment 1 — Summary Statistics', pad=8)
    y = 0.92
    for label, value in metrics:
        ax5.text(0.02, y, label, transform=ax5.transAxes,
                 fontsize=9, color='#555', va='top')
        ax5.text(0.98, y, value, transform=ax5.transAxes,
                 fontsize=9, fontweight='bold', color='#1F4E79',
                 ha='right', va='top')
        y -= 0.12
        ax5.plot([0.01, 0.99], [y + 0.04, y + 0.04],
                 color=GRID, lw=0.8, transform=ax5.transAxes)

    # Highlight the headline result
    ax5.text(0.5, 0.06,
             f'✓  {total_valid/total_cfs*100:.1f}% causal pass rate',
             transform=ax5.transAxes, fontsize=11, fontweight='bold',
             color=C_VALID, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#D4EDD4',
                       edgecolor=C_VALID, linewidth=1.5))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f'{OUTPUT_DIR}/experiment1_violations.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  ✓ Saved: {path}')
    return path


def print_text_summary(eval_df):
    """Print the paper-ready numbers to terminal."""
    ok, total_cfs, total_valid, total_violations = compute_violation_summary(
        eval_df)

    print('\n' + '='*60)
    print('  EXPERIMENT 1 — PAPER-READY NUMBERS')
    print('='*60)
    print(f'  Molecules evaluated:          {len(ok)}')
    print(
        f'  CFs generated (total):        {int(total_cfs)}  ({int(ok["n_cfs_generated"].mean())} per molecule)')
    print(f'  CFs with causal violations:   {int(total_violations)}')
    print(f'  CFs passing causal filter:    {int(total_valid)}')
    print(f'  Causal pass rate:             {total_valid/total_cfs*100:.1f}%')
    print(
        f'  Molecules with all 5 valid:   {int((ok["n_cfs_causal_valid"] == 5).sum())}')
    print(
        f'  Molecules with 0 valid:       {int((ok["n_cfs_causal_valid"] == 0).sum())}')
    print(
        f'  Avg causal score (uncon.):    {ok["causal_score_unconstrained"].mean():.4f}')
    print(
        f'  Avg causal score (con.):      {ok["causal_score_constrained"].mean():.4f}')
    print(
        f'  Score improvement:            +{(ok["causal_score_constrained"]-ok["causal_score_unconstrained"]).mean():.4f}')
    print('='*60)
    print('\n  Paper sentence:')
    print(
        f'  "Of {int(total_cfs)} counterfactuals generated across {len(ok)} molecules,')
    print(f'   {int(total_violations)} ({total_violations/total_cfs*100:.1f}%) were flagged as causally')
    print(f'   inconsistent and removed by the DoWhy filter, yielding a')
    print(
        f'   {total_valid/total_cfs*100:.1f}% causal pass rate. The average causal score')
    print(
        f'   improved from {ok["causal_score_unconstrained"].mean():.4f} to {ok["causal_score_constrained"].mean():.4f}')
    print(f'   after filtering."')
    print('='*60)


if __name__ == '__main__':
    print('='*60)
    print('  Experiment 1: Violation Analysis')
    print('='*60)

    if not os.path.exists(EVAL_CSV):
        print(f'ERROR: {EVAL_CSV} not found.')
        print('Run counterfactual/evaluate.py first.')
        sys.exit(1)

    eval_df, checks_df = load_data()
    print(f'  Loaded: {len(eval_df)} molecules\n')

    plot_experiment1(eval_df, checks_df)
    print_text_summary(eval_df)
    print('\n✓ Experiment 1 complete.')
