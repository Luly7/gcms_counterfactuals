#!/usr/bin/env python3
"""
Experiment 3: NIST Case Studies
Pick 3 real molecules from the eval CSV, produce a formatted
report for manual chemist evaluation.

Project: CS 6460 - Causally-Constrained Counterfactual Explanations

Run from project root:
    PYTHONPATH=. python counterfactual/experiment3.py
"""

import warnings
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

C_ORIG = '#5B8DB8'
C_TRUE = '#A78BFA'
C_UNCON = '#E07B54'
C_CON = '#4CAF82'
BG = '#F8F8FB'
GRID = '#E4E4ED'


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
        'axes.labelsize':    10,
        'axes.titlesize':    11,
        'axes.titleweight':  'bold',
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,
        'figure.dpi':        150,
    })


def load_data():
    if not os.path.exists(EVAL_CSV):
        print(
            f"ERROR: {EVAL_CSV} not found.\nRun counterfactual/evaluate.py first.")
        sys.exit(1)
    df = pd.read_csv(EVAL_CSV)
    print(f"  Loaded: {len(df)} molecules")
    return df


def pick_case_studies(df):
    """Pick best, middle, and worst by causal_score_constrained."""
    score_col = 'causal_score_constrained'
    if score_col not in df.columns:
        idxs = [0, len(df)//2, len(df)-1]
        labels = ['Molecule A (first)',
                  'Molecule B (middle)', 'Molecule C (last)']
        return [df.iloc[i] for i in idxs], labels

    valid = df.dropna(subset=[score_col]).sort_values(
        score_col).reset_index(drop=True)

    row_best = valid.iloc[-1]
    row_worst = valid.iloc[0]
    row_mid = valid.iloc[len(valid)//2]

    def short_label(row):
        smi = str(row.get('smiles', ''))
        return smi[:22] + ('…' if len(smi) > 22 else '') if smi else f"mol {int(row.get('mol_index', 0))}"

    rows = [row_best, row_mid, row_worst]
    labels = [
        f"Best CF  — {short_label(row_best)}",
        f"Mid CF   — {short_label(row_mid)}",
        f"Worst CF — {short_label(row_worst)}",
    ]
    print("\n  Selected case studies:")
    for r, l in zip(rows, labels):
        print(f"    {l}  |  causal_score={r[score_col]:.3f}")
    return rows, labels


def plot_panel(axes_row, row, case_label, idx):
    ax_rt, ax_met = axes_row

    # ── RT bar chart ──────────────────────────────────────────────────────────
    bars = []
    if pd.notna(row.get('original_rt')):
        bars.append(('Original\nRT',  row['original_rt'], C_ORIG))
    if pd.notna(row.get('true_rt')):
        bars.append(('True\nRT',      row['true_rt'],     C_TRUE))
    t_min = row.get('target_min')
    t_max = row.get('target_max')
    if pd.notna(t_min) and pd.notna(t_max):
        bars.append(('Target\nmidpoint', (t_min + t_max) / 2.0, C_UNCON))

    if bars:
        xlbls = [b[0] for b in bars]
        yvals = [b[1] for b in bars]
        clrs = [b[2] for b in bars]
        rects = ax_rt.bar(xlbls, yvals, color=clrs, edgecolor='white',
                          linewidth=0.8, alpha=0.88, width=0.5)
        for rect, v in zip(rects, yvals):
            ax_rt.text(rect.get_x() + rect.get_width()/2,
                       v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        if pd.notna(t_min) and pd.notna(t_max):
            ax_rt.axhspan(t_min, t_max, alpha=0.12,
                          color=C_CON, label='Target range')
            ax_rt.legend(fontsize=7)
        ax_rt.set_ylabel('Retention Time (min)')
    else:
        ax_rt.text(0.5, 0.5, 'RT data not available',
                   transform=ax_rt.transAxes, ha='center', va='center', color='gray')
    ax_rt.set_title(f'Case {idx+1}: RT Values\n{case_label}', fontsize=9)

    # ── Metrics grouped bar chart ─────────────────────────────────────────────
    metric_pairs = [
        ('Validity',      'validity_unconstrained',      'validity_constrained'),
        ('Sparsity',      'sparsity_unconstrained',      'sparsity_constrained'),
        ('Proximity',     'proximity_unconstrained',     'proximity_constrained'),
        ('Causal\nScore', 'causal_score_unconstrained',  'causal_score_constrained'),
        ('Causal\nRate',  'causal_rate_unconstrained',   'causal_rate_constrained'),
        ('Diversity',     'diversity_unconstrained',     'diversity_constrained'),
    ]
    present = [(n, u, c) for n, u, c in metric_pairs
               if u in row.index and c in row.index
               and pd.notna(row.get(u)) and pd.notna(row.get(c))]

    if present:
        x = np.arange(len(present))
        w = 0.35
        ax_met.bar(x - w/2, [row[p[1]] for p in present], w,
                   label='Unconstrained', color=C_UNCON, alpha=0.82, edgecolor='white')
        ax_met.bar(x + w/2, [row[p[2]] for p in present], w,
                   label='Constrained',   color=C_CON,   alpha=0.82, edgecolor='white')
        ax_met.set_xticks(x)
        ax_met.set_xticklabels([p[0] for p in present], fontsize=7)
        ax_met.set_ylabel('Value')
        ax_met.set_title('Unconstrained vs. Constrained Metrics', fontsize=9)
        ax_met.legend(fontsize=7)
    else:
        ax_met.text(0.5, 0.5, 'Metric columns not detected',
                    transform=ax_met.transAxes, ha='center', va='center', color='gray')
        ax_met.set_title('Metrics', fontsize=9)


def make_figure(rows, labels):
    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        'Experiment 3: NIST Case Studies\n'
        'Best, Mid, and Worst Causally-Constrained Counterfactuals',
        fontsize=13, fontweight='bold', y=1.01
    )
    for i, (row, label) in enumerate(zip(rows, labels)):
        plot_panel(axes[i], row, label, i)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'experiment3_case_studies.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f"\n  Figure saved: {out}")
    plt.close()


def print_narratives(rows, labels):
    print(f"\n{'='*65}")
    print("  EXPERIMENT 3 PAPER NARRATIVES")
    print(f"{'='*65}")

    for i, (row, label) in enumerate(zip(rows, labels)):
        smi = row.get('smiles', 'N/A')
        orig_rt = row.get('original_rt')
        true_rt = row.get('true_rt')
        t_min = row.get('target_min')
        t_max = row.get('target_max')
        n_gen = row.get('n_cfs_generated')
        n_valid = row.get('n_cfs_causal_valid')
        score_con = row.get('causal_score_constrained')
        score_unc = row.get('causal_score_unconstrained')
        prox_con = row.get('proximity_constrained')
        prox_unc = row.get('proximity_unconstrained')

        rt_ref = orig_rt if pd.notna(orig_rt) else true_rt
        sc_str = f"{score_con:.3f}" if pd.notna(score_con) else "N/A"
        su_str = f"{score_unc:.3f}" if pd.notna(score_unc) else "N/A"
        tgt_str = f"{t_min:.1f}–{t_max:.1f} min" if pd.notna(
            t_min) else "the target range"
        nv_str = f"{int(n_valid)}/{int(n_gen)}" if (pd.notna(n_valid)
                                                    and pd.notna(n_gen)) else "several"

        print(f"\n  ── Case {i+1}: {label} ──")
        print(f"    SMILES         : {smi}")
        if pd.notna(orig_rt):
            print(f"    Original RT    : {orig_rt:.3f} min")
        if pd.notna(true_rt):
            print(f"    True RT        : {true_rt:.3f} min")
        if pd.notna(t_min):
            print(f"    Target range   : {t_min:.2f}–{t_max:.2f} min")
        if pd.notna(n_gen):
            print(f"    CFs generated  : {int(n_gen)}")
        if pd.notna(n_valid):
            print(f"    CFs valid      : {int(n_valid)}")
        if pd.notna(score_con):
            print(f"    Causal score ✓ : {score_con:.3f}")
        if pd.notna(score_unc):
            print(f"    Causal score ✗ : {score_unc:.3f}")
        if pd.notna(prox_con):
            print(f"    Proximity ✓    : {prox_con:.4f}")
        if pd.notna(prox_unc):
            print(f"    Proximity ✗    : {prox_unc:.4f}")

        if rt_ref is not None and not pd.isna(rt_ref):
            print(f"""
    Draft narrative:
    The molecule (SMILES: {str(smi)[:35]}) eluted at
    {rt_ref:.2f} min. The system generated {nv_str} causally-valid
    counterfactuals targeting {tgt_str}. The constrained CF
    achieved a causal score of {sc_str} vs. {su_str} unconstrained.
    [ADD CHEMIST EVALUATION: do the suggested molecular changes
    make physical sense for shifting RT in this direction?]
""")

    print(f"{'='*65}")
    print("  Fill in [ADD CHEMIST EVALUATION] for your Results section.")
    print(f"{'='*65}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'='*65}")
    print("  Experiment 3: NIST Case Studies")
    print(f"{'='*65}\n")

    df = load_data()
    rows, labels = pick_case_studies(df)
    make_figure(rows, labels)
    print_narratives(rows, labels)
