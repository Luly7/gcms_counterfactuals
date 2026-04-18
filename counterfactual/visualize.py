#!/usr/bin/env python3
"""
Visualization Report for Causally-Constrained Counterfactual Explanations
Step 4: Generate paper-ready figures

Figures produced:
  1. causal_graph.png         — DoWhy causal graph of molecular descriptors
  2. validity_scatter.png     — Per-molecule validity vs causal score
  3. feature_heatmap.png      — Which features DiCE changes most often
  4. metrics_comparison.png   — Unconstrained vs constrained bar chart
  5. rt_shift_distribution.png — Distribution of RT shifts across CFs

Project: CS 6460 - Causally-Constrained Counterfactual Explanations
"""

import warnings
from matplotlib.patches import FancyArrowPatch
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE, 'counterfactual/output/figures')
EVAL_CSV = os.path.join(BASE, 'counterfactual/output/evaluation_results.csv')
CF_FULL = os.path.join(BASE, 'counterfactual/output/cf_full.csv')

# Consistent color palette
C_UNCONSTRAINED = '#5B8DB8'   # steel blue
C_CONSTRAINED = '#E07B54'   # burnt orange
C_VALID = '#4CAF82'   # green
C_INVALID = '#E05454'   # red
C_NEUTRAL = '#8B8B9E'   # grey
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
        'axes.labelsize':    11,
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   9,
        'figure.dpi':        150,
    })


# ============================================================================
# FIGURE 1: Causal Graph
# ============================================================================

def plot_causal_graph(out_dir: str):
    """Draw the DoWhy causal graph of molecular descriptors → RT."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Node positions
    nodes = {
        # Root causes (left)
        'molecular_weight':   (1.2, 4.5),
        'num_aromatic_rings': (1.2, 2.5),
        'num_heteroatoms':    (1.2, 0.5),
        'fraction_csp3':      (1.2, 3.5),
        # Mediators (middle)
        'logp':               (3.8, 5.0),
        'tpsa':               (3.8, 3.5),
        'labuteasa':          (3.8, 2.2),
        'molmr':              (3.8, 4.2),
        'chi0v':              (3.8, 1.5),
        'kappa1':             (3.8, 0.8),
        # VSA children (right-middle)
        'peoe_vsa':           (6.5, 5.2),
        'slogp_vsa':          (6.5, 4.4),
        'smr_vsa':            (6.5, 3.6),
        'h_bond':             (6.5, 2.8),
        'kappa2':             (6.5, 1.8),
        'balabanj':           (6.5, 1.0),
        # Outcome
        'retention_time':     (9.0, 3.0),
    }

    # Node styles
    root_style = dict(boxstyle='round,pad=0.4', facecolor='#C8DDF5',
                      edgecolor='#3A6EA5', linewidth=1.5)
    mediator_style = dict(boxstyle='round,pad=0.4', facecolor='#FFE4C8',
                          edgecolor='#C87840', linewidth=1.5)
    child_style = dict(boxstyle='round,pad=0.4', facecolor='#D4EDD4',
                       edgecolor='#3A8A3A', linewidth=1.5)
    outcome_style = dict(boxstyle='round,pad=0.5', facecolor='#E8C8E8',
                         edgecolor='#803A80', linewidth=2.0)

    style_map = {
        'molecular_weight':   root_style,
        'num_aromatic_rings': root_style,
        'num_heteroatoms':    root_style,
        'fraction_csp3':      root_style,
        'logp':               mediator_style,
        'tpsa':               mediator_style,
        'labuteasa':          mediator_style,
        'molmr':              mediator_style,
        'chi0v':              mediator_style,
        'kappa1':             mediator_style,
        'peoe_vsa':           child_style,
        'slogp_vsa':          child_style,
        'smr_vsa':            child_style,
        'h_bond':             child_style,
        'kappa2':             child_style,
        'balabanj':           child_style,
        'retention_time':     outcome_style,
    }

    label_map = {
        'molecular_weight':   'Molecular\nWeight',
        'num_aromatic_rings': 'Aromatic\nRings',
        'num_heteroatoms':    'Heteroatoms',
        'fraction_csp3':      'Fraction\nCSP3',
        'logp':               'LogP',
        'tpsa':               'TPSA',
        'labuteasa':          'Labute\nASA',
        'molmr':              'MolMR',
        'chi0v':              'Chi0v',
        'kappa1':             'Kappa1',
        'peoe_vsa':           'PEOE_VSA\n1-5',
        'slogp_vsa':          'SLogP_VSA\n1-2',
        'smr_vsa':            'SMR_VSA\n1-2',
        'h_bond':             'H-Acceptors\nH-Donors',
        'kappa2':             'Kappa2',
        'balabanj':           'BalabanJ',
        'retention_time':     'Retention\nTime',
    }

    # Draw edges first
    edges = [
        ('molecular_weight', 'logp'),
        ('molecular_weight', 'tpsa'),
        ('molecular_weight', 'labuteasa'),
        ('molecular_weight', 'molmr'),
        ('logp', 'peoe_vsa'),
        ('logp', 'slogp_vsa'),
        ('molecular_weight', 'smr_vsa'),
        ('tpsa', 'h_bond'),
        ('num_heteroatoms', 'tpsa'),
        ('num_aromatic_rings', 'chi0v'),
        ('num_aromatic_rings', 'balabanj'),
        ('num_aromatic_rings', 'kappa2'),
        ('fraction_csp3', 'kappa1'),
        ('fraction_csp3', 'kappa2'),
        # All → RT
        ('logp', 'retention_time'),
        ('tpsa', 'retention_time'),
        ('labuteasa', 'retention_time'),
        ('molmr', 'retention_time'),
        ('peoe_vsa', 'retention_time'),
        ('slogp_vsa', 'retention_time'),
        ('smr_vsa', 'retention_time'),
        ('h_bond', 'retention_time'),
        ('chi0v', 'retention_time'),
        ('kappa1', 'retention_time'),
        ('kappa2', 'retention_time'),
        ('balabanj', 'retention_time'),
    ]

    arrowprops = dict(arrowstyle='-|>', color='#555566',
                      lw=1.5, mutation_scale=18,
                      shrinkA=10, shrinkB=20, connectionstyle='arc3,rad=0.08'
                      )
    for src, dst in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrowprops)

    # Draw nodes
    for name, (x, y) in nodes.items():
        ax.text(x, y, label_map[name], ha='center', va='center',
                fontsize=8, bbox=style_map[name])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#C8DDF5', edgecolor='#3A6EA5',
                       label='Root causes'),
        mpatches.Patch(facecolor='#FFE4C8',
                       edgecolor='#C87840', label='Mediators'),
        mpatches.Patch(facecolor='#D4EDD4', edgecolor='#3A8A3A',
                       label='Derived features'),
        mpatches.Patch(facecolor='#E8C8E8',
                       edgecolor='#803A80', label='Outcome'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              bbox_to_anchor=(0.0, 0.1))

    ax.set_title('Causal Graph: Molecular Descriptor Dependencies → Retention Time',
                 pad=12, fontsize=13, fontweight='bold')

    path = f'{out_dir}/causal_graph.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ causal_graph.png")
    return path


# ============================================================================
# FIGURE 2: Metrics Comparison Bar Chart
# ============================================================================

def plot_metrics_comparison(results_df: pd.DataFrame, out_dir: str):
    """Side-by-side bar chart: unconstrained vs constrained metrics."""
    metrics = {
        'Validity':        ('validity_unconstrained',    'validity_constrained'),
        'Causal Score':    ('causal_score_unconstrained', 'causal_score_constrained'),
        'Causal\nValid %': ('causal_rate_unconstrained', 'causal_rate_constrained'),
    }

    ok = results_df.dropna(subset=['validity_constrained'])
    means_unc = [ok[v[0]].mean() for v in metrics.values()]
    means_con = [ok[v[1]].mean() for v in metrics.values()]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, means_unc, width, label='Unconstrained (DiCE only)',
                   color=C_UNCONSTRAINED, alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width/2, means_con, width, label='Constrained (+ DoWhy)',
                   color=C_CONSTRAINED, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Unconstrained vs Causally-Constrained Counterfactuals')
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    path = f'{out_dir}/metrics_comparison.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ metrics_comparison.png")
    return path


# ============================================================================
# FIGURE 3: Validity vs Causal Score Scatter
# ============================================================================

def plot_validity_scatter(results_df: pd.DataFrame, out_dir: str):
    """Scatter: per-molecule causal score vs validity, colored by pass/fail."""
    ok = results_df.dropna(
        subset=['causal_score_constrained', 'validity_constrained'])

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [C_VALID if v == 1.0 else C_UNCONSTRAINED
              for v in ok['validity_constrained']]

    sc = ax.scatter(ok['causal_score_constrained'],
                    ok['validity_constrained'],
                    c=colors, s=80, alpha=0.85, zorder=3,
                    edgecolors='white', linewidths=0.5)

    # Annotate with SMILES (truncated)
    for _, row in ok.iterrows():
        label = row['smiles'][:12] + ('…' if len(row['smiles']) > 12 else '')
        ax.annotate(label,
                    (row['causal_score_constrained'],
                     row['validity_constrained']),
                    fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords='offset points')

    ax.axhline(1.0, color=C_VALID, lw=1, ls='--',
               alpha=0.5, label='Perfect validity')
    ax.axvline(0.85, color=C_CONSTRAINED, lw=1, ls='--', alpha=0.5,
               label='Causal threshold (0.85)')
    ax.set_xlabel('Causal Score (constrained CFs)')
    ax.set_ylabel('Validity (fraction in target RT range)')
    ax.set_title('Per-Molecule: Causal Score vs CF Validity')
    ax.legend(fontsize=8)

    valid_patch = mpatches.Patch(color=C_VALID,         label='Validity = 1.0')
    partial_patch = mpatches.Patch(
        color=C_UNCONSTRAINED, label='Validity < 1.0')
    ax.legend(handles=[valid_patch, partial_patch], fontsize=8)

    path = f'{out_dir}/validity_scatter.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ validity_scatter.png")
    return path


# ============================================================================
# FIGURE 4: Feature Change Frequency Heatmap
# ============================================================================

def plot_feature_heatmap(out_dir: str):
    """
    Which features does DiCE change most often?
    Uses cf_full.csv from the benzene demo run.
    Computes frequency and average delta magnitude across all CFs.
    """
    # Load the full CF output from benzene demo
    if not os.path.exists(CF_FULL):
        print(f"  ⚠  {CF_FULL} not found, skipping heatmap.")
        return None

    from counterfactual.causal_constraints import CAUSAL_EDGES

    cf_df = pd.read_csv(CF_FULL)

    from feature_extraction.feature_pipeline import CompleteFeatureExtractor
    extractor = CompleteFeatureExtractor()
    feature_cols = extractor.get_feature_columns()
    mol_features = [f for f in feature_cols if f in cf_df.columns]

    # We need the original — use first row as a proxy baseline
    # (In a real run we'd load it; here we use column means from training)
    # Load eval results to get original RT reference
    benzene = None  # not needed for heatmap

    # Compute frequency each feature appears changed (non-zero delta)
    # Since we don't have the original here, use variance across CFs as proxy
    variances = cf_df[mol_features].var().sort_values(ascending=False)
    top_features = variances[variances > 0].head(15).index.tolist()

    if len(top_features) == 0:
        print("  ⚠  No feature variance found in cf_full.csv, skipping heatmap.")
        return None

    # Build matrix: rows = CFs, cols = top features
    matrix = cf_df[top_features].values

    # Normalise each column for display
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    matrix_norm = (matrix - col_min) / col_range

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix_norm.T, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xticks(range(len(cf_df)))
    ax.set_xticklabels([f'CF {i+1}' for i in range(len(cf_df))], fontsize=8)
    ax.set_title('Feature Values Across Counterfactuals (normalised)\n'
                 'Green = high value, Red = low value', fontsize=11)

    plt.colorbar(im, ax=ax, label='Normalised feature value', shrink=0.8)

    path = f'{out_dir}/feature_heatmap.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ feature_heatmap.png")
    return path


# ============================================================================
# FIGURE 5: RT Shift Distribution
# ============================================================================

def plot_rt_distribution(results_df: pd.DataFrame, out_dir: str):
    """Distribution of original RTs and how far CFs shift them."""
    ok = results_df.dropna(subset=['original_rt', 'target_min'])
    if len(ok) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: distribution of original RTs
    axes[0].hist(ok['original_rt'], bins=12, color=C_UNCONSTRAINED,
                 alpha=0.8, edgecolor='white', zorder=3)
    axes[0].set_xlabel('Predicted Retention Time (min)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Query Molecule RTs')

    # Right: causal pass rate by RT bin
    ok = ok.copy()
    ok['rt_bin'] = pd.cut(ok['original_rt'], bins=5)
    bin_stats = ok.groupby('rt_bin', observed=True).agg(
        causal_rate=('causal_rate_constrained', 'mean'),
        count=('original_rt', 'count')
    ).reset_index()

    colors = [C_VALID if r >= 0.9 else C_UNCONSTRAINED
              for r in bin_stats['causal_rate']]
    bars = axes[1].bar(range(len(bin_stats)),
                       bin_stats['causal_rate'],
                       color=colors, alpha=0.85, edgecolor='white', zorder=3)
    axes[1].set_xticks(range(len(bin_stats)))
    axes[1].set_xticklabels(
        [str(b) for b in bin_stats['rt_bin']], rotation=25, ha='right', fontsize=7)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel('Causal Valid Rate')
    axes[1].set_title('Causal Valid Rate by RT Range')
    axes[1].axhline(0.9, color=C_CONSTRAINED, ls='--', lw=1, alpha=0.6,
                    label='90% threshold')
    axes[1].legend(fontsize=8)

    for bar, (_, row) in zip(bars, bin_stats.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.03,
                     f'n={int(row["count"])}',
                     ha='center', va='bottom', fontsize=7)

    fig.suptitle('Retention Time Coverage and Causal Quality',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = f'{out_dir}/rt_distribution.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ rt_distribution.png")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  GC-MS Counterfactual Visualization — Step 4")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    # Load evaluation results
    if not os.path.exists(EVAL_CSV):
        print(f"ERROR: {EVAL_CSV} not found.")
        print("Run counterfactual/evaluate.py first.")
        sys.exit(1)

    results_df = pd.read_csv(EVAL_CSV)
    print(f"Loaded evaluation results: {len(results_df)} molecules\n")
    print("Generating figures...")

    paths = []
    paths.append(plot_causal_graph(OUTPUT_DIR))
    paths.append(plot_metrics_comparison(results_df, OUTPUT_DIR))
    paths.append(plot_validity_scatter(results_df, OUTPUT_DIR))
    paths.append(plot_feature_heatmap(OUTPUT_DIR))
    paths.append(plot_rt_distribution(results_df, OUTPUT_DIR))

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("\nFigure summary for paper:")
    print("  Fig 1: causal_graph.png        — causal structure of descriptors")
    print("  Fig 2: metrics_comparison.png  — key metric deltas (main result)")
    print("  Fig 3: validity_scatter.png    — per-molecule causal score vs validity")
    print("  Fig 4: feature_heatmap.png     — which features DiCE changes most")
    print("  Fig 5: rt_distribution.png     — RT coverage and causal quality")
    print("\n✓ Step 4 complete. Full pipeline done.")


if __name__ == '__main__':
    main()
