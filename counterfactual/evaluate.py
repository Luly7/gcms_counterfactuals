#!/usr/bin/env python3
"""
Evaluation Metrics for Causally-Constrained Counterfactual Explanations
Step 3: Quantitative evaluation across multiple query molecules

Metrics computed:
  - Validity:            % of CFs whose predicted RT falls in target range
  - Sparsity:            Average number of features changed per CF
  - Proximity (L1):      Average L1 distance from original features
  - Causal Plausibility: Average causal score of generated CFs
  - Causal Validity:     % of CFs passing causal filter
  - Improvement:         Causal validity of constrained vs unconstrained CFs

Project: CS 6460 - Causally-Constrained Counterfactual Explanations
"""

from counterfactual.causal_constraints import (
    CausalConstraintChecker, CausalCFFilter, run_causal_analysis
)
from counterfactual.dice_explainer import (
    load_artifacts, load_and_prepare_data,
    make_prediction, GCMSDiceExplainer
)
from feature_extraction.feature_pipeline import CompleteFeatureExtractor
import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CONFIGURATION
# ============================================================================

N_QUERY_MOLECULES = 20    # Number of molecules to evaluate
N_COUNTERFACTUALS = 5     # DiCE CFs per query
RT_SHIFT = 3.0   # Target: shift RT by this many minutes
MIN_CAUSAL_SCORE = 0.85  # Causal filter threshold
RANDOM_SEED = 42
OUTPUT_DIR = 'counterfactual/output'


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def compute_validity(cf_df: pd.DataFrame, target_range: tuple) -> float:
    """% of CFs whose predicted RT is within target range."""
    if cf_df is None or len(cf_df) == 0:
        return 0.0
    in_range = cf_df['retention_time'].between(*target_range).sum()
    return float(in_range / len(cf_df))


def compute_sparsity(original: pd.Series, cf_df: pd.DataFrame,
                     feature_cols: list, tol: float = 1e-4) -> float:
    """Average number of features changed per CF."""
    if cf_df is None or len(cf_df) == 0:
        return float('nan')
    counts = []
    for _, row in cf_df.iterrows():
        n_changed = (row[feature_cols] -
                     original[feature_cols]).abs().gt(tol).sum()
        counts.append(n_changed)
    return float(np.mean(counts))


def compute_proximity(original: pd.Series, cf_df: pd.DataFrame,
                      feature_cols: list) -> float:
    """Average L1 distance between original and CF feature vectors."""
    if cf_df is None or len(cf_df) == 0:
        return float('nan')
    dists = []
    for _, row in cf_df.iterrows():
        l1 = (row[feature_cols] - original[feature_cols]).abs().sum()
        dists.append(float(l1))
    return float(np.mean(dists))


def compute_causal_metrics(original: pd.Series, cf_df: pd.DataFrame,
                           feature_cols: list,
                           min_score: float = MIN_CAUSAL_SCORE) -> dict:
    """Causal plausibility score and % passing causal filter."""
    if cf_df is None or len(cf_df) == 0:
        return {'avg_causal_score': float('nan'), 'causal_validity_rate': 0.0}

    checker = CausalConstraintChecker()
    scores = []
    for _, row in cf_df.iterrows():
        result = checker.check_counterfactual(original, row[feature_cols])
        scores.append(result['causal_score'])

    avg_score = float(np.mean(scores))
    causal_validity_rate = float(np.mean([s >= min_score for s in scores]))
    return {
        'avg_causal_score':    round(avg_score, 4),
        'causal_validity_rate': round(causal_validity_rate, 4),
        'causal_scores':       scores,
    }


def compute_diversity(cf_df: pd.DataFrame, feature_cols: list) -> float:
    """
    Average pairwise L1 distance between CFs — higher = more diverse.
    DiCE is designed to maximise this.
    """
    if cf_df is None or len(cf_df) < 2:
        return float('nan')
    vectors = cf_df[feature_cols].values
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dists.append(np.abs(vectors[i] - vectors[j]).sum())
    return float(np.mean(dists))


# ============================================================================
# PER-MOLECULE EVALUATOR
# ============================================================================

def evaluate_molecule(
    query_instance: pd.DataFrame,
    original_features: pd.Series,
    explainer: GCMSDiceExplainer,
    scaler,
    model,
    feature_cols: list,
    rt_shift: float = RT_SHIFT,
    n_cfs: int = N_COUNTERFACTUALS,
) -> dict:
    """
    Run DiCE + causal filter for one molecule and return all metrics.
    """
    current_rt = make_prediction(query_instance, scaler, model)
    target_range = (max(0.5, current_rt - rt_shift - 1.0),
                    current_rt - rt_shift + 1.0)

    # Generate CFs
    try:
        cf_obj, _ = explainer.generate_counterfactuals(
            query_instance,
            target_rt_range=target_range,
            n_counterfactuals=n_cfs,
        )
        cf_df = cf_obj.cf_examples_list[0].final_cfs_df
    except Exception as e:
        return {'error': str(e), 'original_rt': current_rt}

    if cf_df is None or len(cf_df) == 0:
        return {'error': 'no_cfs', 'original_rt': current_rt}

    # --- Unconstrained metrics (raw DiCE output) ---
    validity_unconstrained = compute_validity(cf_df, target_range)
    sparsity_unconstrained = compute_sparsity(
        original_features, cf_df, feature_cols)
    proximity_unconstrained = compute_proximity(
        original_features, cf_df, feature_cols)
    causal_unconstrained = compute_causal_metrics(
        original_features, cf_df, feature_cols)
    diversity_unconstrained = compute_diversity(cf_df, feature_cols)

    # --- Causal filter ---
    filt = CausalCFFilter()
    valid_df, checks_df = filt.filter_and_rank(
        original_features, cf_df, feature_cols,
        min_causal_score=MIN_CAUSAL_SCORE
    )

    # Constrained metrics (after causal filter)
    if len(valid_df) > 0:
        # Reconstruct filtered cf_df rows
        valid_indices = valid_df['cf_index'].astype(int) - 1
        cf_df_filtered = cf_df.iloc[valid_indices.values].reset_index(
            drop=True)

        validity_constrained = compute_validity(cf_df_filtered, target_range)
        sparsity_constrained = compute_sparsity(
            original_features, cf_df_filtered, feature_cols)
        proximity_constrained = compute_proximity(
            original_features, cf_df_filtered, feature_cols)
        causal_constrained = compute_causal_metrics(
            original_features, cf_df_filtered, feature_cols)
        diversity_constrained = compute_diversity(cf_df_filtered, feature_cols)
        n_valid_cfs = len(valid_df)
    else:
        validity_constrained = 0.0
        sparsity_constrained = float('nan')
        proximity_constrained = float('nan')
        causal_constrained = {'avg_causal_score': float('nan'),
                              'causal_validity_rate': 0.0}
        diversity_constrained = float('nan')
        n_valid_cfs = 0

    return {
        'original_rt':              round(current_rt, 4),
        'target_min':               round(target_range[0], 4),
        'target_max':               round(target_range[1], 4),
        'n_cfs_generated':          len(cf_df),
        'n_cfs_causal_valid':       n_valid_cfs,

        # Unconstrained
        'validity_unconstrained':   round(validity_unconstrained, 4),
        'sparsity_unconstrained':   round(sparsity_unconstrained, 4),
        'proximity_unconstrained':  round(proximity_unconstrained, 4),
        'causal_score_unconstrained': causal_unconstrained['avg_causal_score'],
        'causal_rate_unconstrained':  causal_unconstrained['causal_validity_rate'],
        'diversity_unconstrained':  round(diversity_unconstrained, 4),

        # Constrained
        'validity_constrained':     round(validity_constrained, 4),
        'sparsity_constrained':     round(sparsity_constrained, 4) if not np.isnan(sparsity_constrained) else float('nan'),
        'proximity_constrained':    round(proximity_constrained, 4) if not np.isnan(proximity_constrained) else float('nan'),
        'causal_score_constrained': causal_constrained['avg_causal_score'],
        'causal_rate_constrained':  causal_constrained['causal_validity_rate'],
        'diversity_constrained':    round(diversity_constrained, 4) if not np.isnan(diversity_constrained) else float('nan'),
    }


# ============================================================================
# AGGREGATE REPORT
# ============================================================================

def print_aggregate_report(results_df: pd.DataFrame):
    """Print a formatted summary table of all metrics."""
    ok = results_df[results_df.get('error', pd.Series([None]*len(results_df))).isna()
                    if 'error' in results_df.columns
                    else results_df.index >= 0]

    metrics = [
        ('Validity',           'validity_unconstrained',   'validity_constrained'),
        ('Sparsity (↓ better)', 'sparsity_unconstrained',   'sparsity_constrained'),
        ('Proximity L1 (↓)',   'proximity_unconstrained',  'proximity_constrained'),
        ('Causal Score (↑)',   'causal_score_unconstrained',
         'causal_score_constrained'),
        ('Causal Valid Rate',  'causal_rate_unconstrained', 'causal_rate_constrained'),
        ('Diversity',          'diversity_unconstrained',   'diversity_constrained'),
    ]

    print(f"\n{'='*70}")
    print("  AGGREGATE EVALUATION RESULTS")
    print(f"  Molecules evaluated: {len(ok)}")
    print(f"{'='*70}")
    print(f"  {'Metric':<26} {'Unconstrained':>15} {'Constrained':>15} {'Δ':>8}")
    print(f"  {'-'*66}")

    for label, unc_col, con_col in metrics:
        unc = ok[unc_col].mean() if unc_col in ok.columns else float('nan')
        con = ok[con_col].mean() if con_col in ok.columns else float('nan')
        delta = con - unc if not (np.isnan(unc)
                                  or np.isnan(con)) else float('nan')
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "  n/a"
        print(f"  {label:<26} {unc:>15.4f} {con:>15.4f} {delta_str:>8}")

    print(f"  {'='*66}")

    # Causal valid CFs
    avg_valid = ok['n_cfs_causal_valid'].mean(
    ) if 'n_cfs_causal_valid' in ok.columns else float('nan')
    avg_gen = ok['n_cfs_generated'].mean(
    ) if 'n_cfs_generated' in ok.columns else float('nan')
    print(f"\n  Avg CFs generated:      {avg_gen:.1f}")
    print(f"  Avg CFs causal-valid:   {avg_valid:.1f}")
    print(f"  Overall causal pass rate: {avg_valid/avg_gen*100:.1f}%")
    print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  GC-MS Counterfactual Evaluation — Step 3")
    print("=" * 70)

    # Load
    scaler, model = load_artifacts()
    X, y, feature_cols, df_features = load_and_prepare_data()

    # Sample N_QUERY_MOLECULES diverse molecules
    np.random.seed(RANDOM_SEED)
    # Prefer HP-5MS for consistency, sample across RT range
    hpms = df_features[df_features['column_type'] == 'HP-5MS'].copy()
    hpms['rt_bin'] = pd.cut(hpms['retention_time'], bins=N_QUERY_MOLECULES)
    sampled_idx = (
        hpms.groupby('rt_bin', observed=True)
        .apply(lambda g: g.sample(1, random_state=RANDOM_SEED)
               if len(g) > 0 else g)
        .index.get_level_values(1)
    )
    sampled_idx = list(sampled_idx[:N_QUERY_MOLECULES])

    print(f"\nEvaluating {len(sampled_idx)} molecules on HP-5MS...")
    print(f"DiCE CFs per molecule: {N_COUNTERFACTUALS}")
    print(f"Target RT shift:       -{RT_SHIFT} min")
    print(f"Causal score threshold: {MIN_CAUSAL_SCORE}")

    # Build explainer once (reuse across queries)
    explainer = GCMSDiceExplainer(scaler, model, X, feature_cols)

    # Evaluate each molecule
    all_results = []
    for i, idx in enumerate(sampled_idx):
        smiles = df_features.loc[idx, 'smiles']
        true_rt = y[idx]
        print(
            f"\n[{i+1:02d}/{len(sampled_idx)}] {smiles}  RT={true_rt:.2f} min", end='  ')

        query_instance = X.loc[[idx]]
        original_features = X.loc[idx]

        result = evaluate_molecule(
            query_instance, original_features,
            explainer, scaler, model, feature_cols
        )
        result['smiles'] = smiles
        result['true_rt'] = true_rt
        result['mol_index'] = idx

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"✓  causal_valid={result['n_cfs_causal_valid']}/{result['n_cfs_generated']}  "
                  f"causal_score={result['causal_score_constrained']}")

        all_results.append(result)

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    # Print aggregate report
    print_aggregate_report(results_df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(f'{OUTPUT_DIR}/evaluation_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/evaluation_results.csv")
    print("\n✓ Step 3 complete. Full pipeline: DiCE → DoWhy → Evaluation.")


if __name__ == '__main__':
    main()
