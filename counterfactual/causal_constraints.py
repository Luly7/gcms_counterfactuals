#!/usr/bin/env python3
"""
DoWhy Causal Constraints for GC-MS Counterfactual Explanations
Step 2: Define causal graph and filter/score DiCE counterfactuals

The causal graph encodes domain knowledge from analytical chemistry:

  molecular_weight ──► logp ──────────────────► retention_time
  molecular_weight ──► tpsa ──────────────────► retention_time
  molecular_weight ──► labuteasa ─────────────► retention_time
  molecular_weight ──► molmr ─────────────────► retention_time
  logp             ──► peoe_vsa1/2/3/4/5 ─────► retention_time
  logp             ──► slogp_vsa1/2 ──────────► retention_time
  tpsa             ──► num_h_acceptors ────────► retention_time
  tpsa             ──► num_h_donors ──────────► retention_time
  num_aromatic_rings ► chi0v/chi1v ───────────► retention_time
  num_aromatic_rings ► balabanj ──────────────► retention_time
  column_id        ──► retention_time  (immutable)
  start_temp       ──► retention_time  (immutable)
  heating_rate     ──► retention_time  (immutable)

Causal constraints mean: if a counterfactual changes a PARENT feature,
its CHILD features must move consistently — you can't freely vary them
independently.

"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CAUSAL GRAPH DEFINITION
# ============================================================================

# Each entry: parent → [children]
# Direction: changing the parent should cause children to move accordingly
CAUSAL_EDGES = {
    'molecular_weight': ['logp', 'tpsa', 'labuteasa', 'molmr',
                         'labuteasa', 'smr_vsa1', 'smr_vsa2'],
    'logp':             ['peoe_vsa1', 'peoe_vsa2', 'peoe_vsa3',
                         'peoe_vsa4', 'peoe_vsa5',
                         'slogp_vsa1', 'slogp_vsa2'],
    'tpsa':             ['num_h_acceptors', 'num_h_donors'],
    'num_aromatic_rings': ['chi0v', 'chi1v', 'balabanj', 'kappa1', 'kappa2'],
    'num_heteroatoms':  ['tpsa', 'num_h_acceptors', 'num_h_donors'],
    'fraction_csp3':    ['kappa1', 'kappa2', 'hallkieralpha'],
}

# Expected direction of correlation (sign of Pearson r in the data)
# +1 means child increases when parent increases, -1 means it decreases
EXPECTED_DIRECTION = {
    ('molecular_weight', 'logp'): +1,
    ('molecular_weight', 'tpsa'): +1,
    ('molecular_weight', 'labuteasa'): +1,
    ('molecular_weight', 'molmr'): +1,
    ('molecular_weight', 'smr_vsa1'): +1,
    ('molecular_weight', 'smr_vsa2'): +1,
    ('logp', 'peoe_vsa1'): -1,
    ('logp', 'peoe_vsa2'): -1,
    ('logp', 'peoe_vsa3'): -1,
    ('logp', 'peoe_vsa4'): -1,
    ('logp', 'peoe_vsa5'): -1,
    ('logp', 'slogp_vsa1'): +1,
    ('logp', 'slogp_vsa2'): +1,
    ('tpsa', 'num_h_acceptors'): +1,
    ('tpsa', 'num_h_donors'): +1,
    ('num_aromatic_rings', 'chi0v'): +1,
    ('num_aromatic_rings', 'chi1v'): +1,
    ('num_aromatic_rings', 'balabanj'): +1,
    ('num_aromatic_rings', 'kappa1'): -1,
    ('num_aromatic_rings', 'kappa2'): -1,
    ('num_heteroatoms', 'tpsa'): +1,
    ('num_heteroatoms', 'num_h_acceptors'): +1,
    ('num_heteroatoms', 'num_h_donors'): +1,
    ('fraction_csp3', 'kappa1'): +1,
    ('fraction_csp3', 'kappa2'): +1,
    ('fraction_csp3', 'hallkieralpha'): +1,
}


# ============================================================================
# CAUSAL CONSTRAINT CHECKER
# ============================================================================

class CausalConstraintChecker:
    """
    Checks whether a counterfactual respects the causal structure
    of molecular descriptors.

    A counterfactual violates causal consistency if it changes a
    child feature in the WRONG direction relative to its parent's change,
    or changes a child WITHOUT changing its parent (orphan change).
    """

    def __init__(self, causal_edges: dict = None,
                 expected_direction: dict = None):
        self.causal_edges = causal_edges or CAUSAL_EDGES
        self.expected_direction = expected_direction or EXPECTED_DIRECTION

        # Build reverse index: child → parents
        self.parents_of = {}
        for parent, children in self.causal_edges.items():
            for child in children:
                self.parents_of.setdefault(child, []).append(parent)

    def check_counterfactual(
        self,
        original: pd.Series,
        counterfactual: pd.Series,
        tol: float = 1e-4,
    ) -> dict:
        """
        Check a single counterfactual for causal consistency.

        Args:
            original:        Original feature values (pd.Series)
            counterfactual:  Counterfactual feature values (pd.Series)
            tol:             Minimum delta to count as a change

        Returns:
            dict with keys:
              - violations: list of (parent, child, reason) tuples
              - n_violations: int
              - causal_score: float in [0, 1], 1 = fully consistent
              - changed_features: list of features that changed
        """
        delta = counterfactual - original
        changed = set(delta[delta.abs() > tol].index)

        violations = []

        for feat in changed:
            # Check: if this feature is a child, did its parent change
            # in a causally consistent direction?
            if feat in self.parents_of:
                for parent in self.parents_of[feat]:
                    if parent not in changed:
                        # Child changed but parent didn't — orphan change
                        violations.append((
                            parent, feat,
                            f"'{feat}' changed but its causal parent "
                            f"'{parent}' did not"
                        ))
                    else:
                        # Both changed — check direction
                        key = (parent, feat)
                        if key in self.expected_direction:
                            expected_sign = self.expected_direction[key]
                            actual_sign_parent = np.sign(delta[parent])
                            actual_sign_child = np.sign(delta[feat])
                            if actual_sign_parent * actual_sign_child != expected_sign:
                                violations.append((
                                    parent, feat,
                                    f"'{feat}' moved {'+' if delta[feat] > 0 else '-'} "
                                    f"but '{parent}' moved "
                                    f"{'+' if delta[parent] > 0 else '-'} "
                                    f"(expected {'same' if expected_sign == 1 else 'opposite'} direction)"
                                ))

        # Score: fraction of causal relationships respected
        total_relationships = sum(
            len(children) for children in self.causal_edges.values()
        )
        n_violations = len(violations)
        causal_score = max(0.0, 1.0 - n_violations /
                           max(total_relationships, 1))

        return {
            'violations':        violations,
            'n_violations':      n_violations,
            'causal_score':      round(causal_score, 4),
            'changed_features':  list(changed),
            'n_changed':         len(changed),
        }

    def check_all(
        self,
        original: pd.Series,
        cf_df: pd.DataFrame,
        feature_cols: list,
    ) -> pd.DataFrame:
        """
        Check all counterfactuals in a DataFrame.

        Args:
            original:     Original feature values
            cf_df:        DiCE output DataFrame (rows = counterfactuals)
            feature_cols: The 37 feature names

        Returns:
            DataFrame with causal check results per CF
        """
        results = []
        for i, (_, row) in enumerate(cf_df.iterrows()):
            cf_features = row[feature_cols]
            check = self.check_counterfactual(original, cf_features)
            check['cf_index'] = i + 1
            check['predicted_rt'] = row.get('retention_time', float('nan'))
            results.append(check)

        return pd.DataFrame(results)


# ============================================================================
# CAUSAL FILTER — keeps only causally valid CFs
# ============================================================================

class CausalCFFilter:
    """
    Filters and ranks DiCE counterfactuals by causal consistency.
    """

    def __init__(self, checker: CausalConstraintChecker = None):
        self.checker = checker or CausalConstraintChecker()

    def filter_and_rank(
        self,
        original: pd.Series,
        cf_df: pd.DataFrame,
        feature_cols: list,
        min_causal_score: float = 0.85,
    ) -> pd.DataFrame:
        """
        Filter CFs by causal score and rank by (causal_score, proximity).

        Args:
            original:         Original feature values
            cf_df:            DiCE output DataFrame
            feature_cols:     37 feature names
            min_causal_score: Minimum causal score to keep a CF

        Returns:
            Filtered + ranked DataFrame with causal metadata
        """
        checks = self.checker.check_all(original, cf_df, feature_cols)

        # Compute L1 proximity (smaller = closer to original)
        proximity = []
        for _, row in cf_df.iterrows():
            l1 = (row[feature_cols] - original[feature_cols]).abs().sum()
            proximity.append(float(l1))
        checks['l1_proximity'] = proximity

        # Filter
        valid = checks[checks['causal_score'] >= min_causal_score].copy()

        # Rank: highest causal score first, then closest
        valid = valid.sort_values(
            ['causal_score', 'l1_proximity'],
            ascending=[False, True]
        ).reset_index(drop=True)

        return valid, checks


# ============================================================================
# PRETTY PRINTER
# ============================================================================

def print_causal_report(
    original: pd.Series,
    cf_df: pd.DataFrame,
    checks_df: pd.DataFrame,
    feature_cols: list,
):
    """Print a detailed causal consistency report."""
    print(f"\n{'='*65}")
    print("  CAUSAL CONSISTENCY REPORT")
    print(f"{'='*65}")

    for _, row in checks_df.iterrows():
        i = int(row['cf_index'])
        score = row['causal_score']
        n_viol = row['n_violations']
        rt = row['predicted_rt']
        status = "✓ VALID" if n_viol == 0 else f"✗ {n_viol} violation(s)"

        print(f"\n  CF #{i}  RT={rt:.3f} min  "
              f"causal_score={score:.3f}  {status}")

        if n_viol > 0:
            violations = row['violations']
            for parent, child, reason in violations:
                print(f"    ⚠  {reason}")

        # Show changes
        cf_row = cf_df.iloc[i - 1]
        changed = [f for f in feature_cols
                   if abs(cf_row[f] - original[f]) > 1e-4]
        if changed:
            print(f"    Changed features: {', '.join(changed)}")


# ============================================================================
# MAIN — integrates with dice_explainer.py
# ============================================================================

def run_causal_analysis(
    original_features: pd.Series,
    cf_obj,                    # DiCE cf_examples object
    feature_cols: list,
    min_causal_score: float = 0.85,
    output_dir: str = 'counterfactual/output',
):
    """
    Run full causal constraint analysis on DiCE counterfactuals.

    Args:
        original_features:  pd.Series of original molecule features
        cf_obj:             DiCE counterfactual examples object
        feature_cols:       37 feature column names
        min_causal_score:   Minimum score to keep a CF (0–1)
        output_dir:         Where to save results

    Returns:
        valid_cfs_df:  Filtered + ranked counterfactuals
        checks_df:     Full check results for all CFs
    """
    print("\n" + "=" * 65)
    print("  DoWhy Causal Constraint Analysis")
    print("=" * 65)

    cf_df = cf_obj.cf_examples_list[0].final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        print("  No counterfactuals to analyse.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Causal graph: {len(CAUSAL_EDGES)} parent nodes, "
          f"{sum(len(v) for v in CAUSAL_EDGES.values())} edges")
    print(f"  Checking {len(cf_df)} counterfactuals...")

    checker = CausalConstraintChecker()
    filt = CausalCFFilter(checker)

    valid_df, checks_df = filt.filter_and_rank(
        original_features, cf_df, feature_cols,
        min_causal_score=min_causal_score
    )

    # Print report
    print_causal_report(original_features, cf_df, checks_df, feature_cols)

    # Summary
    n_total = len(cf_df)
    n_valid = len(valid_df)
    print(f"\n  Summary: {n_valid}/{n_total} counterfactuals pass "
          f"causal filter (score ≥ {min_causal_score})")

    if len(valid_df) > 0:
        print(f"\n  Top causally-valid CF:")
        best = valid_df.iloc[0]
        print(f"    predicted RT   = {best['predicted_rt']:.3f} min")
        print(f"    causal score   = {best['causal_score']:.3f}")
        print(f"    features changed = {best['n_changed']}")
        print(f"    L1 proximity   = {best['l1_proximity']:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    checks_df.to_csv(f'{output_dir}/causal_checks.csv', index=False)
    valid_df.to_csv(f'{output_dir}/causal_valid_cfs.csv', index=False)
    print(f"\n  Saved to {output_dir}/")
    print(f"    causal_checks.csv     — all CFs with violation details")
    print(f"    causal_valid_cfs.csv  — filtered + ranked valid CFs")

    return valid_df, checks_df


# ============================================================================
# STANDALONE DEMO (runs DiCE + DoWhy together)
# ============================================================================

if __name__ == '__main__':
    import joblib
    from counterfactual.dice_explainer import (
        load_artifacts, load_and_prepare_data,
        make_prediction, GCMSDiceExplainer
    )

    print("=" * 65)
    print("  GC-MS Causally-Constrained Counterfactual System")
    print("  Step 2: DoWhy Causal Constraints")
    print("=" * 65)

    # Load
    scaler, model = load_artifacts()
    X, y, feature_cols, df_features = load_and_prepare_data()

    # Query: benzene on HP-5MS
    hpms_mask = df_features['column_type'] == 'HP-5MS'
    query_idx = df_features[hpms_mask].index[0]
    query_instance = X.loc[[query_idx]]
    original_features = X.loc[query_idx]

    print(f"\nQuery: {df_features.loc[query_idx, 'smiles']} "
          f"on {df_features.loc[query_idx, 'column_type']}")
    print(f"True RT: {y[query_idx]:.3f} min")

    # DiCE
    explainer = GCMSDiceExplainer(scaler, model, X, feature_cols)
    current_rt = make_prediction(query_instance, scaler, model)
    target_range = (max(0.5, current_rt - 5.0), current_rt - 2.0)

    print(f"\nGenerating DiCE counterfactuals...")
    cf_obj, current_rt = explainer.generate_counterfactuals(
        query_instance, target_rt_range=target_range, n_counterfactuals=5
    )

    # DoWhy causal filtering
    valid_df, checks_df = run_causal_analysis(
        original_features, cf_obj, feature_cols
    )

    print("\n✓ Step 2 complete. Pipeline: DiCE → DoWhy causal filter.")
    print("  Next: evaluation metrics (validity, sparsity, plausibility).")
