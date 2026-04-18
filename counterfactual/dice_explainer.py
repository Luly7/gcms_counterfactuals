#!/usr/bin/env python3
"""
DiCE Counterfactual Explainer for GC-MS Retention Time Prediction
Step 1: Generate counterfactual explanations using DiCE

Usage:
    python counterfactual/dice_explainer.py

Project: CS 6460 - Causally-Constrained Counterfactual Explanations
"""

from feature_extraction.feature_pipeline import CompleteFeatureExtractor
import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'models/xgboost_final.pkl'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/synthetic_gcms_data_v3_2000.csv'

# Features the model can suggest changing (actionable)
# We exclude purely identity features like column_id which are fixed per experiment
ACTIONABLE_FEATURES = [
    'molecular_weight', 'logp', 'tpsa', 'num_rotatable_bonds',
    'num_h_acceptors', 'num_h_donors', 'num_aromatic_rings',
    'num_aliphatic_rings', 'num_saturated_rings', 'num_heteroatoms',
    'fraction_csp3', 'num_bridgehead_atoms', 'chi0v', 'chi1v',
    'kappa1', 'kappa2', 'balabanj', 'hallkieralpha', 'molmr',
    'labuteasa', 'peoe_vsa1', 'peoe_vsa2', 'peoe_vsa3', 'peoe_vsa4',
    'peoe_vsa5', 'smr_vsa1', 'smr_vsa2', 'slogp_vsa1', 'slogp_vsa2',
    'max_partial_charge', 'min_partial_charge',
]

# Features fixed for a given experiment (cannot be changed by counterfactual)
IMMUTABLE_FEATURES = [
    'column_id', 'start_temp', 'end_temp', 'temp_range',
    'heating_rate', 'flow_rate',
]


# ============================================================================
# HELPERS
# ============================================================================

def load_artifacts():
    """Load trained scaler and model."""
    print("Loading model artifacts...")
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print(f"  Scaler: {scaler.n_features_in_} features")
    print(f"  Model:  loaded OK")
    return scaler, model


def load_and_prepare_data():
    """Load CSV and extract 37 features."""
    print("Loading and extracting features from dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        'SMILES': 'smiles',
        'RT': 'retention_time',
        'Column': 'column_type',
        'TempProgram': 'temperature_program',
        'FlowRate': 'flow_rate',
    })

    extractor = CompleteFeatureExtractor()
    df_features = extractor.extract_batch(df)
    feature_cols = extractor.get_feature_columns()

    # Deduplicate columns (safety net)
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]

    X = df_features[feature_cols].copy()
    X = X.fillna(0.0)
    y = df_features['retention_time'].values

    print(f"  Dataset: {len(X)} rows x {len(feature_cols)} features")
    return X, y, feature_cols, df_features


def make_prediction(X_row: pd.DataFrame, scaler, model) -> float:
    """Predict RT for a single feature row."""
    X_scaled = scaler.transform(X_row)
    return float(model.predict(X_scaled)[0])


# ============================================================================
# DiCE EXPLAINER CLASS
# ============================================================================

class GCMSDiceExplainer:
    """
    Wraps DiCE to generate counterfactual explanations for RT predictions.

    DiCE (Diverse Counterfactual Explanations) answers:
    "What minimal changes to molecular features would shift the
     predicted retention time to a different target range?"
    """

    def __init__(self, scaler, model, X_train: pd.DataFrame,
                 feature_cols: list):
        self.scaler = scaler
        self.model = model
        self.feature_cols = feature_cols
        self.X_train = X_train
        self.dice_exp = None
        self._setup_dice(X_train)

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction function DiCE calls internally."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def _setup_dice(self, X_train: pd.DataFrame):
        """Initialize DiCE data and explainer objects."""
        try:
            import dice_ml
        except ImportError:
            raise ImportError(
                "DiCE not installed. Run:\n"
                "  pip install dice-ml"
            )

        print("Setting up DiCE explainer...")

        # DiCE needs a DataFrame with outcome column
        train_with_outcome = X_train.copy()
        train_with_outcome['retention_time'] = self.model.predict(
            self.scaler.transform(X_train.values)
        )

        # Define feature types for DiCE
        continuous_features = self.feature_cols  # all 37 are continuous

        d = dice_ml.Data(
            dataframe=train_with_outcome,
            continuous_features=continuous_features,
            outcome_name='retention_time'
        )

        m = dice_ml.Model(
            model=self,           # use self as wrapper with predict()
            backend='sklearn',
            model_type='regressor'
        )

        self.dice_exp = dice_ml.Dice(d, m, method='random')
        print("  DiCE explainer ready.")

    def predict(self, X):
        """Callable interface DiCE uses for the model."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        return self._predict_fn(X)

    def generate_counterfactuals(
        self,
        query_instance: pd.DataFrame,
        target_rt_range: tuple,
        n_counterfactuals: int = 5,
        features_to_vary: list = None,
    ) -> pd.DataFrame:
        """
        Generate counterfactual explanations.

        Args:
            query_instance:      Single-row DataFrame with 37 features
            target_rt_range:     (min_rt, max_rt) desired RT window
            n_counterfactuals:   How many diverse CFs to generate
            features_to_vary:    Which features DiCE can change
                                 (defaults to ACTIONABLE_FEATURES)

        Returns:
            DataFrame of counterfactual feature vectors with predicted RT
        """
        if features_to_vary is None:
            features_to_vary = ACTIONABLE_FEATURES

        # Clip to features actually in our set
        features_to_vary = [f for f in features_to_vary
                            if f in self.feature_cols]

        current_rt = make_prediction(query_instance, self.scaler, self.model)
        print(f"\n  Query RT (predicted): {current_rt:.3f} min")
        print(f"  Target RT range:      {target_rt_range[0]:.1f} – "
              f"{target_rt_range[1]:.1f} min")
        print(f"  Features DiCE may vary: {len(features_to_vary)}")

        cf = self.dice_exp.generate_counterfactuals(
            query_instance,
            total_CFs=n_counterfactuals,
            desired_range=list(target_rt_range),
            features_to_vary=features_to_vary,
            verbose=False,
        )

        return cf, current_rt

    @staticmethod
    def summarise_counterfactuals(cf_obj, query_instance: pd.DataFrame,
                                  current_rt: float,
                                  feature_cols: list) -> pd.DataFrame:
        """
        Pretty-print which features changed and by how much.
        Returns a summary DataFrame.
        """
        cf_df = cf_obj.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            print("  No valid counterfactuals found.")
            return pd.DataFrame()

        print(f"\n{'='*65}")
        print(
            f"  COUNTERFACTUAL SUMMARY  (original RT = {current_rt:.3f} min)")
        print(f"{'='*65}")

        rows = []
        for i, (_, cf_row) in enumerate(cf_df.iterrows()):
            cf_rt = cf_row.get('retention_time', float('nan'))
            changed = {}
            for feat in feature_cols:
                orig_val = float(query_instance[feat].iloc[0])
                cf_val = float(cf_row[feat])
                if not np.isclose(orig_val, cf_val, rtol=1e-3, atol=1e-6):
                    changed[feat] = (orig_val, cf_val, cf_val - orig_val)

            print(f"\n  CF #{i+1}  →  predicted RT = {cf_rt:.3f} min")
            print(f"  {'Feature':<28} {'Original':>10} {'CF Value':>10} "
                  f"{'Δ':>10}")
            print(f"  {'-'*60}")
            for feat, (orig, new, delta) in sorted(
                    changed.items(), key=lambda x: abs(x[1][2]), reverse=True):
                print(f"  {feat:<28} {orig:>10.4f} {new:>10.4f} "
                      f"{delta:>+10.4f}")

            rows.append({
                'cf_index': i + 1,
                'predicted_rt': cf_rt,
                'n_features_changed': len(changed),
                **{f'delta_{k}': v[2] for k, v in changed.items()}
            })

        return pd.DataFrame(rows)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("=" * 65)
    print("  GC-MS DiCE Counterfactual Explainer")
    print("=" * 65)

    # 1. Load artifacts
    scaler, model = load_artifacts()

    # 2. Load data
    X, y, feature_cols, df_features = load_and_prepare_data()

    # 3. Pick a query molecule — benzene on HP-5MS (row 0 is benzene/ZB-5,
    #    find first HP-5MS row)
    hpms_mask = df_features['column_type'] == 'HP-5MS'
    query_idx = df_features[hpms_mask].index[0]
    query_instance = X.loc[[query_idx]]
    true_rt = y[query_idx]

    print(f"\nQuery molecule:")
    print(f"  SMILES:       {df_features.loc[query_idx, 'smiles']}")
    print(f"  Column:       {df_features.loc[query_idx, 'column_type']}")
    print(f"  True RT:      {true_rt:.3f} min")

    # 4. Build DiCE explainer
    explainer = GCMSDiceExplainer(scaler, model, X, feature_cols)

    # 5. Generate counterfactuals — ask: what would make RT < 8 min?
    current_rt = make_prediction(query_instance, scaler, model)
    target_range = (max(0.5, current_rt - 5.0), current_rt - 2.0)

    print(f"\nGenerating counterfactuals to shift RT lower...")
    cf_obj, current_rt = explainer.generate_counterfactuals(
        query_instance,
        target_rt_range=target_range,
        n_counterfactuals=5,
    )

    # 6. Summarise
    summary_df = GCMSDiceExplainer.summarise_counterfactuals(
        cf_obj, query_instance, current_rt, feature_cols
    )

    # 7. Save
    os.makedirs('counterfactual/output', exist_ok=True)
    summary_df.to_csv('counterfactual/output/cf_summary.csv', index=False)
    cf_obj.cf_examples_list[0].final_cfs_df.to_csv(
        'counterfactual/output/cf_full.csv', index=False
    )
    print(f"\nSaved results to counterfactual/output/")
    print("  cf_summary.csv  — delta per feature per CF")
    print("  cf_full.csv     — full feature vectors")

    print("\n✓ Step 1 complete. Next: integrate DoWhy causal constraints.")


if __name__ == '__main__':
    main()
