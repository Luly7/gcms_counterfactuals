import joblib

# Load the correct files
scaler = joblib.load(
    '/Users/luly/Desktop/gcms_counterfactual/models/scaler.pkl')
model = joblib.load(
    '/Users/luly/Desktop/gcms_counterfactual/models/xgboost_final.pkl')

# What did the scaler see when it was trained?
print("=== scaler.pkl expects ===")
print(f"N features: {scaler.n_features_in_}")
if hasattr(scaler, 'feature_names_in_'):
    for i, f in enumerate(scaler.feature_names_in_):
        print(f"  {i:02d}  {f}")
else:
    print("(no feature names stored)")

# What does XGBoost expect?
print("\n=== xgboost_final.pkl expects ===")
if hasattr(model, 'feature_names_in_'):
    print(f"N features: {len(model.feature_names_in_)}")
    for i, f in enumerate(model.feature_names_in_):
        print(f"  {i:02d}  {f}")
else:
    print("(no feature names stored)")
