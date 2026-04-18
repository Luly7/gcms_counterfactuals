from feature_extraction.feature_pipeline import CompleteFeatureExtractor
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd

sys.path.insert(0, '.')

# Load data and rename columns to what the extractor expects
df = pd.read_csv('data/synthetic_gcms_data_v3_2000.csv')
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

X = np.nan_to_num(df_features[feature_cols].values, nan=0.0)
y = df_features['retention_time'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(n_estimators=200, learning_rate=0.1,
                     max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} min")

Path('models').mkdir(exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(model, 'models/xgboost_final.pkl')
print("Saved models/scaler.pkl and models/xgboost_final.pkl (37 features)")
