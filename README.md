# Causally-Constrained Counterfactual Explanations for GC-MS Retention Time Prediction

**CS 6460 · Lourdes Castleton · Utah Valley University**  
See video presentation :
https://youtu.be/8lQN3KkD7M0
## Overview
A four-step pipeline that extends an XGBoost-based GC-MS retention time 
prediction model with causally-constrained counterfactual explanations 
using DiCE and DoWhy.

## Key Results
- **95% causal pass rate** → 100% after constraint enforcement
- No systematic proximity cost (r = −0.14, p = 0.603)
- Robust across full GC-MS RT range (7–38 min)

## Pipeline
1. XGBoost predicts retention time for query molecule
2. DiCE generates 5 diverse counterfactuals (±2 min window)
3. DoWhy validates each CF against the causal DAG
4. Metrics computed and figures generated

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
cd gcms_counterfactual
PYTHONPATH=. python counterfactual/dice_explainer.py
PYTHONPATH=. python counterfactual/evaluate.py
PYTHONPATH=. python counterfactual/visualize.py
```

## Built on
- XGBoost · DiCE · DoWhy · RDKit · scikit-learn
- Extends CS 6610 capstone: github.com/Luly7/CAPSTONE
