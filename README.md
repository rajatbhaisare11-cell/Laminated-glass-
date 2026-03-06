# Laminated-glass-

## Rebound–Penetration Threshold Prediction

### How to Run

Open MATLAB (R2020a or later) in the repository root and run:

```matlab
>> Threshold_Prediction
```

### Requirements

- MATLAB R2020a+
- Statistics and Machine Learning Toolbox
- `Combined_DATA_SET_TO_BE_USED.xlsx` present in the working directory

### What It Does

`Threshold_Prediction.m` is a single self-contained script that:

1. **Loads** impact test data from `Combined_DATA_SET_TO_BE_USED.xlsx` (2 sheets, 32 rows total)
2. **Trains** three classifiers — SVM (RBF kernel), Logistic Regression, and Random Forest — and evaluates them with leave-one-out cross-validation (LOOCV)
3. **Computes** the penetration threshold KE and velocity for every combination of {Config 1, Config 2} × {center, corner} × {mass 1, 2, 3, 4 kg} via bisection search on the best-performing model
4. **Generates** six figures saved as PNG:
   - `KE_vs_Status.png` — scatter of KE vs penetration status per config
   - `Threshold_KE.png` — threshold KE vs mass curves
   - `Threshold_Vel.png` — threshold velocity vs mass
   - `Confusion_Matrix.png` — confusion matrix for the best model
   - `Decision_Boundary.png` — classification decision boundary in (Mass, KE) space
   - `ROC_Curve.png` — ROC curves for all three models
5. **Exports** results to `Threshold_Results.xlsx`:
   - Sheet 1: threshold KE and velocity table
   - Sheet 2: model accuracy comparison

### Dataset

| Config | Layers | Rows |
|--------|--------|------|
| 1 | 3 mm Glass / 1.52 mm PVB / 3 mm Glass / 1.52 mm PVB / 3 mm Glass (5-layer) | 16 |
| 2 | 6 mm Glass / 1.52 mm PVB / 6 mm Glass (3-layer) | 16 |