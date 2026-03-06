# MATLAB Predictive Framework — Laminated Glass Impact Analysis

## Project Description

This MATLAB framework implements a comprehensive data-driven approach to determine the **rebound–penetration threshold** of laminated glass panels subjected to lumbar impact. It uses finite-element simulation data (LS-DYNA), trains multiple classification models, and calculates the threshold kinetic energy at which an impactor transitions from rebound to penetration for each glass configuration, impact location, and impactor mass.

---

## Dataset Description

The hardcoded dataset contains **32 rows** from LS-DYNA FEA simulations across two laminated glass configurations and two impact locations.

### Configuration 1 — 5-layer (rows 1–16)
| Layer  | Material | Thickness |
|--------|----------|-----------|
| Glass1 | Glass    | 3 mm      |
| PVB1   | PVB      | 1.52 mm   |
| Glass2 | Glass    | 3 mm      |
| PVB2   | PVB      | 1.52 mm   |
| Glass3 | Glass    | 3 mm      |

Total glass: 9 mm | Total PVB: 3.04 mm | Panel: 12.04 mm

### Configuration 2 — 3-layer (rows 17–32)
| Layer  | Material | Thickness |
|--------|----------|-----------|
| Glass1 | Glass    | 6 mm      |
| PVB1   | PVB      | 1.52 mm   |
| Glass2 | Glass    | 6 mm      |

Total glass: 12 mm | Total PVB: 1.52 mm | Panel: 13.52 mm

### Impact Conditions
- **Impactor masses:** 1, 2, 3, 4 kg
- **Impact velocities:** 15 m/s and 25 m/s
- **KE = 0.5 × mass × velocity²**
- **Impact locations:** center, corner

---

## Requirements

### MATLAB Version
- MATLAB R2019b or newer (R2021b+ recommended)

### Required Toolboxes
- **Statistics and Machine Learning Toolbox** (required for `fitcsvm`, `fitglm`, `TreeBagger`, `fitPosterior`, `perfcurve`)

### Optional
- **Parallel Computing Toolbox** — speeds up ensemble training

---

## How to Run

### Quick Start
```matlab
cd src
main_threshold_prediction
```

### Step by Step

1. Clone/download the repository.
2. Open MATLAB and navigate to the repository root.
3. Run the master script:
   ```matlab
   cd src
   main_threshold_prediction
   ```
4. All outputs (figures + Excel) are saved to the `results/` folder.

### Load Dataset Only
```matlab
data = load_dataset();              % hardcoded data
data = load_dataset('myfile.xlsx'); % from Excel (falls back to hardcoded on error)
```

### Train a Single Model
```matlab
data = load_dataset();
X = [data.Mass_kg, data.KE_J, data.Velocity_mps, ...
     data.Glass1_mm, data.PVB1_mm, data.Glass2_mm, data.PVB2_mm, data.Glass3_mm, ...
     data.TotalGlassThickness, data.TotalPVBThickness, data.ImpactLocation];
y = data.Penetration_Status;

[model, acc, cv_acc, metrics] = train_svm_model(X, y);
```

### Compute Threshold KE
```matlab
cfg = struct('Glass1_mm',3,'PVB1_mm',1.52,'Glass2_mm',3,'PVB2_mm',1.52,'Glass3_mm',3);
feat_order = {'Mass_kg','KE_J','Velocity_mps','Glass1_mm','PVB1_mm', ...
              'Glass2_mm','PVB2_mm','Glass3_mm', ...
              'TotalGlassThickness','TotalPVBThickness','ImpactLocation'};
[thr_ke, thr_v] = compute_threshold_ke(model, 2, 1, cfg, 'SVM', feat_order);
fprintf('Threshold KE = %.1f J  |  Velocity = %.2f m/s\n', thr_ke, thr_v);
```

---

## Expected Outputs

### Console Output
- Dataset loading confirmation
- Training and cross-validation accuracy for each model
- Formatted threshold KE table

### Figures (saved to `results/`)
| File | Description |
|------|-------------|
| `KE_vs_PenetrationStatus.png` | Scatter by config and location |
| `Threshold_KE_vs_Mass.png` | Threshold KE curves |
| `Threshold_Velocity_vs_Mass.png` | Threshold velocity curves |
| `ConfusionMatrix_SVM.png` | Confusion matrix — SVM |
| `ConfusionMatrix_Logistic.png` | Confusion matrix — Logistic |
| `ConfusionMatrix_Ensemble.png` | Confusion matrix — Ensemble |
| `DecisionBoundary_Config1.png` | Decision boundary Config 1 |
| `DecisionBoundary_Config2.png` | Decision boundary Config 2 |
| `ROC_Curves.png` | ROC curves for all models |
| `3DSurface_Config1.png` | 3D P(penetration) surface Config 1 |
| `3DSurface_Config2.png` | 3D P(penetration) surface Config 2 |
| `Model_Comparison.png` | Bar chart of model metrics |
| `Feature_Importance.png` | Feature importance (Random Forest) |
| `OAT_Sensitivity.png` | One-at-a-time sensitivity |
| `PDP_Mass_kg.png` | Partial dependence — Mass |
| `PDP_KE_J.png` | Partial dependence — KE |
| `PDP_Velocity_mps.png` | Partial dependence — Velocity |

### Excel Workbook (`results/Threshold_Results.xlsx`)
| Sheet | Contents |
|-------|----------|
| Threshold_KE | Threshold KE and velocity per config/location/mass |
| Model_Comparison | Accuracy, precision, recall, F1 for each model |
| Full_Predictions | Original data + predicted labels + probabilities |
| Feature_Importance | Feature names and importance scores |

---

## File Descriptions

| File | Description |
|------|-------------|
| `main_threshold_prediction.m` | Master script — runs the full pipeline |
| `load_dataset.m` | Hardcoded dataset loader with derived features |
| `train_svm_model.m` | SVM (RBF kernel) with LOOCV and Platt scaling |
| `train_logistic_model.m` | Logistic regression (binomial GLM) with LOOCV |
| `train_ensemble_model.m` | Random Forest (TreeBagger, 100 trees) with OOB |
| `compute_threshold_ke.m` | Bisection method for threshold KE |
| `plot_all_results.m` | All publication-quality figures |
| `sensitivity_analysis.m` | OAT sensitivity and partial dependence plots |
| `export_results_to_excel.m` | Write results to Excel workbook |

---

## Methodology

### Classification Models

#### Support Vector Machine (SVM)
- Kernel: Radial Basis Function (RBF)
- Features are z-score standardised before training
- Probability estimates via Platt scaling (`fitPosterior`)
- Validation: Leave-One-Out Cross-Validation (LOOCV)

#### Logistic Regression
- Binomial family, logit link function (`fitglm`)
- Features are z-score standardised
- Validation: LOOCV

#### Random Forest
- 100 decision trees (`TreeBagger`)
- Out-of-Bag (OOB) error used as cross-validation proxy
- Feature importance via OOB permuted predictor importance

### Threshold KE via Bisection
The decision boundary is found by bisecting the KE range [0, 2000 J] until P(penetration) = 0.5:

```
KE_lo = 0, KE_hi = 2000
while (KE_hi - KE_lo) > tolerance:
    KE_mid = (KE_lo + KE_hi) / 2
    if P(penetration | KE_mid) < 0.5:
        KE_lo = KE_mid
    else:
        KE_hi = KE_mid
threshold_KE = (KE_lo + KE_hi) / 2
threshold_V  = sqrt(2 * threshold_KE / mass)
```

If no threshold exists in [0, 2000 J], `NaN` is returned.

### Feature Engineering
Raw features: `Mass_kg`, `KE_J`, `Velocity_mps`, `Glass1_mm`–`Glass3_mm`, `PVB1_mm`–`PVB2_mm`

Derived features:
- `TotalGlassThickness = Glass1 + Glass2 + Glass3`
- `TotalPVBThickness = PVB1 + PVB2`
- `TotalPanelThickness = TotalGlass + TotalPVB`
- `NumLayers` — count of non-zero glass layers
- `ConfigType` — 1 for 5-layer, 2 for 3-layer
- `ImpactLocation` — 1 for center, 0 for corner

---

## References

1. Hooper, P. A., et al. (2012). *On the blast resistance of laminated glass.* International Journal of Solids and Structures, 49(6), 899–918.
2. Martín, M., et al. (2020). *Mechanical and fracture behaviour of laminated glass: experimental and numerical study.* Engineering Structures, 224, 111185.
3. Haldimann, M., Luible, A., & Overend, M. (2008). *Structural Use of Glass.* IABSE.
4. LS-DYNA Keyword User's Manual. Livermore Software Technology (LST).
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning.* Springer.
