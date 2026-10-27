# Applied Machine Learning Portfolio (CS7CS4)

Three machine learning projects completed for **CS7CS4 Machine Learning** at Trinity College Dublin.

Each project follows an end-to-end data science workflow: exploratory visualisation, feature engineering, leakage-safe pipelines, hyperparameter tuning via cross-validation, comparison against simple baselines, and analysis of the resulting decision boundaries and prediction surfaces.

---

## Table of Contents

- [Skills Demonstrated](#skills-demonstrated)
- [Projects](#projects)
  - [Assignment 1: Logistic Regression and Linear SVM]([#assignment-1-logistic-regression-and-linear-svm](https://github.com/uri3lrex/A1_Logistic-Regression-and-Linear-SVM))
  - [Assignment 2: Lasso and Ridge Regression]([#assignment-2-lasso-and-ridge-regression](https://github.com/uri3lrex/A2_Lasso-and-Ridge-Regression))
  - [Assignment 3: Classification and No-Signal Auditing]([#assignment-3-classification-and-no-signal-auditing](https://github.com/uri3lrex/A3_Classification--Model-Selection-and-Evaluation))
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)

---

## Skills Demonstrated

| Area | Details |
|---|---|
| **Supervised learning** | Classification (Logistic Regression, LinearSVC, kNN) and regression (Lasso, Ridge) |
| **Feature engineering** | Polynomial feature maps, feature interactions, and explicit mappings for non-linear geometries |
| **Model selection and tuning** | Grid search, 5-fold stratified cross-validation, and variance reporting with error bars (mean ± std) |
| **Evaluation** | Confusion matrices, precision, recall, F1, ROC/AUC, and strict hold-out test validation |
| **Engineering practice** | scikit-learn `Pipeline`s to prevent train/validation leakage, feature standardisation, reproducible scripts |

---

## Projects

### Assignment 1: Logistic Regression and Linear SVM

**Objective:** Binary classification of 999 two-dimensional data points with a non-linear class structure.

**Methodology**
- Visualised a dome-shaped class boundary that limits purely linear separators.
- Extracted decision boundaries from trained `LogisticRegression` and `LinearSVC` parameters across regularisation strengths (`C = 0.001, 1, 100`).
- Added squared features (`x₁²`, `x₂²`) so that linear models could fit the curved boundary.

**Key result:** Quadratic features raised test accuracy from **85.0%** to **96.25%**, well above the **66.5%** majority-class baseline.

---

### Assignment 2: Lasso and Ridge Regression

**Objective:** Predict a real-valued target from a non-linear surface.

**Methodology**
- Analysed a curved 3D scatter surface and expanded the inputs to a 5th-degree polynomial space (20 features).
- Compared L1 (Lasso) and L2 (Ridge) regularisation across a wide range of penalty values.
- Plotted prediction surfaces on an extrapolated grid to contrast under-fitting, good fit, and over-fitting.
- Selected regularisation constants with 5-fold cross-validation, reporting standard error bars.

**Key result:** Lasso produced sparse models by driving some coefficients to exactly zero, while Ridge shrank all coefficients smoothly. The extrapolation grid made the risk of over-fitting visible.

---

### Assignment 3: Classification and No-Signal Auditing

**Objective:** Compare L2 Logistic Regression and k-Nearest Neighbours on two datasets with different structure.

**Methodology**
- Used an isolated 75/25 stratified train/test split so hyperparameter searches never saw the test partition.
- Ran a kNN grid search over:
  - neighbours: `k ∈ [1, 49]`
  - weighting: uniform vs Gaussian-kernel distance weights
  - metric: Euclidean vs Manhattan
- Analysed how high-gamma Gaussian kernels shrink the influence of distant neighbours, making the model behave like a 1-NN classifier and overfit.

**Key results**
- **Dataset 1:** No usable signal. Both models reached **67.5%** test accuracy (AUC ≈ 0.52), matching the majority-class baseline, so the recommendation is to deploy no model.
- **Dataset 2:** Clear non-linear structure. Logistic Regression reached **96.4%** test accuracy (AUC 0.998) and kNN reached **94.9%** (AUC 0.989), both well above the **78.3%** baseline.

---

## Results Summary

| Assignment | Task | Models | Baseline | Best Result | Takeaway |
|---|---|---|---|---|---|
| **1. Classification** | Binary boundary mapping | Logistic Regression, LinearSVC | 66.5% (majority class) | 96.25% test accuracy | Linear models are limited by the dome-shaped data; quadratic features remove the bottleneck. |
| **2. Regression** | 3D surface fitting | Lasso (L1), Ridge (L2) | Mean predictor | CV-optimised model | L1 performs feature selection through sparsity; L2 keeps all parameters and shrinks them smoothly. Extrapolation exposes over-fitting. |
| **3. Model selection** | Two-dataset benchmark | L2 Logistic Regression, kNN | D1: 67.5%<br>D2: 78.3% | D1: 67.5% (no signal)<br>D2: 96.4% (LogReg) | Dataset 1 shows no signal beyond the baseline. On Dataset 2, polynomial Logistic Regression generalises more smoothly than kNN. |

---

## Tech Stack

- **Language:** Python 3.x
- **Core libraries:** scikit-learn, pandas, NumPy
- **Visualisation:** Matplotlib (3D surfaces, contour decision boundaries, error bars)

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/uri3lrex/ML_assignments.git
   cd applied-ml-tcd
   ```

2. **Install dependencies**

   ```bash
   pip install numpy pandas notebook matplotlib scikit-learn
   ```

3. **Run the scripts or open the notebooks**

   In Assignment 3, switch datasets by changing the `dataset = 1` / `dataset = 2` flag near line 40 of the script.

---

## Acknowledgements

Completed as coursework for CS7CS4 Machine Learning, Trinity College Dublin.
