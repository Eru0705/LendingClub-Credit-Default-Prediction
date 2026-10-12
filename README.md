# Credit Default Prediction – LendingClub (R)

Predicting which LendingClub loans will default, with an honest evaluation that avoids data leakage.

> MSc Computer Science and Technology · Ulster University · team project (4 people, team lead: Eru Poudel)

**Notebook:** [`LendingClub.ipynb`](LendingClub.ipynb) (R kernel, runs in Google Colab)

---

## Key results

| | |
|---|---|
| Loans modelled | **38,577** (train 30,861 / test 7,716, 14.6% default) |
| Best model | **Stacking ensemble (LR + XGBoost + LightGBM), AUC 0.716** |
| Leakage finding | One post-approval field lifts AUC from **0.711 → 0.866**; all leakage fields → **0.999** |
| Business value | Reviewing the riskiest **20%** of loans catches **42.9%** of defaulters (**2.4× lift**) |
| Explainability | SHAP for XGBoost, LightGBM and the ensemble; standardised LR coefficients |

## Pipeline

1. **Cleaning:** dropped all-missing and single-value columns, free-text fields (`desc`, `title`, `emp_title`) and leakage columns. Every imputation has a stated reason; for example, a missing `mths_since_last_delinq` means "never delinquent", not missing data.
2. **Feature engineering:** derived risk and behaviour features, ordinal encoding of `sub_grade`, binary public-record flags, log transforms of skewed money columns.
3. **Statistics and EDA:** chi-square tests for categorical features, correlation heatmap, risk plots by sub-grade, purpose, term and state.
4. **Segmentation:** K-means (elbow and silhouette, k = 3) gives three borrower groups, visualised with PCA:

   | Cluster | Loans | Default rate | Avg interest rate |
   |---|---|---|---|
   | High risk | 11,570 | 22.6% | 15.7% |
   | Medium risk | 7,639 | 17.1% | 12.1% |
   | Low risk | 19,368 | 8.8% | 9.6% |

5. **Modelling:** stratified train/test split, scaling and one-hot encoding fitted on train only, class weights for imbalance, hyperparameter tuning with cross-validation, and F1-tuned decision thresholds.

## Model comparison (test set)

| Model | AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic regression (L2) | 0.713 | 0.624 | 0.251 | 0.358 |
| XGBoost | 0.713 | 0.449 | 0.303 | 0.362 |
| LightGBM | 0.712 | 0.496 | 0.284 | 0.361 |
| **Stacking (LR + XGB + LGB)** | **0.716** | 0.556 | 0.269 | **0.362** |

SMOTE oversampling was also tested and did **not** beat class weighting (AUC 0.699–0.712).

## Effect of data leakage

Many published LendingClub models report 90%+ scores. This experiment shows why:

| Features | AUC | F1 |
|---|---|---|
| Honest (no leakage) | 0.711 | 0.365 |
| + 1 leakage column | 0.866 | 0.522 |
| + 2 leakage columns | 0.912 | 0.641 |
| + all leakage columns | **0.999** | 0.997 |

Post-approval fields such as payments received describe what happened *after* the loan was issued, so they cannot be used to predict default at application time.

## Lift

| Top % reviewed | Defaulters caught | Lift |
|---|---|---|
| 10% | 26.0% | 2.8× |
| 20% | **42.9%** | **2.4×** |
| 30% | 55.6% | 2.0× |

## Extras

- **Forecasting:** monthly loan volume, with candidate models compared on a 12-month backtest. Auto ARIMA was selected (**MAPE 9.4%**) and used for a 12-month forecast.
- **Error analysis:** fully paid loans the model flagged as risky ("survivors") compared with actual defaults using Wilcoxon tests.

## Tech

R · tidyverse/dplyr · glmnet · xgboost · lightgbm · SHAP · factoextra (K-means) · pROC · smotefamily · forecast

## Data

LendingClub accepted-loans data, 2007–2011 (39,717 rows before cleaning). The notebook downloads `loan.csv` from Google Drive with `gdown`; the CSV itself is not stored in this repo.

## Author

**Eru Poudel** ([@Eru0705](https://github.com/Eru0705)), MSc Computer Science and Technology, Ulster University
