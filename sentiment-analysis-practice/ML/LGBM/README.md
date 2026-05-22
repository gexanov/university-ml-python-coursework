# LGBM

`lgbm.ipynb` trains `LGBM` on the prepared feature tables from `../data_with_all_features`.

## Inputs

Required files:

- `../data_with_all_features/train_with_features.csv`
- `../data_with_all_features/test_with_features.csv`

## Resource limits

- CPU usage is limited inside the notebook by `CPU_THREADS = floor(os.cpu_count() * 0.30)`.
- `RandomizedSearchCV` uses `n_jobs=1` so it does not multiply CPU load across folds.
- GPU is disabled by default with `USE_GPU = False`.
- If you enable GPU manually, there is no reliable hard 30% GPU cap from these libraries, so keep it off if you need strict limits.

## Outputs

The notebook saves:

- `all_metrics.csv`
- `model_test_report.txt`
- `confusion_matrix.png`
- `roc_curves_ovr.png`
- `pr_curves_ovr.png`
- `training_history.png`
- `feature_importance.csv`
- `feature_importance_top25.png`
- `permutation_importance.csv`
- `permutation_importance_top20.png`
- `pdp_ice_*.png`

Artifacts are saved in `artifacts/`:

- `lgbm_final_model.joblib`
- `feature_columns.joblib`
- `best_params.json`

## SHAP

Run SHAP outside Jupyter:

```bash
cd ML/LGBM
python run_shap_analysis.py
```

Outputs go to `shap_outputs/`.
