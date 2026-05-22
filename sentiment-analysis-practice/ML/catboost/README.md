# CatBoost

`catb.ipynb` обучает `CatBoostClassifier` на уже собранных признаках из `../data_with_all_features`.

## Входные данные

Нужны два CSV-файла:

- `../data_with_all_features/train_with_features.csv`
- `../data_with_all_features/test_with_features.csv`

В этих таблицах ожидаются колонки:

- `review`
- `target`
- `split`
- все числовые признаки модели

## Что делает ноутбук

1. Загружает train/test из CSV.
2. Подбирает гиперпараметры через `RandomizedSearchCV` на GPU.
3. Обучает финальную модель с `eval_set`.
4. Считает метрики:
   - accuracy
   - precision
   - recall
   - F1
   - ROC-AUC OvR
   - PR-AUC
5. Строит и сохраняет графики:
   - confusion matrix
   - ROC curves
   - PR curves
   - training history
   - feature importance
   - permutation importance
   - PDP + ICE
   - SHAP beeswarm / bar / waterfall / force
6. Сохраняет артефакты модели.

## Что сохраняется

Файлы сохраняются прямо в папку `ML/catboost`:

- `all_metrics.csv`
- `catboost_test_report.txt`
- `confusion_matrix.png`
- `roc_curves_ovr.png`
- `pr_curves_ovr.png`
- `training_history.png`
- `feature_importance.csv`
- `feature_importance_top25.png`
- `permutation_importance.csv`
- `permutation_importance_top20.png`
- `pdp_ice_*.png`
- `shap_beeswarm_class_*.png`
- `shap_bar_class_*.png`
- `shap_waterfall_class_*.png`
- `shap_force_class_*.html`
- `shap_importance.csv`

Артефакты модели сохраняются в `ML/catboost/artifacts`:

- `catboost_final_model.cbm`
- `feature_columns.joblib`
- `best_params.json`

## Отдельный SHAP

Если SHAP роняет Jupyter kernel, запускай его отдельно:

```bash
cd ML/catboost
python run_shap_analysis.py
```

Скрипт:

- использует `artifacts/catboost_final_model.cbm`, если он уже сохранён;
- если артефакта нет, заново обучает модель из `best_params.json`;
- сохраняет SHAP-файлы в `ML/catboost/shap_outputs`.

Можно уменьшить размер подвыборки:

```bash
python run_shap_analysis.py --sample-size 200
```

## Порядок запуска

Запускай ячейки сверху вниз. Если подбор гиперпараметров уже завершён и объект `random_search` есть в памяти, можно продолжить с ячейки `best_params = random_search.best_params_`.
