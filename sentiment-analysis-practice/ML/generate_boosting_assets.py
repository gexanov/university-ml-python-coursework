from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def to_source(text: str) -> list[str]:
    cleaned = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in cleaned.splitlines()]


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": to_source(text),
    }


def notebook_dict(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "base",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_LOAD_CELL = """
from pathlib import Path
import math
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path('../data_with_all_features')
OUT_DIR = Path('.')
ARTIFACTS_DIR = OUT_DIR / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CPU_FRACTION = 0.30
CPU_THREADS = max(1, math.floor((os.cpu_count() or 1) * CPU_FRACTION))
USE_GPU = False

train_df = pd.read_csv(DATA_DIR / 'train_with_features.csv')
test_df = pd.read_csv(DATA_DIR / 'test_with_features.csv')

X_train = train_df.drop(columns=['review', 'target', 'split'])
y_train = train_df['target']
X_test = test_df.drop(columns=['review', 'target', 'split'])
y_test = test_df['target']

print('train:', X_train.shape, 'test:', X_test.shape)
print('classes:', sorted(y_train.unique().tolist()))
print('cpu_threads:', CPU_THREADS, 'use_gpu:', USE_GPU)
"""


COMMON_SPLIT_CELL = """
from sklearn.model_selection import train_test_split

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print(X_train_sub.shape, X_val.shape, X_test.shape)
"""


COMMON_METRICS_CELL = """
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

classes_ = np.array(sorted(pd.Series(y_train).unique()))
y_pred = final_model.predict(X_test)
if hasattr(y_pred, 'reshape'):
    y_pred = y_pred.reshape(-1)
y_proba = final_model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=classes_)

metrics_dict = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    'roc_auc_ovr_macro': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'),
    'roc_auc_ovr_weighted': roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'),
    'pr_auc_macro': average_precision_score(y_test_bin, y_proba, average='macro'),
    'pr_auc_weighted': average_precision_score(y_test_bin, y_proba, average='weighted'),
}

metrics_df = pd.DataFrame(metrics_dict, index=['score']).T
report_text = classification_report(y_test, y_pred, zero_division=0)

print(report_text)
metrics_df

metrics_df.to_csv(OUT_DIR / 'all_metrics.csv', encoding='utf-8-sig')
with open(OUT_DIR / 'model_test_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
    f.write('\\n\\n')
    f.write(metrics_df.to_string())
"""


COMMON_CURVES_CELL = """
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

cm = confusion_matrix(y_test, y_pred, labels=classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_, yticklabels=classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
for i, class_label in enumerate(classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'class {class_label} (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (OvR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / 'roc_curves_ovr.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
for i, class_label in enumerate(classes_):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'class {class_label} (AUC={pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (OvR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / 'pr_curves_ovr.png', dpi=300, bbox_inches='tight')
plt.close()
"""


COMMON_PERM_CELL = """
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_,
}).sort_values('importance', ascending=False)
feature_importance.to_csv(OUT_DIR / 'feature_importance.csv', index=False, encoding='utf-8-sig')

top_k = 25
top_features = feature_importance.head(top_k)
plt.figure(figsize=(10, 8))
sns.barplot(data=top_features, x='importance', y='feature')
plt.title(f'Top {top_k} Feature Importances')
plt.tight_layout()
plt.savefig(OUT_DIR / 'feature_importance_top25.png', dpi=300, bbox_inches='tight')
plt.close()

f1_macro_scorer = make_scorer(f1_score, average='macro')
X_perm = X_val.sample(min(3000, len(X_val)), random_state=42)
y_perm = y_val.loc[X_perm.index]
baseline_perm_pred = final_model.predict(X_perm)
if hasattr(baseline_perm_pred, 'reshape'):
    baseline_perm_pred = baseline_perm_pred.reshape(-1)
baseline_macro_f1 = f1_score(y_perm, baseline_perm_pred, average='macro')
perm_result = permutation_importance(
    final_model,
    X_perm,
    y_perm,
    scoring=f1_macro_scorer,
    n_repeats=5,
    random_state=42,
    n_jobs=1,
)

perm_importance_df = pd.DataFrame({
    'feature': X_perm.columns,
    'baseline_macro_f1': baseline_macro_f1,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std,
})
perm_importance_df['score_after_permutation_mean'] = (
    perm_importance_df['baseline_macro_f1'] - perm_importance_df['importance_mean']
)

random_baseline_row = pd.DataFrame([
    {
        'feature': 'random_baseline_feature',
        'baseline_macro_f1': baseline_macro_f1,
        'importance_mean': 0.0,
        'importance_std': 0.0,
        'score_after_permutation_mean': baseline_macro_f1,
    }
])

perm_importance_df = pd.concat([perm_importance_df, random_baseline_row], ignore_index=True)
perm_importance_df = perm_importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
perm_importance_df.to_csv(OUT_DIR / 'permutation_importance.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(10, 8))
sns.barplot(data=perm_importance_df.head(20), x='importance_mean', y='feature')
plt.title('Permutation Importance (Macro F1 drop)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'permutation_importance_top20.png', dpi=300, bbox_inches='tight')
plt.close()

perm_importance_df
"""


COMMON_PDP_CELL = """
from sklearn.inspection import PartialDependenceDisplay

pdp_classes = list(final_model.classes_)
interpretable_features = [
    col for col in feature_importance['feature'].tolist()
]
pdp_features = interpretable_features[:6]
X_pdp = X_val.sample(min(3000, len(X_val)), random_state=42)

for feat in pdp_features:
    fig, axes = plt.subplots(1, len(pdp_classes), figsize=(6 * len(pdp_classes), 5), squeeze=False)
    axes = axes.ravel()
    for ax, class_label in zip(axes, pdp_classes):
        PartialDependenceDisplay.from_estimator(
            final_model,
            X_pdp,
            features=[feat],
            target=class_label,
            kind='both',
            subsample=200,
            random_state=42,
            grid_resolution=30,
            ax=ax,
        )
        ax.set_title(f'PDP + ICE: {feat} | class={class_label}')
    safe_feat = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in feat)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'pdp_ice_{safe_feat}.png', dpi=300, bbox_inches='tight')
    plt.close()
"""


COMMON_SHAP_NOTE_CELL = """
# Run SHAP outside Jupyter to avoid kernel crashes:
# !python run_shap_analysis.py
"""


def lgbm_cells() -> list[dict]:
    return [
        markdown_cell("# LGBM"),
        code_cell(COMMON_LOAD_CELL),
        code_cell(
            """
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.metrics import make_scorer, f1_score

            f1_macro_scorer = make_scorer(f1_score, average='macro')

            base_params = {
                'objective': 'multiclass',
                'random_state': 42,
                'n_jobs': CPU_THREADS,
                'verbosity': -1,
                'device_type': 'gpu' if USE_GPU else 'cpu',
                'force_col_wise': True,
            }

            param_dist = {
                'n_estimators': [300, 500, 700, 900],
                'learning_rate': [0.03, 0.05, 0.07, 0.1],
                'num_leaves': [31, 63, 127],
                'max_depth': [-1, 6, 8, 10],
                'min_child_samples': [20, 40, 80],
                'subsample': [0.7, 0.85, 1.0],
                'colsample_bytree': [0.7, 0.85, 1.0],
                'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            }

            random_search = RandomizedSearchCV(
                estimator=LGBMClassifier(**base_params),
                param_distributions=param_dist,
                n_iter=20,
                scoring=f1_macro_scorer,
                cv=3,
                verbose=2,
                random_state=42,
                n_jobs=1,
                refit=True,
            )

            random_search.fit(X_train, y_train)

            print('Best params:', random_search.best_params_)
            print('Best CV macro F1:', random_search.best_score_)
            """
        ),
        code_cell(COMMON_SPLIT_CELL),
        code_cell(
            """
            best_params = random_search.best_params_
            best_params
            """
        ),
        code_cell(
            """
            import lightgbm as lgb
            from lightgbm import LGBMClassifier

            final_params = {
                **best_params,
                'objective': 'multiclass',
                'random_state': 42,
                'n_jobs': CPU_THREADS,
                'verbosity': -1,
                'device_type': 'gpu' if USE_GPU else 'cpu',
                'force_col_wise': True,
            }

            final_model = LGBMClassifier(**final_params)
            final_model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
                eval_names=['train', 'valid'],
                eval_metric='multi_logloss',
                callbacks=[lgb.log_evaluation(period=100)],
            )
            """
        ),
        code_cell(COMMON_METRICS_CELL),
        code_cell(COMMON_CURVES_CELL),
        code_cell(COMMON_PERM_CELL),
        code_cell(
            """
            evals_result = final_model.evals_result_

            plt.figure(figsize=(10, 5))
            for dataset_name, metrics in evals_result.items():
                for metric_name, values in metrics.items():
                    plt.plot(values, label=f'{dataset_name}_{metric_name}')

            plt.title('Training History')
            plt.xlabel('Iteration')
            plt.ylabel('Metric')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(OUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            """
        ),
        code_cell(COMMON_PDP_CELL),
        code_cell(
            """
            model_path = ARTIFACTS_DIR / 'lgbm_final_model.joblib'
            params_path = ARTIFACTS_DIR / 'best_params.json'
            columns_path = ARTIFACTS_DIR / 'feature_columns.joblib'

            joblib.dump(final_model, model_path)
            joblib.dump(list(X_train.columns), columns_path)
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)

            print('Artifacts saved to:', ARTIFACTS_DIR.resolve())
            """
        ),
        code_cell(COMMON_SHAP_NOTE_CELL),
    ]


def xgb_cells() -> list[dict]:
    return [
        markdown_cell("# XGBoost"),
        code_cell(COMMON_LOAD_CELL),
        code_cell(
            """
            from xgboost import XGBClassifier
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.metrics import make_scorer, f1_score

            f1_macro_scorer = make_scorer(f1_score, average='macro')
            num_classes = len(np.unique(y_train))

            base_params = {
                'objective': 'multi:softprob',
                'num_class': num_classes,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': CPU_THREADS,
                'tree_method': 'hist',
                'device': 'cuda' if USE_GPU else 'cpu',
            }

            param_dist = {
                'n_estimators': [300, 500, 700, 900],
                'learning_rate': [0.03, 0.05, 0.07, 0.1],
                'max_depth': [4, 6, 8, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.85, 1.0],
                'colsample_bytree': [0.7, 0.85, 1.0],
                'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 2.0, 5.0],
            }

            random_search = RandomizedSearchCV(
                estimator=XGBClassifier(**base_params),
                param_distributions=param_dist,
                n_iter=20,
                scoring=f1_macro_scorer,
                cv=3,
                verbose=2,
                random_state=42,
                n_jobs=1,
                refit=True,
            )

            random_search.fit(X_train, y_train)

            print('Best params:', random_search.best_params_)
            print('Best CV macro F1:', random_search.best_score_)
            """
        ),
        code_cell(COMMON_SPLIT_CELL),
        code_cell(
            """
            best_params = random_search.best_params_
            best_params
            """
        ),
        code_cell(
            """
            from xgboost import XGBClassifier

            num_classes = len(np.unique(y_train))
            final_params = {
                **best_params,
                'objective': 'multi:softprob',
                'num_class': num_classes,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': CPU_THREADS,
                'tree_method': 'hist',
                'device': 'cuda' if USE_GPU else 'cpu',
            }

            final_model = XGBClassifier(**final_params)
            final_model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
                verbose=False,
            )
            """
        ),
        code_cell(COMMON_METRICS_CELL),
        code_cell(COMMON_CURVES_CELL),
        code_cell(COMMON_PERM_CELL),
        code_cell(
            """
            evals_result = final_model.evals_result()

            plt.figure(figsize=(10, 5))
            for dataset_name, metrics in evals_result.items():
                for metric_name, values in metrics.items():
                    plt.plot(values, label=f'{dataset_name}_{metric_name}')

            plt.title('Training History')
            plt.xlabel('Iteration')
            plt.ylabel('Metric')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(OUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            """
        ),
        code_cell(COMMON_PDP_CELL),
        code_cell(
            """
            model_path = ARTIFACTS_DIR / 'xgboost_final_model.joblib'
            params_path = ARTIFACTS_DIR / 'best_params.json'
            columns_path = ARTIFACTS_DIR / 'feature_columns.joblib'

            joblib.dump(final_model, model_path)
            joblib.dump(list(X_train.columns), columns_path)
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)

            print('Artifacts saved to:', ARTIFACTS_DIR.resolve())
            """
        ),
        code_cell(COMMON_SHAP_NOTE_CELL),
    ]


LGBM_SHAP_SCRIPT = """
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / 'data_with_all_features'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
OUT_DIR = ROOT_DIR / 'shap_outputs'
CPU_THREADS = max(1, math.floor((os.cpu_count() or 1) * 0.30))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--max-display', type=int, default=20)
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def load_tables():
    train_df = pd.read_csv(DATA_DIR / 'train_with_features.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_with_features.csv')
    X_train = train_df.drop(columns=['review', 'target', 'split'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['review', 'target', 'split'])
    return X_train, y_train, X_test


def train_model(X_train, y_train):
    params = json.loads((ARTIFACTS_DIR / 'best_params.json').read_text(encoding='utf-8'))
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    model = LGBMClassifier(
        **params,
        objective='multiclass',
        random_state=42,
        n_jobs=CPU_THREADS,
        verbosity=-1,
        device_type='cpu',
        force_col_wise=True,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=['train', 'valid'],
        eval_metric='multi_logloss',
        callbacks=[lgb.log_evaluation(period=100)],
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / 'lgbm_final_model.joblib')
    return model


def load_model(X_train, y_train):
    model_path = ARTIFACTS_DIR / 'lgbm_final_model.joblib'
    if model_path.exists():
        return joblib.load(model_path), 'loaded'
    return train_model(X_train, y_train), 'trained'


def normalize_shap_values(shap_values, n_classes):
    if isinstance(shap_values, list):
        return [np.asarray(values) for values in shap_values]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[2] == n_classes:
            return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        if shap_values.shape[1] == n_classes:
            return [shap_values[:, i, :] for i in range(shap_values.shape[1])]
    return [np.asarray(shap_values)]


def normalize_expected_values(expected_values, n_classes):
    if isinstance(expected_values, (list, tuple)):
        return [float(x) for x in expected_values]
    if isinstance(expected_values, np.ndarray):
        if expected_values.ndim == 1 and len(expected_values) == n_classes:
            return [float(x) for x in expected_values]
        if expected_values.ndim == 0:
            return [float(expected_values.item()) for _ in range(n_classes)]
    return [float(expected_values) for _ in range(n_classes)]


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test = load_tables()
    model, model_source = load_model(X_train, y_train)
    sample_size = min(args.sample_size, len(X_test))
    X_shap = X_test.sample(sample_size, random_state=args.random_state).copy()

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    classes_ = list(model.classes_)
    class_shap_values = normalize_shap_values(shap_values, len(classes_))
    base_values = normalize_expected_values(explainer.expected_value, len(classes_))

    for class_label, shap_vals in zip(classes_, class_shap_values):
        safe_label = str(class_label).replace('-', 'minus').replace('.', '_')
        shap.summary_plot(shap_vals, X_shap, show=False, max_display=args.max_display)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_beeswarm_class_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()

        shap.summary_plot(shap_vals, X_shap, plot_type='bar', show=False, max_display=args.max_display)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_bar_class_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()

    sample_idx = 0
    for class_label, shap_vals, base_value in zip(classes_, class_shap_values, base_values):
        safe_label = str(class_label).replace('-', 'minus').replace('.', '_')
        explanation = shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=base_value,
            data=X_shap.iloc[sample_idx].values,
            feature_names=X_shap.columns.tolist(),
        )
        shap.plots.waterfall(explanation, max_display=args.max_display, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_waterfall_class_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        force_plot = shap.force_plot(base_value, shap_vals[sample_idx], X_shap.iloc[sample_idx], matplotlib=False)
        shap.save_html(str(OUT_DIR / f'shap_force_class_{safe_label}.html'), force_plot)

    mean_abs_shap = np.mean(np.stack([np.abs(values) for values in class_shap_values], axis=0), axis=(0, 1))
    shap_importance = pd.DataFrame({'feature': X_shap.columns, 'mean_abs_shap': mean_abs_shap}).sort_values(
        'mean_abs_shap', ascending=False
    )
    shap_importance.to_csv(OUT_DIR / 'shap_importance.csv', index=False, encoding='utf-8-sig')

    metadata = {
        'model_source': model_source,
        'sample_size': sample_size,
        'classes': [str(x) for x in classes_],
        'output_dir': str(OUT_DIR.resolve()),
    }
    (OUT_DIR / 'shap_run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'SHAP outputs saved to: {OUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
"""


XGB_SHAP_SCRIPT = """
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / 'data_with_all_features'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
OUT_DIR = ROOT_DIR / 'shap_outputs'
CPU_THREADS = max(1, math.floor((os.cpu_count() or 1) * 0.30))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--max-display', type=int, default=20)
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def load_tables():
    train_df = pd.read_csv(DATA_DIR / 'train_with_features.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_with_features.csv')
    X_train = train_df.drop(columns=['review', 'target', 'split'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['review', 'target', 'split'])
    return X_train, y_train, X_test


def train_model(X_train, y_train):
    params = json.loads((ARTIFACTS_DIR / 'best_params.json').read_text(encoding='utf-8'))
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    model = XGBClassifier(
        **params,
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=CPU_THREADS,
        tree_method='hist',
        device='cpu',
    )
    model.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / 'xgboost_final_model.joblib')
    return model


def load_model(X_train, y_train):
    model_path = ARTIFACTS_DIR / 'xgboost_final_model.joblib'
    if model_path.exists():
        return joblib.load(model_path), 'loaded'
    return train_model(X_train, y_train), 'trained'


def normalize_shap_values(shap_values, n_classes):
    if isinstance(shap_values, list):
        return [np.asarray(values) for values in shap_values]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[2] == n_classes:
            return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        if shap_values.shape[1] == n_classes:
            return [shap_values[:, i, :] for i in range(shap_values.shape[1])]
    return [np.asarray(shap_values)]


def normalize_expected_values(expected_values, n_classes):
    if isinstance(expected_values, (list, tuple)):
        return [float(x) for x in expected_values]
    if isinstance(expected_values, np.ndarray):
        if expected_values.ndim == 1 and len(expected_values) == n_classes:
            return [float(x) for x in expected_values]
        if expected_values.ndim == 0:
            return [float(expected_values.item()) for _ in range(n_classes)]
    return [float(expected_values) for _ in range(n_classes)]


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test = load_tables()
    model, model_source = load_model(X_train, y_train)
    sample_size = min(args.sample_size, len(X_test))
    X_shap = X_test.sample(sample_size, random_state=args.random_state).copy()

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    classes_ = list(model.classes_)
    class_shap_values = normalize_shap_values(shap_values, len(classes_))
    base_values = normalize_expected_values(explainer.expected_value, len(classes_))

    for class_label, shap_vals in zip(classes_, class_shap_values):
        safe_label = str(class_label).replace('-', 'minus').replace('.', '_')
        shap.summary_plot(shap_vals, X_shap, show=False, max_display=args.max_display)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_beeswarm_class_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()

        shap.summary_plot(shap_vals, X_shap, plot_type='bar', show=False, max_display=args.max_display)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_bar_class_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()

    sample_idx = 0
    for class_label, shap_vals, base_value in zip(classes_, class_shap_values, base_values):
        safe_label = str(class_label).replace('-', 'minus').replace('.', '_')
        explanation = shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=base_value,
            data=X_shap.iloc[sample_idx].values,
            feature_names=X_shap.columns.tolist(),
        )
        shap.plots.waterfall(explanation, max_display=args.max_display, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'shap_waterfall_class_{safe_label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        force_plot = shap.force_plot(base_value, shap_vals[sample_idx], X_shap.iloc[sample_idx], matplotlib=False)
        shap.save_html(str(OUT_DIR / f'shap_force_class_{safe_label}.html'), force_plot)

    mean_abs_shap = np.mean(np.stack([np.abs(values) for values in class_shap_values], axis=0), axis=(0, 1))
    shap_importance = pd.DataFrame({'feature': X_shap.columns, 'mean_abs_shap': mean_abs_shap}).sort_values(
        'mean_abs_shap', ascending=False
    )
    shap_importance.to_csv(OUT_DIR / 'shap_importance.csv', index=False, encoding='utf-8-sig')

    metadata = {
        'model_source': model_source,
        'sample_size': sample_size,
        'classes': [str(x) for x in classes_],
        'output_dir': str(OUT_DIR.resolve()),
    }
    (OUT_DIR / 'shap_run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'SHAP outputs saved to: {OUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
"""


def build_readme(model_name: str, notebook_name: str, model_file: str) -> str:
    return textwrap.dedent(
        f"""
        # {model_name}

        `{notebook_name}` trains `{model_name}` on the prepared feature tables from `../data_with_all_features`.

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

        - `{model_file}`
        - `feature_columns.joblib`
        - `best_params.json`

        ## SHAP

        Run SHAP outside Jupyter:

        ```bash
        cd ML/{model_name}
        python run_shap_analysis.py
        ```

        Outputs go to `shap_outputs/`.
        """
    ).strip() + "\n"


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.write_text(json.dumps(notebook_dict(cells), ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    lgbm_dir = ROOT / "LGBM"
    xgb_dir = ROOT / "XGBoost"
    lgbm_dir.mkdir(parents=True, exist_ok=True)
    xgb_dir.mkdir(parents=True, exist_ok=True)

    write_notebook(lgbm_dir / "lgbm.ipynb", lgbm_cells())
    write_notebook(xgb_dir / "xgboost.ipynb", xgb_cells())

    (lgbm_dir / "run_shap_analysis.py").write_text(textwrap.dedent(LGBM_SHAP_SCRIPT).strip() + "\n", encoding="utf-8")
    (xgb_dir / "run_shap_analysis.py").write_text(textwrap.dedent(XGB_SHAP_SCRIPT).strip() + "\n", encoding="utf-8")

    (lgbm_dir / "README.md").write_text(build_readme("LGBM", "lgbm.ipynb", "lgbm_final_model.joblib"), encoding="utf-8")
    (xgb_dir / "README.md").write_text(build_readme("XGBoost", "xgboost.ipynb", "xgboost_final_model.joblib"), encoding="utf-8")

    print("Generated LGBM and XGBoost assets.")


if __name__ == "__main__":
    main()
