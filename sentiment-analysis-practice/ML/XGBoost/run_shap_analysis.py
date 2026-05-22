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
    raw_y_train = train_df['target']
    X_test = test_df.drop(columns=['review', 'target', 'split'])
    label_values = sorted(pd.Series(raw_y_train).unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    y_train = raw_y_train.map(label_to_index)
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

    mapping_path = ARTIFACTS_DIR / 'label_mapping.json'
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
        index_to_label = {int(k): v for k, v in mapping.get('index_to_label', {}).items()}
    else:
        index_to_label = {int(idx): int(idx) for idx in model.classes_}

    classes_ = [index_to_label[int(idx)] for idx in model.classes_]
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
