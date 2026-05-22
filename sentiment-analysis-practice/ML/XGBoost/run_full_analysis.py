from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data_with_all_features"
OUT_DIR = ROOT_DIR
ARTIFACTS_DIR = OUT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CPU_THREADS = max(1, (os.cpu_count() or 1))
USE_GPU = True
SEARCH_ITERATIONS = 20
SEARCH_CV = 3


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_with_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_with_features.csv")
    X_train = train_df.drop(columns=["review", "target", "split"])
    raw_y_train = train_df["target"]
    X_test = test_df.drop(columns=["review", "target", "split"])
    raw_y_test = test_df["target"]

    label_values = sorted(pd.Series(raw_y_train).unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    y_train = raw_y_train.map(label_to_index)
    y_test = raw_y_test.map(label_to_index)
    return X_train, y_train, X_test, y_test, label_to_index, index_to_label


def run_search(X_train, y_train):
    f1_macro_scorer = make_scorer(f1_score, average="macro")
    num_classes = len(np.unique(y_train))
    base_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": CPU_THREADS,
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
    }
    param_dist = {
        "n_estimators": [300, 500, 700, 900],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(**base_params),
        param_distributions=param_dist,
        n_iter=SEARCH_ITERATIONS,
        scoring=f1_macro_scorer,
        cv=SEARCH_CV,
        verbose=2,
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    try:
        random_search.fit(X_train, y_train)
    except Exception:
        base_params["device"] = "cpu"
        random_search = RandomizedSearchCV(
            estimator=XGBClassifier(**base_params),
            param_distributions=param_dist,
            n_iter=SEARCH_ITERATIONS,
            scoring=f1_macro_scorer,
            cv=SEARCH_CV,
            verbose=2,
            random_state=42,
            n_jobs=1,
            refit=True,
        )
        random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    return best_params, random_search.best_score_


def fit_final_model(X_train, y_train, best_params):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train,
    )
    final_params = {
        **best_params,
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_train)),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": CPU_THREADS,
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
    }
    final_model = XGBClassifier(**final_params)
    try:
        final_model.fit(
            X_train_sub,
            y_train_sub,
            eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
            verbose=False,
        )
    except Exception:
        final_params["device"] = "cpu"
        final_model = XGBClassifier(**final_params)
        final_model.fit(
            X_train_sub,
            y_train_sub,
            eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
            verbose=False,
        )
    joblib.dump(final_model, ARTIFACTS_DIR / "xgboost_final_model.joblib")
    joblib.dump(list(X_train.columns), ARTIFACTS_DIR / "feature_columns.joblib")
    return final_model, X_train_sub, X_val, y_train_sub, y_val


def save_metrics_and_curves(final_model, X_test, y_test, y_train, index_to_label):
    classes_ = np.array(sorted(pd.Series(y_train).unique()))
    display_labels = [index_to_label[int(idx)] for idx in classes_]
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=classes_)

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        "roc_auc_ovr_weighted": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
        "pr_auc_macro": average_precision_score(y_test_bin, y_proba, average="macro"),
        "pr_auc_weighted": average_precision_score(y_test_bin, y_proba, average="weighted"),
    }
    metrics_df = pd.DataFrame(metrics_dict, index=["score"]).T
    report_text = classification_report(y_test, y_pred, zero_division=0)
    metrics_df.to_csv(OUT_DIR / "all_metrics.csv", encoding="utf-8-sig")
    with open(OUT_DIR / "model_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n\n")
        f.write(metrics_df.to_string())

    cm = confusion_matrix(y_test, y_pred, labels=classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=display_labels, yticklabels=display_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"class {index_to_label[int(class_label)]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (OvR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves_ovr.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(classes_):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"class {index_to_label[int(class_label)]} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (OvR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves_ovr.png", dpi=300, bbox_inches="tight")
    plt.close()

    return metrics_df


def save_importances_and_pdp(final_model, X_train, X_val, y_val, index_to_label):
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(25), x="importance", y="feature")
    plt.title("Top 25 Feature Importances")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_top25.png", dpi=300, bbox_inches="tight")
    plt.close()

    f1_macro_scorer = make_scorer(f1_score, average="macro")
    X_perm = X_val.sample(min(3000, len(X_val)), random_state=42)
    y_perm = y_val.loc[X_perm.index]
    baseline_perm_pred = final_model.predict(X_perm)
    baseline_macro_f1 = f1_score(y_perm, baseline_perm_pred, average="macro")
    perm_result = permutation_importance(
        final_model,
        X_perm,
        y_perm,
        scoring=f1_macro_scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )

    perm_importance_df = pd.DataFrame(
        {
            "feature": X_perm.columns,
            "baseline_macro_f1": baseline_macro_f1,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }
    )
    perm_importance_df["score_after_permutation_mean"] = (
        perm_importance_df["baseline_macro_f1"] - perm_importance_df["importance_mean"]
    )
    random_baseline_row = pd.DataFrame(
        [
            {
                "feature": "random_baseline_feature",
                "baseline_macro_f1": baseline_macro_f1,
                "importance_mean": 0.0,
                "importance_std": 0.0,
                "score_after_permutation_mean": baseline_macro_f1,
            }
        ]
    )
    perm_importance_df = pd.concat([perm_importance_df, random_baseline_row], ignore_index=True)
    perm_importance_df = perm_importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    perm_importance_df.to_csv(OUT_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    sns.barplot(data=perm_importance_df.head(20), x="importance_mean", y="feature")
    plt.title("Permutation Importance (Macro F1 drop)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "permutation_importance_top20.png", dpi=300, bbox_inches="tight")
    plt.close()

    pdp_classes = list(final_model.classes_)
    pdp_features = feature_importance["feature"].head(6).tolist()
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
                kind="both",
                subsample=200,
                random_state=42,
                grid_resolution=30,
                ax=ax,
            )
            ax.set_title(f"PDP + ICE: {feat} | class={index_to_label[int(class_label)]}")
        safe_feat = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in feat)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pdp_ice_{safe_feat}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_training_history(final_model):
    evals_result = final_model.evals_result()
    plt.figure(figsize=(10, 5))
    for dataset_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            plt.plot(values, label=f"{dataset_name}_{metric_name}")
    plt.title("Training History")
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Running full XGBoost analysis")
    print(
        "CPU_THREADS =",
        CPU_THREADS,
        "USE_GPU =",
        USE_GPU,
        "SEARCH_ITERATIONS =",
        SEARCH_ITERATIONS,
        "SEARCH_CV =",
        SEARCH_CV,
    )
    X_train, y_train, X_test, y_test, label_to_index, index_to_label = load_data()
    best_params, best_cv = run_search(X_train, y_train)
    print("best_cv_macro_f1 =", best_cv)
    final_model, X_train_sub, X_val, y_train_sub, y_val = fit_final_model(X_train, y_train, best_params)
    with open(ARTIFACTS_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_index": label_to_index, "index_to_label": index_to_label}, f, ensure_ascii=False, indent=2)
    save_metrics_and_curves(final_model, X_test, y_test, y_train, index_to_label)
    save_importances_and_pdp(final_model, X_train, X_val, y_val, index_to_label)
    save_training_history(final_model)
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
