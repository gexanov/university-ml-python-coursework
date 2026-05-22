from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data_with_all_features"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEFAULT_OUT_DIR = ROOT_DIR / "shap_outputs"
TARGET_COLUMNS = ["review", "target", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis in a separate process.")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of rows to use for SHAP summary plots.")
    parser.add_argument("--max-display", type=int, default=20, help="Max number of features on SHAP plots.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for SHAP outputs.")
    return parser.parse_args()


def load_feature_tables() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(DATA_DIR / "train_with_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_with_features.csv")

    X_train = train_df.drop(columns=TARGET_COLUMNS)
    y_train = train_df["target"]
    X_test = test_df.drop(columns=TARGET_COLUMNS)
    y_test = test_df["target"]
    return X_train, y_train, X_test, y_test


def train_model_from_best_params(X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostClassifier:
    best_params_path = ROOT_DIR / "best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"Could not find {best_params_path}. Run the CatBoost notebook first so best params exist."
        )

    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train,
    )

    model = CatBoostClassifier(
        **best_params,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=100,
        use_best_model=True,
    )

    try:
        model.fit(X_fit, y_fit, eval_set=(X_val, y_val))
    except Exception:
        # Fallback in case GPU is unavailable in this standalone process.
        model = CatBoostClassifier(
            **best_params,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            task_type="CPU",
            random_seed=42,
            verbose=100,
            use_best_model=True,
        )
        model.fit(X_fit, y_fit, eval_set=(X_val, y_val))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(ARTIFACTS_DIR / "catboost_final_model.cbm")
    return model


def load_or_train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[CatBoostClassifier, str]:
    model_path = ARTIFACTS_DIR / "catboost_final_model.cbm"
    if model_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model, "loaded"

    model = train_model_from_best_params(X_train, y_train)
    return model, "trained"


def normalize_shap_values(shap_values: object, n_classes: int) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(values) for values in shap_values]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[2] == n_classes:
            return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        if shap_values.shape[1] == n_classes:
            return [shap_values[:, i, :] for i in range(shap_values.shape[1])]
    return [np.asarray(shap_values)]


def normalize_expected_values(expected_values: object, n_classes: int) -> list[float]:
    if isinstance(expected_values, (list, tuple)):
        return [float(x) for x in expected_values]
    if isinstance(expected_values, np.ndarray):
        if expected_values.ndim == 1 and len(expected_values) == n_classes:
            return [float(x) for x in expected_values]
        if expected_values.ndim == 0:
            return [float(expected_values.item()) for _ in range(n_classes)]
    return [float(expected_values) for _ in range(n_classes)]


def save_summary_plots(
    X_shap: pd.DataFrame,
    classes_: list,
    class_shap_values: list[np.ndarray],
    out_dir: Path,
    max_display: int,
) -> None:
    for class_label, shap_vals in zip(classes_, class_shap_values):
        safe_label = str(class_label).replace("-", "minus").replace(".", "_")

        shap.summary_plot(shap_vals, X_shap, show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_beeswarm_class_{safe_label}.png", dpi=300, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_vals, X_shap, plot_type="bar", show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_bar_class_{safe_label}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_local_plots(
    X_shap: pd.DataFrame,
    classes_: list,
    class_shap_values: list[np.ndarray],
    base_values: list[float],
    out_dir: Path,
    max_display: int,
) -> None:
    sample_idx = 0

    for class_label, shap_vals, base_value in zip(classes_, class_shap_values, base_values):
        safe_label = str(class_label).replace("-", "minus").replace(".", "_")

        explanation = shap.Explanation(
            values=shap_vals[sample_idx],
            base_values=base_value,
            data=X_shap.iloc[sample_idx].values,
            feature_names=X_shap.columns.tolist(),
        )

        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_waterfall_class_{safe_label}.png", dpi=300, bbox_inches="tight")
        plt.close()

        force_plot = shap.force_plot(base_value, shap_vals[sample_idx], X_shap.iloc[sample_idx], matplotlib=False)
        shap.save_html(str(out_dir / f"shap_force_class_{safe_label}.html"), force_plot)


def save_importance_table(
    X_shap: pd.DataFrame,
    class_shap_values: list[np.ndarray],
    out_dir: Path,
) -> None:
    mean_abs_shap = np.mean(
        np.stack([np.abs(values) for values in class_shap_values], axis=0),
        axis=(0, 1),
    )

    shap_importance = pd.DataFrame(
        {
            "feature": X_shap.columns,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(out_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, _ = load_feature_tables()
    model, model_source = load_or_train_model(X_train, y_train)

    sample_size = min(args.sample_size, len(X_test))
    X_shap = X_test.sample(sample_size, random_state=args.random_state).copy()

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    classes_ = list(model.classes_)
    class_shap_values = normalize_shap_values(shap_values, len(classes_))
    base_values = normalize_expected_values(explainer.expected_value, len(classes_))

    save_summary_plots(X_shap, classes_, class_shap_values, args.out_dir, args.max_display)
    save_local_plots(X_shap, classes_, class_shap_values, base_values, args.out_dir, args.max_display)
    save_importance_table(X_shap, class_shap_values, args.out_dir)

    metadata = {
        "model_source": model_source,
        "sample_size": sample_size,
        "classes": [str(x) for x in classes_],
        "output_dir": str(args.out_dir.resolve()),
    }
    (args.out_dir / "shap_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"SHAP outputs saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
