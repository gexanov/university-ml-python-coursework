from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data_with_all_features"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "feature_importance.csv"


def load_train_reviews() -> pd.Series:
    train_df = pd.read_csv(
        DATA_DIR / "train_with_features.csv",
        usecols=["review"],
        encoding="utf-8-sig",
    )
    return train_df["review"].astype(str)


def build_vectorizers_and_svds(train_reviews: pd.Series):
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=7,
        max_df=0.95,
        max_features=8000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 7),
        min_df=7,
        max_df=0.95,
        max_features=5000,
        sublinear_tf=True,
    )

    X_train_word = word_vectorizer.fit_transform(train_reviews)
    X_train_char = char_vectorizer.fit_transform(train_reviews)

    svd_word = TruncatedSVD(n_components=200, random_state=42)
    svd_char = TruncatedSVD(n_components=150, random_state=42)

    svd_word.fit(X_train_word)
    svd_char.fit(X_train_char)

    return word_vectorizer, char_vectorizer, svd_word, svd_char


def load_feature_importance_map() -> dict[str, float]:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return {}
    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH, encoding="utf-8-sig")
    if {"feature", "importance"}.issubset(importance_df.columns):
        return dict(zip(importance_df["feature"], importance_df["importance"]))
    return {}


def format_terms(feature_names: np.ndarray, weights: np.ndarray, idx: np.ndarray) -> tuple[str, str]:
    terms = feature_names[idx]
    term_weights = [f"{weights[i]:.6f}" for i in idx]
    return " | ".join(terms.tolist()), " | ".join(term_weights)


def decode_components(
    prefix: str,
    feature_names: np.ndarray,
    components: np.ndarray,
    importance_map: dict[str, float],
    top_n: int = 15,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for component_idx, component_weights in enumerate(components):
        feature_name = f"{prefix}_{component_idx}"

        top_pos_idx = np.argsort(component_weights)[-top_n:][::-1]
        top_neg_idx = np.argsort(component_weights)[:top_n]

        top_positive_terms, top_positive_weights = format_terms(
            feature_names, component_weights, top_pos_idx
        )
        top_negative_terms, top_negative_weights = format_terms(
            feature_names, component_weights, top_neg_idx
        )

        description = (
            f"positive side: {top_positive_terms}; "
            f"negative side: {top_negative_terms}"
        )

        rows.append(
            {
                "feature": feature_name,
                "family": prefix,
                "component_index": component_idx,
                "model_importance": importance_map.get(feature_name, np.nan),
                "explained_variance_ratio": np.nan,
                "top_positive_terms": top_positive_terms,
                "top_positive_weights": top_positive_weights,
                "top_negative_terms": top_negative_terms,
                "top_negative_weights": top_negative_weights,
                "description": description,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    train_reviews = load_train_reviews()
    word_vectorizer, char_vectorizer, svd_word, svd_char = build_vectorizers_and_svds(train_reviews)
    importance_map = load_feature_importance_map()

    word_feature_names = np.array(word_vectorizer.get_feature_names_out())
    char_feature_names = np.array(char_vectorizer.get_feature_names_out())

    word_df = decode_components(
        prefix="word_svd",
        feature_names=word_feature_names,
        components=svd_word.components_,
        importance_map=importance_map,
    )
    word_df["explained_variance_ratio"] = svd_word.explained_variance_ratio_

    char_df = decode_components(
        prefix="char_svd",
        feature_names=char_feature_names,
        components=svd_char.components_,
        importance_map=importance_map,
    )
    char_df["explained_variance_ratio"] = svd_char.explained_variance_ratio_

    combined_df = pd.concat([word_df, char_df], ignore_index=True)
    combined_df = combined_df.sort_values(
        by=["model_importance", "family", "component_index"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    word_out = ROOT_DIR / "word_svd_descriptions.csv"
    char_out = ROOT_DIR / "char_svd_descriptions.csv"
    combined_out = ROOT_DIR / "svd_feature_descriptions.csv"

    word_df.to_csv(word_out, index=False, encoding="utf-8-sig")
    char_df.to_csv(char_out, index=False, encoding="utf-8-sig")
    combined_df.to_csv(combined_out, index=False, encoding="utf-8-sig")

    print(f"Saved: {word_out}")
    print(f"Saved: {char_out}")
    print(f"Saved: {combined_out}")
    print()
    print(combined_df[[
        "feature",
        "family",
        "model_importance",
        "explained_variance_ratio",
        "description",
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
