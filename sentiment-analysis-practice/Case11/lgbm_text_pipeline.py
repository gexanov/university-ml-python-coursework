from __future__ import annotations

import argparse
import json
import re
import shutil
import string
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


POS_WORDS = {
    "хороший",
    "отличный",
    "супер",
    "прекрасный",
    "понравился",
    "понравилось",
    "красивый",
    "удобный",
    "качественный",
    "рекомендую",
    "идеально",
    "люблю",
    "замечательный",
    "классный",
    "мягкий",
    "приятный",
}

NEG_WORDS = {
    "плохой",
    "ужасный",
    "отвратительный",
    "кошмар",
    "брак",
    "некачественный",
    "разочарование",
    "ужас",
    "плохо",
    "обман",
    "тонкий",
    "кривой",
    "неудобный",
    "грязный",
    "порванный",
    "дешевый",
}

INTENSIFIERS = {
    "очень",
    "сильно",
    "крайне",
    "совсем",
    "слишком",
    "безумно",
    "максимально",
    "весьма",
    "очень-очень",
}

NEGATIONS = {"не", "нет", "никогда", "ни", "нельзя"}

DELIVERY_WORDS = {
    "доставка",
    "доставили",
    "пришел",
    "пришла",
    "задержка",
    "вернули",
    "посылка",
    "курьер",
    "приход",
    "отправка",
    "доставку",
}

QUALITY_WORDS = {
    "ткань",
    "шов",
    "материал",
    "качество",
    "нитки",
    "синтетика",
    "брак",
    "рисунок",
    "криво",
    "пошив",
    "подкладка",
}

SIZE_WORDS = {
    "размер",
    "маломерит",
    "большемерит",
    "узкий",
    "широкий",
    "маленький",
    "большой",
    "короткий",
    "длинный",
}

PATTERNS = {
    "has_not_recommend": r"не\s+рекомендую",
    "has_recommend": r"\bрекомендую\b",
    "has_liked": r"очень\s+понрав|понравил",
    "has_not_arrived": r"не\s+пришел|не\s+пришла|не\s+дошел|не\s+дошла",
    "has_refund": r"деньги\s+вернули|возврат|вернули\s+деньги",
    "has_mismatch": r"не\s+соответствует|не\s+такой|как\s+на\s+фото|не\s+как\s+на\s+фото",
    "has_bad_quality": r"плохое\s+качество|ужасн\w+\s+ткан\w+|торчат\s+нитки|кривой\s+шов",
    "has_good_quality": r"отличн\w+\s+качеств\w+|приятн\w+\s+ткан\w+|хорош\w+\s+качеств\w+",
    "has_size_issue": r"маломерит|большемерит|не\s+подошел\s+размер|не\s+подошла\s+по\s+размеру",
    "has_strong_negative_phrase": r"полное\s+разочарование|это\s+ужас|просто\s+ужас|кошмар",
    "has_strong_positive_phrase": r"в\s+восторге|очень\s+довольна|очень\s+доволен|просто\s+супер",
}

STOPWORDS_RU = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "ее",
    "мне",
    "есть",
    "они",
    "тут",
    "где",
    "или",
    "ни",
    "мы",
    "это",
    "от",
    "до",
    "для",
    "из",
    "под",
    "над",
    "при",
    "без",
    "после",
}

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
MULTI_PUNCT_RE = re.compile(r"(\!\!+|\?\?+|\?\!+|\!\?+)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)
CAPS_WORD_RE = re.compile(r"\b[А-ЯЁA-Z]{2,}\b")
UPPERCASE_LETTER_RE = re.compile(r"[А-ЯЁA-Z]")
DIGIT_RE = re.compile(r"\d")
NEWLINE_RE = re.compile(r"[\r\n\t]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_CHARS = set(string.punctuation + "«»—…„“”№")


def normalize_text_soft(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = NEWLINE_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def count_words_from_vocab(tokens: list[str], vocab: set[str]) -> int:
    return sum(token in vocab for token in tokens)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def max_consecutive_char_repeat(text: str) -> int:
    max_repeat = 1
    current = 1
    for idx in range(1, len(text)):
        if text[idx] == text[idx - 1]:
            current += 1
            max_repeat = max(max_repeat, current)
        else:
            current = 1
    return max_repeat if text else 0


def extract_raw_features(text: str) -> dict[str, float]:
    raw = text if isinstance(text, str) else ""
    words_raw = WORD_RE.findall(raw)
    sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(raw) if item.strip()]
    caps_words = CAPS_WORD_RE.findall(raw)

    num_chars = len(raw)
    num_words_raw = len(words_raw)
    num_sentences = len(sentences)
    num_uppercase_letters = len(UPPERCASE_LETTER_RE.findall(raw))
    num_exclaims = raw.count("!")
    num_questions = raw.count("?")
    num_dots = raw.count(".")
    num_commas = raw.count(",")
    num_colons = raw.count(":")
    num_semicolons = raw.count(";")
    num_quotes = sum(raw.count(ch) for ch in ['"', "'", "«", "»", "„", "“", "”"])
    num_brackets = sum(raw.count(ch) for ch in "()[]{}")
    num_digits = len(DIGIT_RE.findall(raw))
    num_spaces = raw.count(" ")
    num_newlines = len(re.findall(r"[\r\n]", raw))
    num_punct = sum(ch in PUNCT_CHARS for ch in raw)
    num_mixed_punct = len(MULTI_PUNCT_RE.findall(raw))
    num_repeated_chars = len(REPEAT_CHAR_RE.findall(raw))
    max_char_repeat = max_consecutive_char_repeat(raw)

    return {
        "num_chars": num_chars,
        "num_words_raw": num_words_raw,
        "num_sentences": num_sentences,
        "num_uppercase_letters": num_uppercase_letters,
        "uppercase_ratio": safe_div(num_uppercase_letters, num_chars),
        "num_exclaims": num_exclaims,
        "num_questions": num_questions,
        "num_dots": num_dots,
        "num_commas": num_commas,
        "num_colons": num_colons,
        "num_semicolons": num_semicolons,
        "num_quotes": num_quotes,
        "num_brackets": num_brackets,
        "num_digits": num_digits,
        "digit_ratio": safe_div(num_digits, num_chars),
        "num_punct": num_punct,
        "punct_ratio": safe_div(num_punct, num_chars),
        "num_spaces": num_spaces,
        "space_ratio": safe_div(num_spaces, num_chars),
        "num_newlines": num_newlines,
        "num_mixed_punct": num_mixed_punct,
        "has_many_exclaims": int(num_exclaims >= 3),
        "has_many_questions": int(num_questions >= 3),
        "has_ellipsis": int("..." in raw),
        "num_caps_words": len(caps_words),
        "has_caps_word": int(bool(caps_words)),
        "max_caps_word_len": max((len(word) for word in caps_words), default=0),
        "num_repeated_chars": num_repeated_chars,
        "max_char_repeat": max_char_repeat,
        "num_long_tokens_raw": sum(len(word) > 10 for word in words_raw),
    }


def extract_length_features(text: str) -> dict[str, float]:
    soft = normalize_text_soft(text)
    tokens = WORD_RE.findall(soft)
    token_lengths = [len(token) for token in tokens]
    sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(soft) if item.strip()]
    sentence_word_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    sentence_char_lengths = [len(sentence) for sentence in sentences]

    num_words = len(tokens)
    num_unique_words = len(set(token.lower() for token in tokens))

    return {
        "avg_word_len": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "median_word_len": float(np.median(token_lengths)) if token_lengths else 0.0,
        "std_word_len": float(np.std(token_lengths)) if token_lengths else 0.0,
        "num_unique_words": num_unique_words,
        "unique_ratio": safe_div(num_unique_words, num_words),
        "num_short_words": sum(len(token) <= 2 for token in tokens),
        "num_medium_words": sum(3 <= len(token) <= 6 for token in tokens),
        "num_long_words": sum(len(token) >= 7 for token in tokens),
        "avg_sentence_len_words": float(np.mean(sentence_word_lengths)) if sentence_word_lengths else 0.0,
        "avg_sentence_len_chars": float(np.mean(sentence_char_lengths)) if sentence_char_lengths else 0.0,
        "num_empty_like_fragments": max(0, len(SENTENCE_SPLIT_RE.split(soft)) - len(sentences)),
    }


def extract_lexicon_features(text: str) -> dict[str, float]:
    lower = normalize_text_soft(text).lower()
    tokens = tokenize(lower)

    pos_count = count_words_from_vocab(tokens, POS_WORDS)
    neg_count = count_words_from_vocab(tokens, NEG_WORDS)
    intensifier_count = count_words_from_vocab(tokens, INTENSIFIERS)
    negation_count = count_words_from_vocab(tokens, NEGATIONS)
    delivery_count = count_words_from_vocab(tokens, DELIVERY_WORDS)
    quality_count = count_words_from_vocab(tokens, QUALITY_WORDS)
    size_count = count_words_from_vocab(tokens, SIZE_WORDS)
    num_words = len(tokens)

    return {
        "num_negations": negation_count,
        "num_intensifiers": intensifier_count,
        "num_positive_words": pos_count,
        "num_negative_words": neg_count,
        "sentiment_lexicon_diff": pos_count - neg_count,
        "num_delivery_words": delivery_count,
        "num_quality_words": quality_count,
        "num_size_words": size_count,
        "neg_words_per_word": safe_div(neg_count, num_words),
        "pos_words_per_word": safe_div(pos_count, num_words),
        "lexicon_balance_ratio": safe_div(pos_count + 1, neg_count + 1),
    }


def extract_pattern_features(text: str) -> dict[str, int]:
    lower = normalize_text_soft(text).lower()
    return {name: int(bool(re.search(pattern, lower))) for name, pattern in PATTERNS.items()}


def extract_ratio_features(features: dict[str, float]) -> dict[str, float]:
    num_words = features.get("num_words_raw", 0)
    num_sentences = features.get("num_sentences", 0)
    return {
        "exclaims_per_word": safe_div(features.get("num_exclaims", 0), num_words),
        "questions_per_word": safe_div(features.get("num_questions", 0), num_words),
        "caps_per_word": safe_div(features.get("num_caps_words", 0), num_words),
        "punct_per_word": safe_div(features.get("num_punct", 0), num_words),
        "repeat_chars_per_word": safe_div(features.get("num_repeated_chars", 0), num_words),
        "avg_punct_per_sentence": safe_div(features.get("num_punct", 0), num_sentences),
        "avg_exclaims_per_sentence": safe_div(features.get("num_exclaims", 0), num_sentences),
    }


def extract_all_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(extract_raw_features(text))
    features.update(extract_length_features(text))
    features.update(extract_lexicon_features(text))
    features.update(extract_pattern_features(text))
    features.update(extract_ratio_features(features))
    return features


def simple_tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[а-яёa-z]+", text.lower())


def get_top_words_by_class(
    texts: list[str],
    labels: list[int],
    top_n: int = 50,
    min_len: int = 3,
    stopwords: set[str] | None = None,
) -> dict[int, list[str]]:
    stopwords = stopwords or set()
    class_top_words: dict[int, list[str]] = {}

    for cls in sorted(set(labels)):
        class_texts = [text for text, label in zip(texts, labels) if label == cls]
        all_tokens: list[str] = []
        for text in class_texts:
            tokens = simple_tokenize(text)
            tokens = [token for token in tokens if len(token) >= min_len and token not in stopwords]
            all_tokens.extend(tokens)
        class_top_words[int(cls)] = [word for word, _ in Counter(all_tokens).most_common(top_n)]

    return class_top_words


def extract_top_word_features(text: str, class_top_words: dict[int, list[str]]) -> dict[str, float]:
    tokens = simple_tokenize(text)
    token_set = set(tokens)
    num_words = len(tokens)
    features: dict[str, float] = {}

    for cls, words in class_top_words.items():
        words_set = set(words)
        match_count = sum(token in words_set for token in tokens)
        unique_match_count = len(token_set & words_set)
        features[f"class_{cls}_top_match_count"] = match_count
        features[f"class_{cls}_top_unique_match_count"] = unique_match_count
        features[f"class_{cls}_top_match_ratio"] = match_count / num_words if num_words else 0.0
        features[f"class_{cls}_has_top_word"] = int(unique_match_count > 0)

    if -1 in class_top_words and 1 in class_top_words:
        features["pos_neg_top_diff"] = (
            features.get("class_1_top_match_count", 0) - features.get("class_-1_top_match_count", 0)
        )

    return features


class LGBMTextFeaturePipeline:
    def __init__(
        self,
        *,
        word_vectorizer: TfidfVectorizer,
        char_vectorizer: TfidfVectorizer,
        svd_word: TruncatedSVD,
        svd_char: TruncatedSVD,
        class_top_words: dict[int, list[str]],
        feature_columns: list[str],
    ) -> None:
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer
        self.svd_word = svd_word
        self.svd_char = svd_char
        self.class_top_words = {int(key): list(value) for key, value in class_top_words.items()}
        self.feature_columns = list(feature_columns)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "LGBMTextFeaturePipeline":
        model_path = Path(model_dir)
        return cls(
            word_vectorizer=joblib.load(model_path / "word_vectorizer.joblib"),
            char_vectorizer=joblib.load(model_path / "char_vectorizer.joblib"),
            svd_word=joblib.load(model_path / "svd_word.joblib"),
            svd_char=joblib.load(model_path / "svd_char.joblib"),
            class_top_words=joblib.load(model_path / "class_top_words.joblib"),
            feature_columns=joblib.load(model_path / "feature_columns.joblib"),
        )

    def transform(self, texts: list[str]) -> pd.DataFrame:
        normalized_texts = [text if isinstance(text, str) else "" for text in texts]

        manual_df = pd.DataFrame([extract_all_features(text) for text in normalized_texts])
        top_df = pd.DataFrame(
            [extract_top_word_features(text, self.class_top_words) for text in normalized_texts]
        )

        word_matrix = self.word_vectorizer.transform(normalized_texts)
        char_matrix = self.char_vectorizer.transform(normalized_texts)
        word_svd = self.svd_word.transform(word_matrix)
        char_svd = self.svd_char.transform(char_matrix)

        word_df = pd.DataFrame(word_svd, columns=[f"word_svd_{idx}" for idx in range(word_svd.shape[1])])
        char_df = pd.DataFrame(char_svd, columns=[f"char_svd_{idx}" for idx in range(char_svd.shape[1])])

        features_df = pd.concat(
            [
                manual_df.reset_index(drop=True),
                top_df.reset_index(drop=True),
                word_df.reset_index(drop=True),
                char_df.reset_index(drop=True),
            ],
            axis=1,
        )
        features_df = features_df.reindex(columns=self.feature_columns, fill_value=0.0)
        return features_df


def build_runtime_artifacts(
    *,
    train_csv_path: str | Path,
    source_artifacts_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    train_csv_path = Path(train_csv_path)
    source_artifacts_dir = Path(source_artifacts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv_path, usecols=["review", "target"])
    train_reviews = train_df["review"].astype(str).tolist()
    train_targets = train_df["target"].astype(int).tolist()

    class_top_words = get_top_words_by_class(
        texts=train_reviews,
        labels=train_targets,
        top_n=50,
        min_len=3,
        stopwords=STOPWORDS_RU,
    )

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

    word_matrix = word_vectorizer.fit_transform(train_reviews)
    char_matrix = char_vectorizer.fit_transform(train_reviews)

    svd_word = TruncatedSVD(n_components=200, random_state=42)
    svd_char = TruncatedSVD(n_components=150, random_state=42)
    svd_word.fit(word_matrix)
    svd_char.fit(char_matrix)

    for filename in ("lgbm_final_model.joblib", "feature_columns.joblib", "best_params.json"):
        source = source_artifacts_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    joblib.dump(word_vectorizer, output_dir / "word_vectorizer.joblib")
    joblib.dump(char_vectorizer, output_dir / "char_vectorizer.joblib")
    joblib.dump(svd_word, output_dir / "svd_word.joblib")
    joblib.dump(svd_char, output_dir / "svd_char.joblib")
    joblib.dump(class_top_words, output_dir / "class_top_words.joblib")

    metadata = {
        "classes": sorted(set(train_targets)),
        "app_score_mapping": {"-1": 0, "0": 1, "1": 2},
        "source_train_csv": str(train_csv_path),
        "source_artifacts_dir": str(source_artifacts_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build runtime LGBM sentiment assets for Case11.")
    parser.add_argument(
        "--train-csv",
        default=str(root_dir.parent / "ML" / "data_with_all_features" / "train_with_features.csv"),
    )
    parser.add_argument(
        "--source-artifacts",
        default=str(root_dir.parent / "ML" / "LGBM" / "artifacts"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "lgbm_sentiment_model"),
    )
    args = parser.parse_args()

    output_dir = build_runtime_artifacts(
        train_csv_path=args.train_csv,
        source_artifacts_dir=args.source_artifacts,
        output_dir=args.output_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
