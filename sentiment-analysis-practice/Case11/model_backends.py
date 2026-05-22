from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentPrediction:
    sentiment_score: int
    confidence: float


class SentimentBackend(Protocol):
    def is_ready(self) -> bool: ...

    def predict(self, text: str) -> SentimentPrediction: ...

    def describe(self) -> Dict[str, Any]: ...


class UnavailableBackend:
    def __init__(self, reason: str, backend_name: str) -> None:
        self.reason = reason
        self.backend_name = backend_name

    def is_ready(self) -> bool:
        return False

    def predict(self, text: str) -> SentimentPrediction:
        raise RuntimeError(self.reason)

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "ready": False, "reason": self.reason}


class TransformersBackend:
    def __init__(self, model_dir: str, max_length: int = 128, device: Optional[str] = None) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.device_name = device or ("cuda" if self._cuda_available() else "cpu")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._device = torch.device(self.device_name)
        self._model = self._model.to(self._device).eval()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def is_ready(self) -> bool:
        return True

    def predict(self, text: str) -> SentimentPrediction:
        import torch.nn.functional as functional

        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        probabilities = functional.softmax(outputs.logits, dim=-1)[0]
        predicted_label = int(self._torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_label].item())
        return SentimentPrediction(sentiment_score=predicted_label, confidence=confidence)

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "transformers",
            "ready": True,
            "model_dir": self.model_dir,
            "device": self.device_name,
            "max_length": self.max_length,
        }


class CatBoostBackend:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)

        from catboost import CatBoostClassifier

        self._vectorizer = self._load_vectorizer()
        self._model = CatBoostClassifier()
        self._model.load_model(str(self.model_dir / "model.cbm"))

    def _load_vectorizer(self) -> Any:
        import joblib

        candidates = ("vectorizer.joblib", "vectorizer.pkl")
        for filename in candidates:
            path = self.model_dir / filename
            if path.exists():
                return joblib.load(path)

        raise FileNotFoundError(
            f"CatBoost vectorizer not found. Expected one of: {', '.join(candidates)}"
        )

    def is_ready(self) -> bool:
        return True

    def predict(self, text: str) -> SentimentPrediction:
        features = self._vectorizer.transform([text])

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features)[0]
            predicted_label = int(max(range(len(probabilities)), key=lambda idx: probabilities[idx]))
            confidence = float(probabilities[predicted_label])
            return SentimentPrediction(sentiment_score=predicted_label, confidence=confidence)

        predicted_label = int(self._model.predict(features)[0])
        return SentimentPrediction(sentiment_score=predicted_label, confidence=1.0)

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "catboost",
            "ready": True,
            "model_dir": str(self.model_dir),
            "artifacts": ["model.cbm", "vectorizer.joblib|vectorizer.pkl"],
        }


class LGBMBackend:
    MODEL_TO_APP_SCORE = {-1: 0, 0: 1, 1: 2}

    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)

        import joblib

        from lgbm_text_pipeline import LGBMTextFeaturePipeline

        self._pipeline = LGBMTextFeaturePipeline.from_dir(self.model_dir)
        self._model = joblib.load(self.model_dir / "lgbm_final_model.joblib")

    def is_ready(self) -> bool:
        return True

    def _to_app_score(self, internal_score: int) -> int:
        if internal_score not in self.MODEL_TO_APP_SCORE:
            raise ValueError(f"Unexpected LGBM class label: {internal_score}")
        return self.MODEL_TO_APP_SCORE[internal_score]

    def predict(self, text: str) -> SentimentPrediction:
        features = self._pipeline.transform([text])

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features)[0]
            classes = [int(label) for label in getattr(self._model, "classes_", range(len(probabilities)))]
            best_index = int(max(range(len(probabilities)), key=lambda idx: probabilities[idx]))
            internal_score = classes[best_index]
            confidence = float(probabilities[best_index])
            return SentimentPrediction(
                sentiment_score=self._to_app_score(internal_score),
                confidence=confidence,
            )

        raw_prediction = self._model.predict(features)[0]
        return SentimentPrediction(
            sentiment_score=self._to_app_score(int(raw_prediction)),
            confidence=1.0,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "lgbm",
            "ready": True,
            "model_dir": str(self.model_dir),
            "artifacts": [
                "lgbm_final_model.joblib",
                "feature_columns.joblib",
                "word_vectorizer.joblib",
                "char_vectorizer.joblib",
                "svd_word.joblib",
                "svd_char.joblib",
                "class_top_words.joblib",
            ],
            "label_mapping": {"-1": 0, "0": 1, "1": 2},
        }


def create_sentiment_backend(
    *,
    backend_name: Optional[str] = None,
    model_dir: Optional[str] = None,
    max_length: Optional[int] = None,
) -> SentimentBackend:
    selected_backend = (backend_name or os.getenv("SENTIMENT_BACKEND", "lgbm")).strip().lower()
    selected_model_dir = model_dir or os.getenv("SENTIMENT_MODEL_DIR", "./lgbm_sentiment_model")
    selected_max_length = max_length or int(os.getenv("SENTIMENT_MAX_LENGTH", "128"))

    if not os.path.isdir(selected_model_dir):
        return UnavailableBackend(
            reason=f"Model directory '{selected_model_dir}' was not found.",
            backend_name=selected_backend,
        )

    try:
        if selected_backend == "transformers":
            return TransformersBackend(model_dir=selected_model_dir, max_length=selected_max_length)
        if selected_backend == "catboost":
            return CatBoostBackend(model_dir=selected_model_dir)
        if selected_backend == "lgbm":
            return LGBMBackend(model_dir=selected_model_dir)

        return UnavailableBackend(
            reason=(
                f"Unknown model backend '{selected_backend}'. "
                "Supported backends: transformers, catboost, lgbm."
            ),
            backend_name=selected_backend,
        )
    except Exception as exc:
        LOGGER.exception("Failed to initialize backend '%s'", selected_backend)
        return UnavailableBackend(reason=str(exc), backend_name=selected_backend)
