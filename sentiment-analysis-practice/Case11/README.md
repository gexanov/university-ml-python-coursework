# Case11
cd "проектпо питону\Case11"
pip install -r requirements.txt
npm install
npm run start:all

Приложение для анализа тональности отзывов:
- backend на `FastAPI`
- frontend на `React`

## Что используется сейчас

По умолчанию backend запускается на `LGBM`, потому что эта модель показала лучший `macro F1` в сравнении с остальными моделями проекта.

Артефакты runtime-модели лежат в:

`./lgbm_sentiment_model`

Переключение backend можно сделать через переменные окружения:

- `SENTIMENT_BACKEND=lgbm|catboost|transformers`
- `SENTIMENT_MODEL_DIR=...`
- `SENTIMENT_MAX_LENGTH=128` для `transformers`

## Быстрый старт

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend поднимется на `http://localhost:8000`.

### Frontend

```bash
npm install
npm start
```

Frontend поднимется на `http://localhost:3000`.

### Запуск всего сразу

```bash
npm install
pip install -r requirements.txt
npm run start:all
```

## LGBM runtime-артефакты

Если нужно пересобрать runtime-пакет для приложения из обученной `LGBM`, используй:

```bash
python lgbm_text_pipeline.py
```

Скрипт:
- берет `train_with_features.csv` из `../ML/data_with_all_features`
- берет модель и список признаков из `../ML/LGBM/artifacts`
- собирает и сохраняет:
  - `lgbm_final_model.joblib`
  - `feature_columns.joblib`
  - `word_vectorizer.joblib`
  - `char_vectorizer.joblib`
  - `svd_word.joblib`
  - `svd_char.joblib`
  - `class_top_words.joblib`
  - `metadata.json`

## Полезные команды

- `npm run build` — production-сборка frontend
- `npm test` — тесты frontend
