import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sklearn.metrics import f1_score
import pandas as pd
import io
import re 
import uuid # Для генерации task_id
from typing import Dict, Any
import logging
import json
import asyncio
import numpy as np # Для работы с метриками

from model_backends import SentimentBackend, create_sentiment_backend

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## --- Настройки Модели и Устройства ---
MODEL_BACKEND_NAME = os.getenv("SENTIMENT_BACKEND", "lgbm")
MODEL_DIR = os.getenv("SENTIMENT_MODEL_DIR", "./lgbm_sentiment_model")
RATING_CLASSES: Dict[int, str] = {0: "Негатив 🔴", 1: "Нейтрально 🟡", 2: "Позитив 🟢"}
RATING_MAP: Dict[str, int] = {"Негатив 🔴": 0, "Нейтрально 🟡": 1, "Позитив 🟢": 2}

# --- Временное хранилище для результатов анализа ---
# Ключ: task_id (UUID), Значение: pd.DataFrame с результатами
analysis_storage: Dict[str, pd.DataFrame] = {} 

# --- Схемы Данных ---
class TextInput(BaseModel):
    review: str

class CorrectionInput(BaseModel):
    data_id: int # Уникальный ID записи в DataFrame (создается при загрузке)
    new_sentiment: int # Новая оценка (0, 1 или 2)
    new_confidence: str = "100.00%" # Уверенность 100% для ручной корректировки

# --- Инициализация API ---
app = FastAPI(title="Russian Sentiment API Platform")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

@app.get("/")
async def root() -> dict:
    return {"message": "Russian Sentiment API Platform is running"}

# --- Глобальная Загрузка Модели ---
MODEL_BACKEND: SentimentBackend = create_sentiment_backend(
    backend_name=MODEL_BACKEND_NAME,
    model_dir=MODEL_DIR,
)
if MODEL_BACKEND.is_ready():
    logging.info("Модель успешно загружена: %s", MODEL_BACKEND.describe())
else:
    logging.error("Модель недоступна: %s", MODEL_BACKEND.describe())


# --- ФУНКЦИИ ---

def clean_text_adaptive(text: str) -> str:
    """Очищает текст (токенызация, удаление URL/Email/др.)."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text) # Удаление email
    text = re.sub(r'(\s*\bвиталий\s*\.\s*e-mail[:\.]*.*)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Удаление ссылок
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def is_model_ready() -> bool:
    return MODEL_BACKEND.is_ready()

def get_model_error_detail() -> Dict[str, Any]:
    description = MODEL_BACKEND.describe()
    return {"error": "Модель тональности не загружена.", "details": description}

def predict_sentiment(text: str) -> Dict[str, Any]:
    """Предсказывает тональность текста."""
    if not is_model_ready():
        return {"sentiment_label": "Ошибка API: Модель не загружена", "confidence": "0%", "sentiment_score": -1}

    if not text.strip():
        return {"sentiment_label": "Нет текста", "confidence": "0%", "sentiment_score": -1}

    prediction = MODEL_BACKEND.predict(text)

    return {
        "sentiment_label": RATING_CLASSES.get(prediction.sentiment_score, 'Неизвестно'),
        "confidence": f"{prediction.confidence * 100:.2f}%",
        "sentiment_score": prediction.sentiment_score
    }
    
def get_sentiment_from_label(label: str) -> int:
    """Конвертирует строковую метку в числовую оценку (0, 1, 2)."""
    return RATING_MAP.get(label, -1)


# --- МАРШРУТЫ API ---

# --- 1. Одиночный Анализ (без изменений) ---
@app.post("/analyze")
async def analyze_text(data: TextInput) -> Dict[str, Any]:
    if not is_model_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=get_model_error_detail())
        
    cleaned_review = clean_text_adaptive(data.review)
    prediction = predict_sentiment(cleaned_review)
    
    return {
        "text_original": data.review, 
        "text_cleaned": cleaned_review, 
        "sentiment": prediction['sentiment_label'],
        "confidence": prediction['confidence']
    }


# --- 2. Загрузка, Анализ и Хранение (Замена /start-analysis) ---
@app.post("/upload-and-store")
async def upload_and_store(
    file: UploadFile = File(...), 
    review_column: str = Form("text"), 
    csv_delimiter: str = Form(";") 
):
    """
    Загружает CSV, анализирует, сохраняет результат и возвращает task_id.
    Включает SSE для отслеживания прогресса.
    """
    if not is_model_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=get_model_error_detail())

    content = await file.read()
    
    # Генератор для отправки событий SSE
    async def generate_response():
        task_id = str(uuid.uuid4())
        yield f"data: {json.dumps({'status': 'INITIALIZING', 'message': f'Начинаю анализ файла: {file.filename}', 'progress': 0, 'task_id': task_id})}\n\n"
        await asyncio.sleep(0.01)

        try:
            # Чтение и проверка файла
            try:
                csv_data = io.StringIO(content.decode('utf-8'))
            except UnicodeDecodeError:
                csv_data = io.StringIO(content.decode('cp1251')) 
            
            df = pd.read_csv(csv_data, sep=csv_delimiter, engine='python', on_bad_lines='warn')
            df.dropna(axis=1, how='all', inplace=True) 

            if review_column not in df.columns:
                 yield f"data: {json.dumps({'status': 'ERROR', 'message': f'Ошибка: Столбец "{review_column}" не найден.'})}\n\n"
                 return
        
        except Exception as e:
            yield f"data: {json.dumps({'status': 'ERROR', 'message': f'Ошибка чтения файла: {str(e)}'})}\n\n"
            return

        # Инициализация столбцов
        df['text_original'] = df[review_column].astype(str).fillna('') 
        df['text_cleaned'] = df['text_original'].apply(clean_text_adaptive)
        df['source'] = file.filename # Для фильтрации по источникам
        df['data_id'] = range(len(df)) # Уникальный ID для корректировки
        df['sentiment_label'] = None
        df['sentiment_score'] = -1
        df['confidence'] = "0%"

        texts_to_analyze = df['text_cleaned'] 
        total_records = len(texts_to_analyze)
        
        # --- Анализ тональности (10% -> 90%) ---
        progress_start = 10 
        progress_range = 80 
        
        for i, text in enumerate(texts_to_analyze):
            prediction = predict_sentiment(text)
            
            # Обновляем DataFrame
            df.loc[df.index[i], 'sentiment_label'] = prediction['sentiment_label']
            df.loc[df.index[i], 'sentiment_score'] = prediction['sentiment_score']
            df.loc[df.index[i], 'confidence'] = prediction['confidence']
            
            if (i + 1) % 100 == 0 or i == total_records - 1:
                current_progress = progress_start + (i + 1) / total_records * progress_range
                yield f"data: {json.dumps({'status': 'ANALYZING', 'message': f'Обработано {i + 1} из {total_records} записей.', 'progress': min(95, int(current_progress))})}\n\n"
        
        # --- Финализация и сохранение (90% -> 100%) ---
        analysis_storage[task_id] = df
        logging.info(f"Анализ завершен. task_id: {task_id}. Сохранено {len(df)} записей.")

        yield f"data: {json.dumps({'status': 'FINALIZED', 'message': 'Анализ завершен, данные сохранены.', 'progress': 100, 'task_id': task_id})}\n\n"
        
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
    
    
# --- 3. Получение, поиск и фильтрация результатов ---
@app.get("/results/{task_id}")
async def get_results(
    task_id: str, 
    source: str = None, 
    sentiment_score: int = None, # Оценка: 0, 1 или 2
    keyword: str = None, 
    page: int = 1,
    page_size: int = 50
):
    """
    Возвращает размеченные данные с пагинацией, поиском и фильтрацией.
    """
    if task_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    df = analysis_storage[task_id].copy()
    initial_count = len(df)
    
    # 1. Фильтрация
    if source:
        df = df[df['source'].str.contains(source, case=False, na=False)]
    
    if sentiment_score is not None and sentiment_score in [0, 1, 2]:
        df = df[df['sentiment_score'] == sentiment_score]

    # 2. Поиск по ключевому слову
    if keyword:
        # Поиск по столбцу очищенного текста
        df = df[df['text_cleaned'].str.contains(keyword, case=False, na=False)] 

    filtered_count = len(df)
    
    # 3. Пагинация
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_df = df.iloc[start_index:end_index]
    
    return {
        "total_records": initial_count,
        "filtered_records": filtered_count,
        "current_page": page,
        "total_pages": int(np.ceil(filtered_count / page_size)),
        "data": paginated_df.to_dict(orient='records'),
        "available_sources": list(df['source'].unique())
    }


# --- 4. Ручная корректировка разметки ---
@app.put("/results/{task_id}/correct")
async def correct_sentiment(task_id: str, correction: CorrectionInput):
    """
    Обновляет тональность для указанной записи, применяя ручную корректировку.
    """
    if task_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    df = analysis_storage[task_id]
    
    # Находим индекс записи по data_id
    idx = df[df['data_id'] == correction.data_id].index
    
    if idx.empty:
        raise HTTPException(status_code=404, detail=f"Запись с data_id={correction.data_id} не найдена.")

    # Обновление данных
    row_index = idx[0]
    new_sentiment_label = RATING_CLASSES.get(correction.new_sentiment, 'Неизвестно')
    
    # Используем .loc для безопасного обновления
    df.loc[row_index, 'sentiment_label'] = new_sentiment_label
    df.loc[row_index, 'sentiment_score'] = correction.new_sentiment 
    df.loc[row_index, 'confidence'] = correction.new_confidence # Обычно 100%

    return {"message": "Корректировка успешно применена", "updated_id": correction.data_id}


# --- 5. Расчет метрик (Macro-F1) ---
@app.post("/metrics/{task_id}")
async def calculate_metrics(task_id: str, validation_file: UploadFile = File(...), validation_column: str = Form("label")):
    """
    Сравнивает размеченные моделью данные (возможно, скорректированные) с валидационным файлом
    и рассчитывает метрику macro-F1.
    """
    if task_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    # 1. Получаем данные модели (model_df)
    model_df = analysis_storage[task_id].copy()
    
    # 2. Чтение валидационного файла
    try:
        content = await validation_file.read()
        try:
            csv_data = io.StringIO(content.decode('utf-8'))
        except UnicodeDecodeError:
            csv_data = io.StringIO(content.decode('cp1251')) 
        
        validation_df = pd.read_csv(csv_data, sep=',', engine='python', on_bad_lines='warn')
        validation_df.dropna(axis=1, how='all', inplace=True) 

    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": f"Не удалось прочитать валидационный файл: {e}"})

    if validation_column not in validation_df.columns:
        raise HTTPException(status_code=400, detail={"error": f"Столбец '{validation_column}' не найден в валидационном файле."})

    # 3. Выравнивание данных (Предполагаем, что порядок записей совпадает, 
    # или что валидационный файл имеет ту же структуру/размер, что и оригинал)
    
    # Для простоты предполагаем, что длина совпадает и порядок важен
    if len(model_df) != len(validation_df):
        return JSONResponse(status_code=400, content={"error": f"Размер валидационного файла ({len(validation_df)}) не совпадает с размером анализируемых данных ({len(model_df)})."})

    # Извлечение меток
    y_true = validation_df[validation_column].astype(int).tolist()
    y_pred = model_df['sentiment_score'].astype(int).tolist()

    # 4. Расчет Macro-F1
    try:
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Можно рассчитать F1 для каждого класса
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
        
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"Ошибка расчета F1: Проверьте, что метки валидационного файла находятся в диапазоне 0-2. Детали: {e}"})

    return {
        "metric_name": "Macro F1-Score",
        "value": f1_macro,
        "class_f1_scores": {
            RATING_CLASSES[0]: f1_per_class[0],
            RATING_CLASSES[1]: f1_per_class[1],
            RATING_CLASSES[2]: f1_per_class[2]
        }
    }


# --- 6. Агрегированные данные для Визуализации ---
@app.get("/visualizations/{task_id}")
async def get_visualization_data(task_id: str, aggregation_type: str = "sentiment_distribution"):
    """
    Возвращает агрегированные данные для построения графиков.
    """
    if task_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    df = analysis_storage[task_id]
    
    if aggregation_type == "sentiment_distribution":
        # Круговая диаграмма: распределение тональности по всем текстам
        counts = df['sentiment_label'].value_counts().reset_index()
        counts.columns = ['label', 'count']
        return JSONResponse(content={
            "type": "pie",
            "title": "Общее распределение тональности",
            "data": counts.to_dict(orient='records')
        })
        
    elif aggregation_type == "sentiment_by_source":
        # Столбчатая диаграмма: тональность по источникам
        pivot_table = df.groupby('source')['sentiment_label'].value_counts().unstack(fill_value=0)
        
        # Добавляем недостающие столбцы, чтобы структура была одинаковой
        for label in RATING_CLASSES.values():
            if label not in pivot_table.columns:
                pivot_table[label] = 0
                
        # Приводим к нужному формату для фронтенда
        data_list = []
        for source, row in pivot_table.iterrows():
            item = {"source": source}
            item.update(row.to_dict())
            data_list.append(item)
            
        return JSONResponse(content={
            "type": "bar_stacked",
            "title": "Распределение тональности по источникам",
            "data": data_list
        })

    # --- 7. Маршрут для скачивания (Обновленный) ---
@app.get("/download-csv/{task_id}")
async def download_csv(task_id: str, csv_delimiter: str = ";"):
    """
    Позволяет скачать размеченные (и возможно, скорректированные) данные.
    """
    if task_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    
    df = analysis_storage[task_id].copy()
    
    # Удаляем служебные столбцы перед скачиванием
    df = df.drop(columns=['data_id', 'sentiment_score', 'text_cleaned'], errors='ignore')

    output = io.StringIO()
    df.to_csv(output, index=False, sep=csv_delimiter)
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=sentiment_results_final.csv",
        }
    )
