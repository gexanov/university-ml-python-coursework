import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import json
import warnings
from pathlib import Path
import nltk
from wordcloud import WordCloud

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# =============================================================================
# 📋 КОНФИГУРАЦИЯ (все настройки в одном месте)
# =============================================================================
CONFIG = {
    # 👇 ВСТАВЛЕН ВАШ ПОЛНЫЙ ПУТЬ К ФАЙЛУ
    'input_file': r'C:\Users\79635\Downloads\rureviews-master\rureviews-master\EDA\women-clothing-accessories.3-class.balanced.csv',
    'output_dir': 'eda_output',
    'test_size': 0.15,
    'val_size': 0.176,  # 0.176 * 0.85 ≈ 0.15 от общего
    'random_state': 42,
    'max_len_percentile': 95,
    'min_word_length': 3,
    'top_n_words': 30,
    'encoding': 'utf-8-sig'
}

# Создаём папку для результатов
Path(CONFIG['output_dir']).mkdir(exist_ok=True)


# =============================================================================
# 🛠 КЛАСС ДЛЯ ПРОВЕДЕНИЯ EDA
# =============================================================================
class ReviewsEDA:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.train = None
        self.val = None
        self.test = None
        self.report = {}
        self.stopwords_ru = set(stopwords.words('russian'))

    def load_data(self, filepath):
        """Загрузка данных с обработкой формата RuReviews и УДАЛЕНИЕМ ДУБЛИКАТОВ"""
        print(f"📂 Загрузка данных из {filepath}...")

        # 1. Читаем с табуляцией как разделителем
        self.df = pd.read_csv(filepath, encoding='utf-8', sep='\t')
        print(f"✅ Файл загружен (разделитель: таб)")

        # 2. Переименовываем колонки под наш формат
        if 'review' in self.df.columns and 'text' not in self.df.columns:
            self.df = self.df.rename(columns={'review': 'text', 'sentiment': 'label'})
            print("✅ Колонки переименованы: review→text, sentiment→label")

        # 3. Принудительно конвертируем метки в числа
        print(f"🏷️ Уникальные метки до конвертации: {self.df['label'].unique()[:10]}")

        label_map = {
            'negative': 0, 'Negative': 0, 'neg': 0, 'NEGATIVE': 0,
            'neutral': 1, 'Neutral': 1, 'neautral': 1, 'Neautral': 1, 'neu': 1, 'NEUTRAL': 1,
            'positive': 2, 'Positive': 2, 'pos': 2, 'POSITIVE': 2
        }

        initial_len = len(self.df)

        # Конвертируем через map
        self.df['label'] = self.df['label'].map(label_map)

        # Считаем сколько стало NaN
        null_count = self.df['label'].isnull().sum()
        if null_count > 0:
            print(f"⚠️ Найдено {null_count} строк с нераспознанными метками")
            unrecognized = self.df[self.df['label'].isnull()]['label'].unique()[:10]
            print(f"   Нераспознанные метки: {unrecognized}")

        # Удаляем строки с NaN
        self.df = self.df.dropna(subset=['label'])

        # Конвертируем в int
        self.df['label'] = self.df['label'].astype(int)

        deleted = initial_len - len(self.df)
        if deleted > 0:
            print(f"⚠️ Удалено {deleted} строк с нераспознанными метками")
        else:
            print("✅ Все метки успешно конвертированы")

        # 4. Проверка обязательных колонок
        required_cols = ['text', 'label']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            print(f"❌ Отсутствуют колонки: {missing}")
            print(f"📋 Доступные колонки: {list(self.df.columns)}")
            raise ValueError(f"Отсутствуют колонки: {missing}")

        print(f"✅ После конвертации меток: {len(self.df):,} строк")

        # 5. 🔥 УДАЛЕНИЕ ДУБЛИКАТОВ ТЕКСТОВ (КРИТИЧЕСКИ ВАЖНО!)
        print("\n🧹 Очистка дубликатов...")
        texts_before = len(self.df)

        # Сначала удаляем полные дубликаты строк
        self.df = self.df.drop_duplicates(subset=['text', 'label'])

        # Затем удаляем дубликаты текстов (оставляем первый встреченный)
        # Это гарантирует, что один текст не попадет и в train, и в test
        self.df = self.df.drop_duplicates(subset=['text'], keep='first')

        texts_after = len(self.df)
        duplicates_removed = texts_before - texts_after

        print(f"📊 Было текстов: {texts_before:,}")
        print(f"📊 Стало текстов: {texts_after:,}")
        print(f"✅ Удалено дубликатов: {duplicates_removed:,}")

        # 6. Удаляем пустые тексты
        self.df = self.df[self.df['text'].astype(str).str.strip() != '']
        empty_removed = texts_after - len(self.df)
        if empty_removed > 0:
            print(f"✅ Удалено пустых текстов: {empty_removed}")

        print(f"\n📋 Итого после очистки: {len(self.df):,} строк")
        print(f"📊 Тип колонки label: {self.df['label'].dtype}")
        print(f"🏷️ Уникальные классы: {sorted(self.df['label'].unique())}")

        # Сохраняем статистику дубликатов в отчёт
        self.report['duplicates'] = {'text_only': int(duplicates_removed), 'full_rows': int(duplicates_removed)}
        self.report['empty_texts'] = int(empty_removed)

        return self

    def basic_quality_check(self):
        """Базовая проверка качества данных"""
        print("\n🔍 1. БАЗОВАЯ ПРОВЕРКА КАЧЕСТВА")
        print("-" * 50)

        # Пропуски
        nulls = self.df.isnull().sum()
        self.report['nulls'] = nulls.to_dict()
        print(f"Пропуски в 'text': {nulls.get('text', 0)}")
        print(f"Пропуски в 'label': {nulls.get('label', 0)}")

        # Дубликаты (должно быть 0, так как мы их удалили в load_data)
        text_duplicates = self.df['text'].duplicated().sum()
        full_duplicates = self.df.duplicated().sum()
        print(f"Дубликаты текстов (остаток): {text_duplicates:,}")
        print(f"Полные дубликаты строк (остаток): {full_duplicates:,}")

        # Пустые тексты
        empty_texts = (self.df['text'].astype(str).str.strip() == '').sum()
        print(f"Пустые тексты (остаток): {empty_texts}")

        # Уникальные классы
        unique_labels = self.df['label'].nunique()
        print(f"Уникальных классов: {unique_labels}")

        return self

    def text_analysis(self):
        """Глубокий анализ текстов"""
        print("\n📝 2. АНАЛИЗ ТЕКСТОВ")
        print("-" * 50)

        # Метрики длины
        self.df['char_len'] = self.df['text'].astype(str).apply(len)
        self.df['word_count'] = self.df['text'].astype(str).apply(lambda x: len(x.split()))

        # Статистика через describe()
        stats = {
            'char_len': self.df['char_len'].describe().to_dict(),
            'word_count': self.df['word_count'].describe().to_dict()
        }
        self.report['text_stats'] = stats

        # Перцентили через numpy (надёжнее)
        p95_char = np.percentile(self.df['char_len'], 95)
        p95_words = np.percentile(self.df['word_count'], 95)

        print(f"Средняя длина (симв): {stats['char_len']['mean']:.1f}")
        print(f"Медиана длины (симв): {stats['char_len']['50%']:.1f}")
        print(f"95-й перцентиль: {p95_char:.1f}")
        print(f"Среднее кол-во слов: {stats['word_count']['mean']:.1f}")

        # Рекомендация для max_len
        max_len_rec = int(p95_char)
        self.report['recommended_max_len'] = max_len_rec
        print(f"\n💡 Рекомендация max_len: {max_len_rec} (покрывает 95% данных)")

        # Качество текста (спецсимволы, URL, капс)
        self.df['has_url'] = self.df['text'].astype(str).apply(lambda x: 1 if 'http' in x.lower() else 0)
        self.df['has_caps'] = self.df['text'].astype(str).apply(lambda x: 1 if x.isupper() and len(x) > 5 else 0)
        self.df['exclamation_count'] = self.df['text'].astype(str).apply(lambda x: x.count('!'))

        print(f"Тексты с URL: {self.df['has_url'].sum()}")
        print(f"Тексты КАПСОМ: {self.df['has_caps'].sum()}")

        return self

    def class_balance_analysis(self):
        """Анализ баланса классов"""
        print("\n⚖️ 3. БАЛАНС КЛАССОВ")
        print("-" * 50)

        class_dist = self.df['label'].value_counts().sort_index()
        total = len(self.df)

        self.report['class_distribution'] = {}

        print(f"{'Класс':<10} {'Кол-во':<10} {'Процент':<10}")
        print("-" * 30)
        for label, count in class_dist.items():
            pct = count / total * 100
            self.report['class_distribution'][int(label)] = {'count': int(count), 'percent': round(pct, 2)}
            print(f"{label:<10} {count:<10} {pct:.2f}%")

        # Проверка на дисбаланс
        max_ratio = class_dist.max() / class_dist.min()
        self.report['imbalance_ratio'] = round(max_ratio, 2)
        print(f"\nКоэффициент дисбаланса: {max_ratio:.2f}")
        if max_ratio > 1.5:
            print("⚠️ Внимание: есть дисбаланс классов!")
        else:
            print("✅ Классы сбалансированы")

        return self

    def visualize(self):
        """Создание визуализаций"""
        print("\n📊 4. ВИЗУАЛИЗАЦИЯ")
        print("-" * 50)

        sns.set_style('whitegrid')

        # 1. Распределение классов
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Class distribution
        ax1 = axes[0, 0]
        sns.countplot(x='label', data=self.df, ax=ax1, palette='Set2')
        ax1.set_title('Распределение классов', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Класс')
        ax1.set_ylabel('Количество')

        # Text length distribution
        ax2 = axes[0, 1]
        sns.histplot(self.df['char_len'], bins=50, kde=True, ax=ax2, color='steelblue')
        ax2.axvline(self.report['recommended_max_len'], color='red', linestyle='--',
                    label=f"95% перцентиль ({self.report['recommended_max_len']})")
        ax2.set_title('Длина текстов (символы)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Длина')
        ax2.set_ylabel('Частота')
        ax2.legend()

        # Boxplot by class
        ax3 = axes[1, 0]
        sns.boxplot(x='label', y='char_len', data=self.df, ax=ax3, palette='Set2')
        ax3.set_title('Длина текстов по классам', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Класс')
        ax3.set_ylabel('Длина (символы)')

        # Word count distribution
        ax4 = axes[1, 1]
        sns.histplot(self.df['word_count'], bins=50, kde=True, ax=ax4, color='coral')
        ax4.set_title('Количество слов', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Слов')
        ax4.set_ylabel('Частота')

        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/eda_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Сохранено: eda_overview.png")

        # 2. WordCloud для каждого класса
        class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        for label, name in class_names.items():
            self._generate_wordcloud(label, name)

        return self

    def _generate_wordcloud(self, label, name):
        """Генерация облака слов для класса"""
        texts = self.df[self.df['label'] == label]['text'].astype(str)
        all_words = ' '.join(texts).lower()

        # Очистка
        words = re.findall(r'[а-яёa-z]+', all_words)
        words = [w for w in words if w not in self.stopwords_ru and len(w) >= self.config['min_word_length']]

        if len(words) > 0:
            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white',
                                  colormap='viridis',
                                  max_words=100).generate(' '.join(words))

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Top Words - {name}', fontsize=16, fontweight='bold')
            plt.savefig(f"{self.config['output_dir']}/wordcloud_{name.lower()}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Сохранено: wordcloud_{name.lower()}.png")

    def split_data(self):
        """Разделение на train/val/test с проверкой на утечки"""
        print("\n🔄 5. РАЗДЕЛЕНИЕ ДАННЫХ")
        print("-" * 50)

        # Стратифицированное разделение
        train_val, self.test = train_test_split(
            self.df,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=self.df['label']
        )

        self.train, self.val = train_test_split(
            train_val,
            test_size=self.config['val_size'],
            random_state=self.config['random_state'],
            stratify=train_val['label']
        )

        # 🔥 ПЕРЕКРЁСТНАЯ ПРОВЕРКА НА УТЕЧКИ (по уникальным текстам)
        train_texts = set(self.train['text'].astype(str).unique())
        val_texts = set(self.val['text'].astype(str).unique())
        test_texts = set(self.test['text'].astype(str).unique())

        leak_train_val = len(train_texts & val_texts)
        leak_train_test = len(train_texts & test_texts)
        leak_val_test = len(val_texts & test_texts)

        self.report['data_leakage'] = {
            'train_val': leak_train_val,
            'train_test': leak_train_test,
            'val_test': leak_val_test
        }

        print(f"Train: {len(self.train):,} ({len(self.train) / len(self.df) * 100:.1f}%)")
        print(f"Val:   {len(self.val):,} ({len(self.val) / len(self.df) * 100:.1f}%)")
        print(f"Test:  {len(self.test):,} ({len(self.test) / len(self.df) * 100:.1f}%)")

        if any([leak_train_val, leak_train_test, leak_val_test]):
            print("\n⚠️ ВНИМАНИЕ: Обнаружена утечка данных!")
            print(f"   Train ↔ Val: {leak_train_val} общих текстов")
            print(f"   Train ↔ Test: {leak_train_test} общих текстов")
            print(f"   Val ↔ Test: {leak_val_test} общих текстов")
        else:
            print("\n✅ Утечек данных не обнаружено — все сплиты чистые!")

        # Проверка распределения классов в сплитах
        print("\n📊 Распределение классов после разделения:")
        for name, data in [('Train', self.train), ('Val', self.val), ('Test', self.test)]:
            dist = data['label'].value_counts(normalize=True).sort_index()
            print(f"{name}: {['{:.1%}'.format(p) for p in dist.values]}")

        return self

    def save_results(self):
        """Сохранение результатов"""
        print("\n💾 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 50)

        # Сохранение датасетов
        self.train.to_csv(f"{self.config['output_dir']}/train.csv", index=False, encoding=self.config['encoding'])
        self.val.to_csv(f"{self.config['output_dir']}/val.csv", index=False, encoding=self.config['encoding'])
        self.test.to_csv(f"{self.config['output_dir']}/test.csv", index=False, encoding=self.config['encoding'])
        print("✅ Датасеты сохранены (train/val/test.csv)")

        # Сохранение отчёта
        with open(f"{self.config['output_dir']}/eda_report.json", 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        print("✅ Отчёт сохранён (eda_report.json)")

        # Сохранение рекомендаций
        self._save_recommendations()

        return self

    def _save_recommendations(self):
        """Генерация текстовых рекомендаций"""
        recs = []
        recs.append("📋 РЕКОМЕНДАЦИИ ПО ДАННЫМ")
        recs.append("=" * 50)
        recs.append(f"1. Используйте max_len = {self.report.get('recommended_max_len', 256)}")

        if self.report.get('imbalance_ratio', 1) > 1.5:
            recs.append("2. ⚠️ Используйте class_weight='balanced' или аугментацию")
        else:
            recs.append("2. ✅ Классы сбалансированы, можно обучать без весов")

        if self.report['duplicates']['text_only'] > 0:
            recs.append(f"3. ✅ Удалено {self.report['duplicates']['text_only']} дубликатов текстов")
        else:
            recs.append("3. ✅ Дубликаты текстов отсутствуют")

        if self.report['empty_texts'] > 0:
            recs.append(f"4. ✅ Удалено {self.report['empty_texts']} пустых текстов")
        else:
            recs.append("4. ✅ Пустые тексты отсутствуют")

        if any(self.report.get('data_leakage', {}).values()):
            recs.append("5. ⚠️ КРИТИЧНО: Устраните утечку данных между сплитами!")
        else:
            recs.append("5. ✅ Утечек данных между сплитами нет")

        recs.append("\n🚀 Следующий шаг: Токенизация и создание DataLoader")

        with open(f"{self.config['output_dir']}/recommendations.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(recs))

        print("✅ Рекомендации сохранены (recommendations.txt)")
        print("\n" + '\n'.join(recs))

    def run(self):
        """Запуск полного пайплайна"""
        print("🎯 ЗАПУСК EDA RUREVIEWS PRO")
        print("=" * 60)

        self.load_data(self.config['input_file'])
        self.basic_quality_check()
        self.text_analysis()
        self.class_balance_analysis()
        self.visualize()
        self.split_data()
        self.save_results()

        print("\n" + "=" * 60)
        print("✅ EDA ЗАВЕРШЕНА УСПЕШНО!")
        print(f"📁 Все результаты в папке: {self.config['output_dir']}/")
        print("=" * 60)


# =============================================================================
# 🎯 ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    eda = ReviewsEDA(CONFIG)
    eda.run()