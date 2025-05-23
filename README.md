# 🌱 NDVI Time Series Analysis

## 📋 Описание

Модульная система для анализа временных рядов NDVI (Normalized Difference Vegetation Index) с использованием глубокого обучения. Проект предназначен для прогнозирования состояния растительности на основе спутниковых данных Sentinel-2 и метеорологической информации.

## ✨ Особенности

- 🛰️ **Интеграция с Google Earth Engine** для получения данных Sentinel-2
- 🌤️ **Автоматическое получение погодных данных** через Open-Meteo API
- 🧠 **LSTM с Multi-Head Attention** для точного прогнозирования
- 📊 **Интерактивная визуализация** результатов с помощью Plotly
- 🔧 **Модульная архитектура** с возможностью расширения
- 🐛 **Подробная система отладки** с эмодзи-логированием
- ⚡ **Автоматическая обработка данных** (фильтрация облаков, интерполяция)

## 🏗️ Архитектура

### Основные компоненты:

- **`DebugLogger`** - Система отладки с информативными сообщениями
- **`ConfigManager`** - Управление конфигурацией проекта
- **`DataManager`** - Получение и обработка данных NDVI и погоды
- **`LSTMModel`** - Нейронная сеть с LSTM и механизмом внимания
- **`ModelTrainer`** - Обучение и сохранение моделей
- **`NDVIForecaster`** - Главный оркестратор системы

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка Google Earth Engine

Скопируйте файл с примером и заполните своими данными:

```bash
cp key.json.example key.json
```

Отредактируйте `key.json` файл, добавив ваши данные Google Earth Engine Service Account:

```json
{
  "type": "service_account",
  "project_id": "ваш-project-id",
  "private_key_id": "ваш-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nВАШ_ПРИВАТНЫЙ_КЛЮЧ\n-----END PRIVATE KEY-----\n",
  "client_email": "ваш-service-account@ваш-project.iam.gserviceaccount.com",
  "client_id": "ваш-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/ваш-service-account%40ваш-project.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

> ⚠️ **Важно:** Файл `key.json` содержит секретные данные и не должен попадать в систему контроля версий!

### 3. Настройка конфигурации

Создайте или отредактируйте файл `configs/config_ndvi.json`:

```json
{
    "coordinates": [
        [51.661535, 39.200287],
        [51.661535, 39.203580],
        [51.659021, 39.203580],
        [51.659021, 39.200287]
    ],
    "start_date": "2023-01-01",
    "end_date": "2023-07-20",
    "n_steps_in": 10,
    "n_steps_out": 5,
    "percentile_filter": 5,
    "bimonthly_period": "2M",
    "spline_smoothing": 0.9
}
```

### 4. Запуск обучения

```bash
python ndvi_ts_lstm.py
```

### 5. Тестирование моделей

```bash
python test_models.py
```

## 📁 Структура проекта

```
NDVI-time-series-analysis/
├── 📄 README.md                    # Данный файл
├── 📄 requirements.txt             # Python зависимости
├── 📄 key.json.example             # Пример ключа Google Earth Engine
├── 📄 .gitignore                   # Игнорируемые файлы
├── 🐍 ndvi_ts_lstm.py             # Основной модуль системы
├── 🐍 test_models.py              # Тестирование моделей
├── 📁 configs/                     # Конфигурационные файлы
│   └── 📄 config_ndvi.json        # Параметры анализа
├── 📁 weights/                     # Сохранённые веса моделей
│   ├── 📄 model_weights_original.pth
│   └── 📄 model_weights_filtered.pth
└── 📁 results/                     # Результаты анализа
    ├── 📄 forecast_metrics.json
    └── 🖼️ ndvi_forecast_comparison.png
```

## ⚙️ Конфигурация

### Основные параметры:

- **`coordinates`** - Полигон области анализа (WGS84)
- **`start_date/end_date`** - Временной период анализа
- **`n_steps_in`** - Количество временных шагов для обучения
- **`n_steps_out`** - Количество шагов прогноза
- **`percentile_filter`** - Фильтр выбросов (в процентилях)
- **`spline_smoothing`** - Параметр сглаживания данных

### Параметры модели (ModelConfig):

- **`LSTM_UNITS`** - Количество LSTM нейронов (244)
- **`NUM_LAYERS`** - Количество слоёв (1)
- **`DROPOUT_RATE`** - Коэффициент dropout (0.29)
- **`LEARNING_RATE`** - Скорость обучения (0.0018)
- **`BATCH_SIZE`** - Размер батча (128)
- **`EPOCHS`** - Количество эпох (200)

## 📊 Результаты

Система генерирует:

1. **Графики-изображения** - PNG файлы с визуализацией результатов
2. **Метрики точности** - JSON файлы с показателями качества
3. **Сохранённые модели** - PyTorch веса для повторного использования
4. **Отладочная информация** - Подробные логи процесса

### Метрики качества:

- **MAE** (Mean Absolute Error) - Средняя абсолютная ошибка
- **MSE** (Mean Squared Error) - Среднеквадратичная ошибка
- **RMSE** (Root MSE) - Корень из среднеквадратичной ошибки
- **R²** - Коэффициент детерминации

## 🔒 Безопасность

- 🔐 Секретные ключи хранятся в файле `key.json`
- 📝 Файл `key.json` добавлен в `.gitignore`
- 🗃️ Оригинальные ключи сохраняются в `key.json.backup`
- 🚫 Никогда не коммитьте файлы с реальными ключами!

## 🛠️ Разработка

### Структура кода:

```python
# Система отладки
DebugLogger.log_ndvi_stats(ndvi_values, "API")

# Управление конфигурацией
config = ConfigManager.load_config()

# Получение данных
data_manager = DataManager(coordinates)
ndvi_df = data_manager.get_ndvi_data(start_date, end_date)

# Обучение модели
trainer = ModelTrainer(ModelConfig())
model = trainer.train_model(model, X, y, "Original")
```

### Добавление новых функций:

1. Наследуйтесь от базовых классов
2. Используйте `DebugLogger` для отладки
3. Следуйте типизации TypeHints
4. Документируйте функции на русском языке

## 📈 Планы развития

- [ ] 🌍 Поддержка нескольких регионов
- [ ] 📡 Интеграция с другими спутниками (Landsat, MODIS)
- [ ] 🤖 Автоматический подбор гиперпараметров
- [ ] 🌐 Web-интерфейс для визуализации
- [ ] 📱 API для внешних приложений
- [ ] 🔄 Потоковая обработка данных в реальном времени

## 🤝 Вклад в проект

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📜 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для деталей.

## 📞 Контакты

- 📧 Email: your-email@example.com
- 💼 LinkedIn: your-linkedin-profile
- 🐱 GitHub: your-github-profile

## 🙏 Благодарности

- Google Earth Engine за предоставление спутниковых данных
- Open-Meteo за метеорологические данные
- PyTorch и Plotly за отличные библиотеки
- Сообществу открытого ПО за вдохновение

---

⭐ **Поставьте звезду, если проект был полезен!** ⭐ 