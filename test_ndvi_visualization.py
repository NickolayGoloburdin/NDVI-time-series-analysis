import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Импортируем класс NDVIForecaster из основного файла
from ndvi_ts_lstm import NDVIForecaster, LSTMModel

# Настройки для визуализации
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 12})
colors = sns.color_palette("Set2", 10)

def load_config(config_path="configs/config_ndvi.json"):
    """Загрузка конфигурации из JSON-файла"""
    if not os.path.exists(config_path):
        # Если конфиг не найден, создаем тестовый конфиг
        test_config = {
            "coordinates": [
                [51.52945, 38.95530],
                [51.52378, 38.95391],
                [51.52216, 38.95444],
                [51.52405, 38.96799],
                [51.53180, 38.96918],
                [51.52958, 38.95528]
            ],
            "start_date": "2022-03-21",
            "end_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "n_steps_in": 72,
            "n_steps_out": 18,
            "percentile": 55,
            "bimonthly_period": "2M",
            "spline_smoothing": 0.96
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(test_config, f, indent=4)
        print(f"Создан тестовый конфигурационный файл: {config_path}")
        return test_config
    
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Конфигурация загружена из: {config_path}")
    return config

def test_ndvi_forecaster():
    """Основная функция для тестирования модели прогнозирования NDVI"""
    # Загрузка конфигурации
    config = load_config()
    
    # Создание экземпляра класса NDVIForecaster
    forecaster = NDVIForecaster(
        coordinates=config["coordinates"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        n_steps_in=config["n_steps_in"],
        n_steps_out=config["n_steps_out"],
        percentile=config["percentile"],
        bimonthly_period=config["bimonthly_period"],
        spline_smoothing=config["spline_smoothing"]
    )
    
    print("1. Инициализация Google Earth Engine...")
    forecaster.initialize_ee()
    
    # ЭТАП 1: Получение и визуализация данных NDVI
    print("2. Получение данных NDVI...")
    ndvi_timeseries = forecaster.get_ndvi_timeseries(config["start_date"], config["end_date"])
    ndvi_df = forecaster.extract_ndvi_data(ndvi_timeseries)
    
    # Визуализация NDVI
    plt.figure(figsize=(14, 7))
    plt.plot(ndvi_df['Date'], ndvi_df['NDVI'], 'o-', color=colors[0], label='NDVI (после интерполяции и фильтрации)')
    plt.title('Временной ряд NDVI', fontsize=16)
    plt.xlabel('Дата', fontsize=14)
    plt.ylabel('Значение NDVI', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/ndvi_timeseries.png')
    print("График NDVI сохранен в images/ndvi_timeseries.png")
    plt.close()
    
    # ЭТАП 2: Получение и визуализация данных о погоде
    print("3. Получение погодных данных...")
    weather_df = forecaster.get_weather_data(config["start_date"], config["end_date"])
    
    # Визуализация погодных данных
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # График температуры
    axs[0].plot(weather_df['Date'], weather_df['TempMax'], '-', color=colors[1], label='Максимальная температура')
    axs[0].plot(weather_df['Date'], weather_df['TempMin'], '-', color=colors[2], label='Минимальная температура')
    axs[0].set_ylabel('Температура (°C)', fontsize=14)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # График относительной влажности
    axs[1].plot(weather_df['Date'], weather_df['RelativeHumidity'], '-', color=colors[3], label='Относительная влажность')
    axs[1].set_ylabel('Влажность (%)', fontsize=14)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # График осадков
    axs[2].bar(weather_df['Date'], weather_df['Precipitation'], color=colors[4], label='Осадки', alpha=0.7, width=3)
    axs[2].set_ylabel('Осадки (мм)', fontsize=14)
    axs[2].set_xlabel('Дата', fontsize=14)
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Погодные параметры', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/weather_data.png')
    print("График погодных данных сохранен в images/weather_data.png")
    plt.close()
    
    # ЭТАП 3: Объединение и подготовка данных
    print("4. Объединение и подготовка данных...")
    forecaster.merge_data()
    forecaster.prepare_data()
    
    # Визуализация объединенных данных
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # NDVI и сглаженный NDVI
    axs[0].plot(forecaster.train_df['Date'], forecaster.train_df['NDVI'], 'o-', color=colors[0], 
                label='NDVI (фильтр+интерполяция)', alpha=0.7, markersize=4)
    axs[0].plot(forecaster.train_df['Date'], forecaster.train_df['NDVI_Smoothed'], '-', color=colors[5], 
                label='NDVI (сглаженный)', linewidth=2)
    axs[0].set_ylabel('NDVI', fontsize=14)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Температура и осадки
    ax1 = axs[1]
    ax1.plot(forecaster.train_df['Date'], forecaster.train_df['TempMax'], '-', color=colors[1], label='Макс. температура')
    ax1.plot(forecaster.train_df['Date'], forecaster.train_df['TempMin'], '-', color=colors[2], label='Мин. температура')
    ax1.set_ylabel('Температура (°C)', fontsize=14)
    ax1.set_xlabel('Дата', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Накладываем график осадков
    ax2 = ax1.twinx()
    ax2.bar(forecaster.train_df['Date'], forecaster.train_df['Precipitation'], 
            color=colors[4], label='Осадки', alpha=0.4, width=3)
    ax2.set_ylabel('Осадки (мм)', fontsize=14)
    ax2.legend(loc='upper right')
    
    plt.suptitle('Объединенные данные NDVI и погоды', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/merged_data.png')
    print("График объединенных данных сохранен в images/merged_data.png")
    plt.close()
    
    # ЭТАП 4: Обучение модели или загрузка весов
    # Проверяем наличие весов моделей
    weights_path_original = "weights/model_weights_original.pth"
    weights_path_filtered = "weights/model_weights_filtered.pth"
    
    if os.path.exists(weights_path_original) and os.path.exists(weights_path_filtered):
        print("5. Загружаем предобученные веса моделей...")
        # Подготавливаем данные
        train_data, smoothed_data = forecaster.scale_data()
        X_train, y_train = forecaster.split_sequences(train_data, config["n_steps_in"], config["n_steps_out"])
        n_features = X_train.shape[2]
        
        # Создаем модели
        forecaster.model_original = forecaster.create_model(n_features)
        forecaster.model_filtered = forecaster.create_model(n_features)
        
        # Загружаем веса
        forecaster.load_model_weights(forecaster.model_original, "original")
        forecaster.load_model_weights(forecaster.model_filtered, "filtered")
    else:
        print("5. Предобученные модели не найдены")
        raise Exception("Предобученные модели не найдены")
    
    # ЭТАП 5: Прогнозирование и оценка
    print("6. Прогнозирование NDVI...")
    test_pred_original, test_pred_smoothed, forecast_pred_original, forecast_pred_smoothed = forecaster.forecast()
    
    # Создаем сводный график с прогнозом
    # Определяем даты для прогноза
    if forecaster.case in [1, 2] and forecaster.test_df is not None:
        # Для случаев с тестовыми данными
        test_dates = forecaster.test_df['Date'][:len(test_pred_smoothed)].values
    
    # Даты для будущего прогноза
    forecast_dates = pd.date_range(start=forecaster.forecast_dates[0], periods=len(forecast_pred_smoothed), freq='5D')
    
    plt.figure(figsize=(14, 7))
    
    # Тренировочные данные
    plt.plot(forecaster.train_df['Date'], forecaster.train_df['NDVI'], 'o-', color=colors[0], 
             label='NDVI (фильтр+интерполяция)', alpha=0.7, markersize=4)
    plt.plot(forecaster.train_df['Date'], forecaster.train_df['NDVI_Smoothed'], '-', color=colors[5], 
             label='NDVI (сглаженный)', linewidth=2)
    
    # Тестовые данные и прогнозы
    if forecaster.case in [1, 2] and forecaster.test_df is not None and not forecaster.test_df.empty:
        plt.plot(forecaster.test_df['Date'], forecaster.test_df['NDVI'], 'o-', color='lightblue', 
                 label='Фактический NDVI (тест)', alpha=0.7, markersize=4)
        plt.plot(forecaster.test_df['Date'], forecaster.test_df['NDVI_Smoothed'], '-', color='blue', 
                 label='Фактический NDVI (сглаженный, тест)', linewidth=2)
        
        if test_pred_smoothed is not None and len(test_pred_smoothed) > 0:
            plt.plot(test_dates, test_pred_smoothed, '--', color='red', 
                     label='LSTM прогноз (сглаженный)', linewidth=2)
    
    # Отображение будущего прогноза
    if forecaster.case in [2, 3]:
        if forecast_pred_smoothed is not None and len(forecast_pred_smoothed) > 0:
            plt.plot(forecast_dates, forecast_pred_smoothed, '--', color='red', 
                     label='LSTM прогноз (сглаженный)', linewidth=2)
        
        # Отображение исторического базового прогноза
        plt.plot(forecaster.baseline_df['Date'], forecaster.baseline_df['Historical_Avg_NDVI_Smoothed'], 
                 '--', color='purple', label='Исторический базовый прогноз', linewidth=2)
    
    # Добавление вертикальной линии, отделяющей историю от прогноза
    current_date = forecaster.current_date
    plt.axvline(x=current_date, color='black', linestyle='--', label='Текущая дата')
    
    plt.title(f'Прогноз NDVI - Сценарий {forecaster.case}', fontsize=16)
    plt.xlabel('Дата', fontsize=14)
    plt.ylabel('NDVI', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images/ndvi_forecast_case_{forecaster.case}.png')
    print(f"Прогноз NDVI сохранен в images/ndvi_forecast_case_{forecaster.case}.png")
    plt.close()
    
    # ЭТАП 6: Оценка точности модели (если есть тестовые данные)
    if forecaster.case in [1, 2] and forecaster.test_df is not None and not forecaster.test_df.empty:
        print("7. Оценка точности модели...")
        
        # Ограничиваем тестовые данные размером прогноза
        test_actual = forecaster.test_df['NDVI_Smoothed'].values[:len(test_pred_smoothed)]
        
        if len(test_actual) > 0 and len(test_pred_smoothed) > 0:
            # Вычисляем метрики
            mae = mean_absolute_error(test_actual, test_pred_smoothed)
            rmse = np.sqrt(mean_squared_error(test_actual, test_pred_smoothed))
            r2 = r2_score(test_actual, test_pred_smoothed)
            
            # Создаем график сравнения прогноза с фактическими данными
            plt.figure(figsize=(14, 7))
            plt.plot(test_dates[:len(test_actual)], test_actual, 'o-', color='blue', 
                    label='Фактический NDVI', alpha=0.7)
            plt.plot(test_dates[:len(test_pred_smoothed)], test_pred_smoothed, 'o--', color='red', 
                    label='Прогноз NDVI', alpha=0.7)
            
            # Добавляем метрики в заголовок
            plt.title(f'Сравнение прогноза с фактическими данными\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}', 
                    fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('NDVI', fontsize=14)
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('images/forecast_accuracy.png')
            print("График точности прогноза сохранен в images/forecast_accuracy.png")
            plt.close()
            
            # Создаем график разброса (scatter plot)
            plt.figure(figsize=(10, 10))
            plt.scatter(test_actual, test_pred_smoothed, color='blue', alpha=0.7)
            plt.plot([min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], 
                    'r--', linewidth=2)
            plt.title('Scatter Plot: Фактический vs Прогнозируемый NDVI', fontsize=16)
            plt.xlabel('Фактический NDVI', fontsize=14)
            plt.ylabel('Прогнозируемый NDVI', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('images/scatter_plot.png')
            print("График разброса сохранен в images/scatter_plot.png")
            plt.close()
        else:
            print("Недостаточно данных для оценки точности модели")
    
    # ЭТАП 7: Визуализация входных последовательностей и механизма внимания
    print("8. Визуализация входных параметров модели...")
    
    # Создаем график входных последовательностей
    plt.figure(figsize=(14, 10))
    
    # Создаем пример входной последовательности для визуализации
    input_sequence = forecaster.train_df.tail(config["n_steps_in"])[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']]
    
    # Масштабируем последовательность
    input_scaled = forecaster.scaler_x.transform(input_sequence)
    
    # Визуализация входных функций
    x_range = np.arange(len(input_sequence))
    
    plt.subplot(4, 1, 1)
    plt.plot(x_range, input_sequence['TempMin'], 'o-', color=colors[2], label='Мин. температура')
    plt.plot(x_range, input_sequence['TempMax'], 'o-', color=colors[1], label='Макс. температура')
    plt.title('Входная последовательность для LSTM модели', fontsize=16)
    plt.ylabel('Температура (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(4, 1, 2)
    plt.plot(x_range, input_sequence['RelativeHumidity'], 'o-', color=colors[3], label='Относительная влажность')
    plt.ylabel('Влажность (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(4, 1, 3)
    plt.bar(x_range, input_sequence['Precipitation'], color=colors[4], label='Осадки', alpha=0.7, width=0.8)
    plt.ylabel('Осадки (мм)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(4, 1, 4)
    plt.plot(x_range, input_sequence.index.map(lambda x: forecaster.train_df.loc[x, 'NDVI']), 
             'o-', color=colors[0], label='NDVI')
    plt.plot(x_range, input_sequence.index.map(lambda x: forecaster.train_df.loc[x, 'NDVI_Smoothed']), 
             'o-', color=colors[5], label='NDVI (сглаженный)')
    plt.ylabel('NDVI', fontsize=12)
    plt.xlabel('Шаг последовательности', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/input_sequence.png')
    print("График входной последовательности сохранен в images/input_sequence.png")
    plt.close()
    
    print("Тестирование модели NDVI-прогнозирования завершено!")
    print("Все графики сохранены в директории 'images/'")

def explain_model():
    """Создает информационный график со схемой модели"""
    # Создаем схематичное изображение LSTM модели с механизмом внимания
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Скрываем оси
    ax.axis('off')
    
    # Заголовок
    ax.text(0.5, 0.95, 'Архитектура LSTM модели с механизмом внимания', 
            fontsize=18, weight='bold', ha='center', va='top')
    
    # Добавляем описание модели
    description = """
    Принцип работы модели NDVI-прогнозирования:
    
    1. Входные данные:
       • Погодные параметры: мин./макс. температура, влажность, осадки
       • Исторические значения NDVI с интервалом 5 дней
    
    2. Предобработка данных:
       • Фильтрация выбросов с использованием процентилей по 2-месячным периодам
       • Линейная интерполяция для заполнения пропущенных значений
       • Сглаживание с использованием сплайнов для уменьшения шума
       • Масштабирование всех значений в диапазон [0,1]
    
    3. Архитектура LSTM:
       • Входной слой: погодные параметры + значения NDVI
       • LSTM слои с дропаутом для предотвращения переобучения
       • Механизм многоголового внимания для определения значимости входных данных
       • Полносвязный слой для выходных значений NDVI
    
    4. Обучение:
       • Формирование временных последовательностей методом "скользящего окна"
       • Оптимизация с помощью Adam с клиппингом градиентов
       • Функция потерь: Mean Squared Error (MSE)
    
    5. Прогнозирование:
       • Входная последовательность: n_steps_in шагов (обычно 72 точки = 360 дней)
       • Выходная последовательность: n_steps_out шагов (обычно 18 точек = 90 дней)
       • Прогноз генерируется с шагом 5 дней
    
    6. Проверка и валидация:
       • Сравнение с фактическими значениями NDVI
       • Оценка точности: MAE, RMSE, R²
       • Сравнение с историческим средним (baseline)
    """
    
    ax.text(0.5, 0.5, description, fontsize=14, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('images/model_explanation.png', dpi=300, bbox_inches='tight')
    print("Схема модели сохранена в images/model_explanation.png")
    plt.close()

if __name__ == "__main__":
    # Создаем директорию для изображений, если ее нет
    os.makedirs('images', exist_ok=True)
    
    # Создаем объяснение модели
    explain_model()
    
    # Запускаем тестирование
    test_ndvi_forecaster() 