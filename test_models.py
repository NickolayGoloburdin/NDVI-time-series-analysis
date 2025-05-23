#!/usr/bin/env python3
"""
Система тестирования моделей NDVI
===============================

Тестирует обученные модели на реальных данных и сравнивает прогнозы с фактическими значениями.
"""

import os
import json
import warnings
from typing import Dict, Tuple, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ndvi_ts_lstm import NDVIForecaster, DataManager, LSTMModel, ModelConfig

warnings.filterwarnings("ignore")

# ===================== КОНСТАНТЫ =====================
DEFAULT_CONFIG_PATH = "configs/config_ndvi.json"
OUTPUT_DIR = "results"
ORIGINAL_WEIGHTS_PATH = "weights/model_weights_original.pth"
FILTERED_WEIGHTS_PATH = "weights/model_weights_filtered.pth" 
FORECAST_STEP_DAYS = 5

# ===================== ГЛАВНЫЙ КЛАСС ТЕСТИРОВАНИЯ =====================

class NDVIModelTester:
    """Класс для тестирования и валидации модели NDVI."""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Инициализация тестера модели.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.forecaster: Optional[NDVIForecaster] = None
        self.data_manager: Optional[DataManager] = None
        self._ensure_output_dir()
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию из JSON файла."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл конфигурации {config_path} не найден")
        except json.JSONDecodeError:
            raise ValueError(f"Некорректный JSON в файле {config_path}")
    
    def _ensure_output_dir(self) -> None:
        """Создает директорию для выходных файлов, если её нет."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _validate_config(self) -> None:
        """Проверяет и валидирует параметры конфигурации."""
        required_params = [
            "coordinates", "n_steps_in", "n_steps_out", "test_start_date", "test_end_date"
        ]
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Отсутствует обязательный параметр в конфигурации: {param}")
        
        # Проверяем наличие тестовых параметров
        test_start = self.config["test_start_date"]
        test_end = self.config["test_end_date"]
        
        print(f"✓ Тестовые параметры: {test_start} - {test_end}")
        
        total_forecast_days = self.config['n_steps_out'] * FORECAST_STEP_DAYS
        print(f"✓ Прогноз: {self.config['n_steps_out']} шагов × {FORECAST_STEP_DAYS} дней = {total_forecast_days} дней")
    
    def setup_forecaster(self) -> NDVIForecaster:
        """
        Создает и настраивает forecaster с данными.
        
        Returns:
            Настроенный объект NDVIForecaster
        """
        print("Инициализация forecaster...")
        
        # Создаем тестовую конфигурацию
        test_config = self.config.copy()
        test_config["start_date"] = self.config["test_start_date"]
        test_config["end_date"] = self.config["test_end_date"]
        
        # Сохраняем временную конфигурацию
        temp_config_path = "temp_test_config.json"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        
        try:
            # Создаем forecaster с временной конфигурацией
            self.forecaster = NDVIForecaster(temp_config_path)
            
            print("Загрузка и обработка данных...")
            self.forecaster.load_and_process_data()
            
            # Создаем отдельный data_manager для получения данных прогноза
            coordinates = [tuple(coord) for coord in self.config["coordinates"]]
            self.data_manager = DataManager(coordinates)
            
            return self.forecaster
            
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def load_trained_models(self) -> None:
        """Загружает веса обученных моделей."""
        if self.forecaster is None:
            raise ValueError("Forecaster должен быть инициализирован перед загрузкой моделей")
        
        # Проверяем наличие файлов весов
        if not (os.path.exists(ORIGINAL_WEIGHTS_PATH) and os.path.exists(FILTERED_WEIGHTS_PATH)):
            raise FileNotFoundError(
                "Веса моделей не найдены. Сначала обучите модель, запустив ndvi_ts_lstm.py"
            )
        
        print("Загрузка весов моделей...")
        
        # Подготовка данных для определения размерности входа
        X, _, _, input_size = self.forecaster.data_manager.prepare_sequences(
            self.forecaster.merged_df, 
            self.forecaster.n_steps_in, 
            self.forecaster.n_steps_out
        )
        
        # Создание моделей
        model_config = ModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_original = LSTMModel(input_size, model_config).to(device)
        model_filtered = LSTMModel(input_size, model_config).to(device)
        
        # Загрузка весов
        model_original.load_state_dict(
            torch.load(ORIGINAL_WEIGHTS_PATH, map_location=device)
        )
        model_filtered.load_state_dict(
            torch.load(FILTERED_WEIGHTS_PATH, map_location=device)
        )
        
        # Перевод в режим оценки
        model_original.eval()
        model_filtered.eval()
        
        self.forecaster.model_original = model_original
        self.forecaster.model_filtered = model_filtered
        
        print("✅ Веса моделей успешно загружены!")
    
    def generate_forecast(self) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Генерирует прогноз NDVI на заданный период.
        
        Returns:
            Кортеж из дат прогноза и предсказанных значений NDVI
        """
        if self.forecaster is None:
            raise ValueError("Forecaster должен быть инициализирован")
        
        print("🔮 Генерация прогноза...")
        
        # Создание дат для прогноза
        forecast_start = pd.Timestamp(self.config["test_end_date"]) + pd.Timedelta(days=FORECAST_STEP_DAYS)
        forecast_dates = pd.date_range(
            start=forecast_start,
            periods=self.config["n_steps_out"],
            freq=f'{FORECAST_STEP_DAYS}D'
        )
        
        print(f"📅 Прогноз: {forecast_dates[0]} - {forecast_dates[-1]}")
        
        # Используем последние данные для прогноза
        recent_data = self.forecaster.merged_df.tail(self.forecaster.n_steps_in).copy()
        
        # Масштабируем последние данные
        features = ['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation', 'NDVI_Smoothed']
        recent_scaled = self.forecaster.data_manager.scaler_x.fit_transform(
            recent_data[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']]
        )
        recent_ndvi_scaled = self.forecaster.data_manager.scaler_y_smoothed.transform(
            recent_data[['NDVI_Smoothed']]
        )
        
        # Объединяем признаки и NDVI
        input_sequence = np.hstack([recent_scaled, recent_ndvi_scaled])
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  # Добавляем batch dimension
        
        # Генерируем прогноз
        with torch.no_grad():
            forecast_tensor = self.forecaster.model_filtered(input_tensor)
            forecast_scaled = forecast_tensor.cpu().numpy().flatten()
        
        # Создаем прогноз для всех шагов (простое повторение одного предсказания)
        forecast_predictions = np.repeat(forecast_scaled[0], self.config["n_steps_out"])
        
        # Обратное масштабирование
        forecast_ndvi = self.forecaster.data_manager.scaler_y_smoothed.inverse_transform(
            forecast_predictions.reshape(-1, 1)
        ).flatten()
        
        # Диагностика предсказанных значений
        print(f"📊 Диагностика прогнозных значений NDVI:")
        print(f"   📈 Диапазон: {forecast_ndvi.min():.3f} - {forecast_ndvi.max():.3f}")
        print(f"   📊 Среднее: {forecast_ndvi.mean():.3f}")
        print(f"   📋 Значения: {[f'{v:.3f}' for v in forecast_ndvi]}")
        
        if forecast_ndvi.max() < 0:
            print("   ⚠️  ВНИМАНИЕ: Все прогнозные значения NDVI отрицательные!")
        elif forecast_ndvi.max() > 0.3:
            print("   ✅ Хорошие значения - прогнозируется растительность")
        else:
            print("   ⚠️  Низкие значения NDVI")
        
        return forecast_dates, forecast_ndvi
    
    def get_real_ndvi_for_dates(self, forecast_dates: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Получает ежедневные реальные значения NDVI в расширенном диапазоне для лучшего сравнения.
        
        Args:
            forecast_dates: Даты прогноза для определения диапазона
            
        Returns:
            Кортеж из дат реальных данных и массива значений NDVI
        """
        if self.data_manager is None:
            raise ValueError("DataManager должен быть инициализирован")
        
        print("🛰️  Получение реальных значений NDVI...")
        
        # Расширяем диапазон: 30 дней до прогноза + весь период прогноза
        start_date = forecast_dates[0] - pd.Timedelta(days=30)
        end_date = forecast_dates[-1]
        
        print(f"📅 Запрос реальных данных: {start_date.date()} - {end_date.date()}")
        
        # Получаем реальные данные NDVI для периода прогноза
        real_ndvi_df = self.data_manager.get_ndvi_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if real_ndvi_df.empty:
            print("⚠️  Нет реальных данных NDVI для периода сравнения")
            return pd.DatetimeIndex([]), np.array([])
        
        print(f"✅ Получено {len(real_ndvi_df)} точек реальных данных NDVI")
        print(f"📊 Диапазон реальных значений: {real_ndvi_df['NDVI'].min():.3f} - {real_ndvi_df['NDVI'].max():.3f}")
        
        return real_ndvi_df['Date'], real_ndvi_df['NDVI'].values
    
    def calculate_metrics(self, forecast_dates: pd.DatetimeIndex, forecast_values: np.ndarray,
                         real_dates: pd.DatetimeIndex, real_values: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет метрики качества прогноза.
        
        Args:
            forecast_dates: Даты прогноза
            forecast_values: Предсказанные значения NDVI
            real_dates: Даты реальных данных
            real_values: Реальные значения NDVI
            
        Returns:
            Словарь с метриками качества
        """
        print("📊 Расчет метрик качества прогноза...")
        
        # Создаем DataFrame для удобной работы
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'NDVI_Forecast': forecast_values
        })
        
        real_df = pd.DataFrame({
            'Date': real_dates,
            'NDVI_Real': real_values
        })
        
        # Объединяем данные по датам
        merged_df = pd.merge(forecast_df, real_df, on='Date', how='inner')
        
        if len(merged_df) == 0:
            print("⚠️  Нет пересекающихся дат для расчета метрик")
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'r2': float('nan'),
                'matched_points': 0,
                'forecast_mean': np.mean(forecast_values),
                'real_mean': float('nan'),
                'forecast_range': (np.min(forecast_values), np.max(forecast_values)),
                'real_range': (float('nan'), float('nan'))
            }
        
        # Извлекаем сопоставленные значения
        forecast_matched = merged_df['NDVI_Forecast'].values
        real_matched = merged_df['NDVI_Real'].values
        
        # Вычисляем метрики
        mae = mean_absolute_error(real_matched, forecast_matched)
        rmse = np.sqrt(mean_squared_error(real_matched, forecast_matched))
        r2 = r2_score(real_matched, forecast_matched)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'matched_points': len(merged_df),
            'forecast_mean': np.mean(forecast_values),
            'real_mean': np.mean(real_values),
            'forecast_range': (np.min(forecast_values), np.max(forecast_values)),
            'real_range': (np.min(real_values), np.max(real_values))
        }
        
        print(f"✅ Метрики рассчитаны для {len(merged_df)} сопоставленных точек:")
        print(f"   🎯 MAE: {mae:.4f}")
        print(f"   🎯 RMSE: {rmse:.4f}")
        print(f"   🎯 R²: {r2:.4f}")
        
        return metrics
    
    def plot_comparison(self, forecast_dates: pd.DatetimeIndex, forecast: np.ndarray, 
                       real_dates: pd.DatetimeIndex, real: np.ndarray) -> None:
        """
        Создает и сохраняет график сравнения прогноза и реальных данных как PNG изображение.
        
        Args:
            forecast_dates: Даты прогноза
            forecast: Предсказанные значения NDVI
            real_dates: Даты реальных данных
            real: Реальные значения NDVI
        """
        print("📈 Создание графика сравнения...")
        
        fig = go.Figure()
        
        # Разделяем реальные данные на исторические и данные для сравнения
        forecast_start = forecast_dates[0]
        
        historical_mask = real_dates < forecast_start
        comparison_mask = real_dates >= forecast_start
        
        # Исторические данные (серый цвет)
        if np.any(historical_mask):
            fig.add_trace(go.Scatter(
                x=real_dates[historical_mask],
                y=real[historical_mask],
                mode='markers+lines',
                name='Исторические данные',
                line=dict(color='gray', width=2),
                marker=dict(size=4, color='gray'),
                opacity=0.7
            ))
        
        # Реальные данные в период прогноза (синий)
        if np.any(comparison_mask):
            fig.add_trace(go.Scatter(
                x=real_dates[comparison_mask],
                y=real[comparison_mask],
                mode='markers+lines',
                name='Реальные данные (для сравнения)',
                line=dict(color='blue', width=3),
                marker=dict(size=6, color='blue')
            ))
        
        # Прогнозные данные (красный)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='markers+lines',
            name='Прогноз модели',
            line=dict(color='red', width=3),
            marker=dict(size=8, color='red', symbol='diamond')
        ))
        
        # Настройка макета
        fig.update_layout(
            title={
                'text': 'Сравнение прогноза NDVI с реальными данными',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Дата',
            yaxis_title='NDVI',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            width=1200,
            height=600
        )
        
        # Добавляем горизонтальные линии для ориентира
        fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.3, 
                     annotation_text="Порог здоровой растительности", annotation_position="left")
        
        # Сохранение графика как PNG изображение
        output_path = os.path.join(OUTPUT_DIR, "ndvi_forecast_comparison.png")
        fig.write_image(output_path, width=1200, height=600, scale=2)
        
        print(f"✅ График сохранен как изображение: {output_path}")
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Сохраняет метрики в JSON файл.
        
        Args:
            metrics: Словарь с метриками
        """
        output_path = os.path.join(OUTPUT_DIR, "forecast_metrics.json")
        
        # Добавляем метаданные
        full_metrics = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_metrics, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Метрики сохранены: {output_path}")
    
    def run_full_test(self) -> None:
        """Выполняет полный цикл тестирования модели."""
        print("🚀 ЗАПУСК ПОЛНОГО ТЕСТИРОВАНИЯ МОДЕЛИ")
        print("=" * 60)
        
        try:
            # Инициализация и настройка
            self.setup_forecaster()
            self.load_trained_models()
            
            # Генерация прогноза
            forecast_dates, forecast_values = self.generate_forecast()
            
            # Получение реальных данных
            real_dates, real_values = self.get_real_ndvi_for_dates(forecast_dates)
            
            # Расчет метрик
            metrics = self.calculate_metrics(
                forecast_dates, forecast_values,
                real_dates, real_values
            )
            
            # Создание графика
            self.plot_comparison(
                forecast_dates, forecast_values,
                real_dates, real_values
            )
            
            # Сохранение результатов
            self.save_metrics(metrics)
            
            print("\n" + "=" * 60)
            print("✅ ТЕСТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            print(f"📁 Результаты сохранены в: {OUTPUT_DIR}/")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ОШИБКА: {e}")
            raise

# ===================== ГЛАВНАЯ ФУНКЦИЯ =====================

def main():
    """Главная функция для запуска тестирования."""
    print("Запуск тестирования модели NDVI...")
    
    try:
        tester = NDVIModelTester()
        tester.run_full_test()
        
    except KeyboardInterrupt:
        print("\n⏹️  Тестирование прервано пользователем")
    except Exception as e:
        print(f"\nОшибка при тестировании модели: {e}")
        raise

if __name__ == "__main__":
    main() 