"""
NDVI Time Series Analysis and Forecasting System
==============================================

Модульная система для анализа временных рядов NDVI с использованием LSTM и multi-head attention.

Основные компоненты:
- DataManager: управление данными NDVI и погоды
- ModelTrainer: обучение нейронных сетей
- NDVIPredictor: генерация прогнозов
- ConfigManager: управление конфигурацией
- DebugLogger: система отладки
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import ee
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
import openmeteo_requests
import requests_cache
from retry_requests import retry

warnings.filterwarnings("ignore")

# ===================== КОНСТАНТЫ =====================
@dataclass
class ModelConfig:
    """Конфигурация модели"""
    LSTM_UNITS: int = 244
    NUM_LAYERS: int = 1
    DROPOUT_RATE: float = 0.2887856831106061
    LEARNING_RATE: float = 0.001827795604676652
    BATCH_SIZE: int = 128
    EPOCHS: int = 200
    WEIGHT_DECAY: float = 1e-5
    ATTENTION_HEADS: int = 4

@dataclass
class DataConfig:
    """Конфигурация данных"""
    RESAMPLE_FREQUENCY: str = '5D'
    CLOUD_PERCENTAGE_THRESHOLD: int = 70
    SCALE_FACTOR: int = 10000
    TEMP_MIN_RANGE: Tuple[float, float] = (-60, 60)
    TEMP_MAX_RANGE: Tuple[float, float] = (-60, 60)
    HUMIDITY_RANGE: Tuple[float, float] = (0, 100)
    PRECIPITATION_MIN: float = 0

WEIGHTS_DIR = "weights"
CONFIG_FILE = "configs/config_ndvi.json"
CACHE_FILE = ".cache"
RESULTS_DIR = "results"  # Изменено с images на results
SERVICE_ACCOUNT_FILE = "key.json"

# ===================== ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ =====================

class DebugLogger:
    """Система отладки и логирования"""
    
    @staticmethod
    def log_data_shape(name: str, data: Any) -> None:
        """Логирует форму данных"""
        if hasattr(data, 'shape'):
            print(f"📊 {name}: shape = {data.shape}")
        elif isinstance(data, (list, tuple)):
            print(f"📊 {name}: length = {len(data)}")
        elif isinstance(data, pd.DataFrame):
            print(f"📊 {name}: DataFrame shape = {data.shape}, columns = {list(data.columns)}")
        else:
            print(f"📊 {name}: type = {type(data)}")
    
    @staticmethod
    def log_ndvi_stats(ndvi_values: List[float], source: str = "API") -> None:
        """Логирует статистику NDVI"""
        if not ndvi_values:
            print(f"⚠️  {source}: Нет данных NDVI")
            return
            
        min_val, max_val = min(ndvi_values), max(ndvi_values)
        mean_val = np.mean(ndvi_values)
        
        print(f"🌿 {source} NDVI статистика:")
        print(f"   📈 Диапазон: {min_val:.3f} - {max_val:.3f}")
        print(f"   📊 Среднее: {mean_val:.3f}")
        print(f"   📋 Количество: {len(ndvi_values)}")
        
        # Оценка качества
        if max_val < 0:
            print("   ⚠️  Все значения отрицательные - проблемная область!")
        elif max_val > 0.8:
            print("   ✅ Высокие значения - здоровая растительность")
        elif max_val > 0.3:
            print("   ✅ Умеренные значения - нормальная растительность")
        else:
            print("   ⚠️  Низкие значения - слабая растительность")
    
    @staticmethod
    def log_training_progress(epoch: int, total_epochs: int, loss: float, model_name: str) -> None:
        """Логирует прогресс обучения"""
        if (epoch + 1) % 20 == 0 or epoch + 1 == total_epochs:
            progress = (epoch + 1) / total_epochs * 100
            print(f"🚀 {model_name}: Epoch [{epoch+1}/{total_epochs}] ({progress:.1f}%) - Loss: {loss:.6f}")

class ConfigManager:
    """Управление конфигурацией"""
    
    @staticmethod
    def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
        """Загружает конфигурацию из JSON файла"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            print(f"✅ Конфигурация загружена из {config_path}")
            print(f"📍 Координаты: {len(config['coordinates'])} точек")
            print(f"📅 Период: {config['start_date']} - {config['end_date']}")
            print(f"🔧 Параметры: n_steps_in={config['n_steps_in']}, n_steps_out={config['n_steps_out']}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл конфигурации {config_path} не найден")
        except json.JSONDecodeError:
            raise ValueError(f"Некорректный JSON в файле {config_path}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Валидирует конфигурацию"""
        required_keys = ['coordinates', 'start_date', 'end_date', 'n_steps_in', 'n_steps_out']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Отсутствуют обязательные параметры: {missing_keys}")
        
        if config['n_steps_in'] <= 0 or config['n_steps_out'] <= 0:
            raise ValueError("n_steps_in и n_steps_out должны быть положительными")
        
        print("✅ Конфигурация валидна")

# ===================== МОДЕЛЬ LSTM =====================

class LSTMModel(nn.Module):
    """LSTM модель с multi-head attention"""
    
    def __init__(self, input_size: int, config: ModelConfig):
        super(LSTMModel, self).__init__()
        self.hidden_size = config.LSTM_UNITS
        self.num_layers = config.NUM_LAYERS
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True, 
            dropout=config.DROPOUT_RATE if self.num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            self.hidden_size, 
            num_heads=config.ATTENTION_HEADS
        )
        
        # Нормализация и выходной слой
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc = nn.Linear(self.hidden_size, 1)  # Выход только 1 значение
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Инициализация скрытых состояний
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention (seq_len, batch, hidden)
        attn_input = lstm_out.permute(1, 0, 2)
        attn_output, _ = self.attention(
            query=attn_input,
            key=attn_input,
            value=attn_input
        )
        
        # Residual connection + layer norm
        attn_output = attn_output.permute(1, 0, 2)  # (batch, seq, hidden)
        output = self.layer_norm(lstm_out + attn_output)
        output = self.dropout(output)
        
        # Выходной слой (только последний временной шаг)
        return self.fc(output[:, -1, :])

# ===================== УПРАВЛЕНИЕ ДАННЫМИ =====================

class DataManager:
    """Управление данными NDVI и погоды"""
    
    def __init__(self, coordinates: List[Tuple[float, float]], service_account_file: str = SERVICE_ACCOUNT_FILE):
        self.coordinates = coordinates
        self.service_account_file = service_account_file
        self._initialize_ee()
        self._setup_weather_client()
        
        # Scalers
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_y_smoothed = MinMaxScaler()
        
    def _initialize_ee(self) -> None:
        """Инициализация Google Earth Engine"""
        try:
            # Проверяем существование файла ключа
            if not os.path.exists(self.service_account_file):
                raise FileNotFoundError(
                    f"Файл ключа {self.service_account_file} не найден. "
                    f"Скопируйте key.json.example в key.json и заполните своими данными."
                )
            
            # Загружаем credentials из JSON файла
            with open(self.service_account_file, 'r') as f:
                service_account_info = json.load(f)
            
            credentials = ee.ServiceAccountCredentials(
                service_account_info["client_email"], 
                self.service_account_file
            )
            ee.Initialize(credentials)
            print("✅ Google Earth Engine инициализирован с ключом из файла")
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Создайте файл key.json на основе key.json.example")
            raise
        except Exception as e:
            print(f"❌ Ошибка инициализации GEE: {e}")
            print("💡 Проверьте правильность данных в key.json")
            raise
    
    def _setup_weather_client(self) -> None:
        """Настройка клиента погодных данных"""
        cache_session = requests_cache.CachedSession(CACHE_FILE, expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.weather_client = openmeteo_requests.Client(session=retry_session)
        print("✅ Клиент погодных данных настроен")
    
    def get_ndvi_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Получает данные NDVI через Google Earth Engine"""
        print(f"🛰️  Получение NDVI данных: {start_date} - {end_date}")
        
        aoi = ee.Geometry.Polygon([self.coordinates])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', DataConfig.CLOUD_PERCENTAGE_THRESHOLD))
                     .map(self._mask_clouds)
                     .map(self._calculate_ndvi))
        
        def compute_ndvi(image):
            values = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10
            )
            
            return ee.Feature(None, {
                'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                'NDVI': values.get('NDVI')
            })
        
        ndvi_collection = collection.filterBounds(aoi).select('NDVI').map(compute_ndvi)
        ndvi_info = ndvi_collection.getInfo()
        
        # Извлечение данных
        dates, ndvi_values = [], []
        for feature in ndvi_info['features']:
            props = feature['properties']
            if props.get('date') and props.get('NDVI') is not None:
                dates.append(props['date'])
                ndvi_values.append(props['NDVI'])
        
        # Отладка
        DebugLogger.log_ndvi_stats(ndvi_values, "GEE API")
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'NDVI': ndvi_values
        })
        
        print(f"✅ Получено {len(df)} записей NDVI")
        return df.sort_values('Date').reset_index(drop=True)
    
    def _mask_clouds(self, image):
        """Маскировка облаков"""
        qa = image.select('QA60')
        scl = image.select('SCL')
        cloud_mask = (qa.bitwiseAnd(1 << 10).eq(0)
                     .And(qa.bitwiseAnd(1 << 11).eq(0))
                     .And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                     .And(scl.neq(10)).And(scl.neq(11)))
        
        return (image.updateMask(cloud_mask)
                    .divide(DataConfig.SCALE_FACTOR)
                    .select("B.*")
                    .copyProperties(image, ["system:time_start"]))
    
    def _calculate_ndvi(self, image):
        """Вычисление NDVI"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def get_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Получает погодные данные"""
        print(f"🌤️  Получение погодных данных: {start_date} - {end_date}")
        
        # Центр полигона
        lats = [coord[0] for coord in self.coordinates]
        lons = [coord[1] for coord in self.coordinates]
        centroid_lat, centroid_lon = np.mean(lats), np.mean(lons)
        
        print(f"📍 Центр области: {centroid_lat:.3f}, {centroid_lon:.3f}")
        
        params = {
            "latitude": centroid_lat,
            "longitude": centroid_lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_min", "temperature_2m_max"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"]
        }
        
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        responses = self.weather_client.weather_api(url, params=params)
        response = responses[0]
        
        # Обработка данных
        weather_df = self._process_weather_response(response)
        weather_df = self._validate_weather_data(weather_df)
        
        print(f"✅ Получено {len(weather_df)} записей погоды")
        DebugLogger.log_data_shape("Weather data", weather_df)
        
        return weather_df
    
    def _process_weather_response(self, response) -> pd.DataFrame:
        """Обрабатывает ответ погодного API"""
        # Дневные данные
        daily = response.Daily()
        daily_dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        
        daily_df = pd.DataFrame({
            "Date": daily_dates,
            "TempMin": daily.Variables(0).ValuesAsNumpy(),
            "TempMax": daily.Variables(1).ValuesAsNumpy()
        })
        daily_df["Date"] = daily_df["Date"].dt.tz_localize(None)
        
        # Почасовые данные
        hourly = response.Hourly()
        hourly_dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        
        hourly_df = pd.DataFrame({
            "DateTime": hourly_dates,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "precipitation": hourly.Variables(2).ValuesAsNumpy()
        })
        
        # Агрегация по дням
        daily_hourly = hourly_df.groupby(hourly_df["DateTime"].dt.date).agg({
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "precipitation": "sum"
        }).reset_index()
        daily_hourly["Date"] = pd.to_datetime(daily_hourly["DateTime"]).dt.tz_localize(None)
        
        # Объединение
        weather_df = pd.merge(daily_df, daily_hourly, on="Date", how="inner")
        weather_df = weather_df.rename(columns={
            "relative_humidity_2m": "RelativeHumidity",
            "precipitation": "Precipitation"
        })
        
        return weather_df[["Date", "TempMin", "TempMax", "RelativeHumidity", "Precipitation"]]
    
    def _validate_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидирует погодные данные"""
        initial_count = len(df)
        
        df = df[
            (df['TempMax'] >= DataConfig.TEMP_MAX_RANGE[0]) & 
            (df['TempMax'] <= DataConfig.TEMP_MAX_RANGE[1]) &
            (df['TempMin'] >= DataConfig.TEMP_MIN_RANGE[0]) & 
            (df['TempMin'] <= DataConfig.TEMP_MIN_RANGE[1]) &
            (df['RelativeHumidity'] >= DataConfig.HUMIDITY_RANGE[0]) & 
            (df['RelativeHumidity'] <= DataConfig.HUMIDITY_RANGE[1]) &
            (df['Precipitation'] >= DataConfig.PRECIPITATION_MIN)
        ]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"⚠️  Отфильтровано {filtered_count} некорректных записей погоды")
        
        return df
    
    def process_data(self, ndvi_df: pd.DataFrame, weather_df: pd.DataFrame, 
                    percentile: float, bimonthly_period: str, 
                    spline_smoothing: float) -> pd.DataFrame:
        """Обрабатывает и объединяет данные NDVI и погоды"""
        print("🔄 Обработка и объединение данных...")
        
        # Интерполяция NDVI
        ndvi_df = self._interpolate_data(ndvi_df)
        DebugLogger.log_data_shape("NDVI после интерполяции", ndvi_df)
        
        # Фильтрация выбросов
        ndvi_df = self._filter_outliers(ndvi_df, percentile, bimonthly_period)
        DebugLogger.log_data_shape("NDVI после фильтрации", ndvi_df)
        
        # Повторная интерполяция
        ndvi_df = self._interpolate_data(ndvi_df)
        
        # Объединение с погодными данными
        merged_df = pd.merge_asof(
            ndvi_df.sort_values('Date'), 
            weather_df.sort_values('Date'), 
            on='Date', 
            direction='nearest'
        )
        
        # Сглаживание NDVI
        merged_df['NDVI_Smoothed'] = self._smooth_data(merged_df, spline_smoothing)
        
        print(f"✅ Объединено {len(merged_df)} записей")
        DebugLogger.log_data_shape("Финальные данные", merged_df)
        
        return merged_df
    
    def _interpolate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Интерполяция данных с фиксированным шагом"""
        # Проверка на дубликаты дат
        duplicates_count = df['Date'].duplicated().sum()
        if duplicates_count > 0:
            print(f"🔍 Найдено {duplicates_count} дубликатов дат, удаляем...")
        
        # Удаляем дубликаты дат, оставляя первое вхождение
        df_clean = df.drop_duplicates(subset=['Date'], keep='first')
        
        df_indexed = df_clean.set_index('Date')
        df_resampled = df_indexed.resample(DataConfig.RESAMPLE_FREQUENCY).interpolate(method='linear')
        return df_resampled.reset_index()
    
    def _filter_outliers(self, df: pd.DataFrame, percentile: float, period: str) -> pd.DataFrame:
        """Фильтрация выбросов по перцентилю"""
        df = df.copy()
        df['Period'] = df['Date'].dt.to_period(period)
        
        thresholds = df.groupby('Period')['NDVI'].quantile(percentile/100).reset_index()
        df = df.merge(thresholds, on='Period', suffixes=('', '_threshold'))
        
        filtered_df = df[df['NDVI'] >= df['NDVI_threshold']]
        removed_count = len(df) - len(filtered_df)
        
        if removed_count > 0:
            print(f"🔍 Отфильтровано {removed_count} выбросов ({percentile}% перцентиль)")
        
        return filtered_df.drop(columns=['Period', 'NDVI_threshold'])
    
    def _smooth_data(self, df: pd.DataFrame, smoothing: float) -> np.ndarray:
        """Сглаживание данных сплайном"""
        if len(df) < 4:
            print("⚠️  Недостаточно данных для сглаживания")
            return df['NDVI'].values
        
        x_ordinal = df['Date'].map(pd.Timestamp.toordinal)
        spline = UnivariateSpline(x_ordinal, df['NDVI'], s=smoothing)
        return spline(x_ordinal)
    
    def prepare_sequences(self, df: pd.DataFrame, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Подготавливает последовательности для обучения"""
        print(f"📦 Подготовка последовательностей: {n_steps_in} входных → {n_steps_out} выходных")
        
        # Масштабирование
        features = ['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']
        X_scaled = self.scaler_x.fit_transform(df[features])
        y_scaled = self.scaler_y.fit_transform(df[['NDVI']])
        y_smoothed_scaled = self.scaler_y_smoothed.fit_transform(df[['NDVI_Smoothed']])
        
        # Создание последовательностей
        X, y_original, y_smoothed = [], [], []
        
        for i in range(len(df) - n_steps_in - n_steps_out + 1):
            # Входная последовательность (признаки + NDVI)
            input_seq = np.hstack([
                X_scaled[i:i+n_steps_in],
                y_scaled[i:i+n_steps_in]
            ])
            X.append(input_seq)
            
            # Выходные последовательности
            y_original.append(y_scaled[i+n_steps_in:i+n_steps_in+n_steps_out].flatten())
            y_smoothed.append(y_smoothed_scaled[i+n_steps_in:i+n_steps_in+n_steps_out].flatten())
        
        X = np.array(X)
        y_original = np.array(y_original)
        y_smoothed = np.array(y_smoothed)
        
        print(f"✅ Создано последовательностей: {len(X)}")
        DebugLogger.log_data_shape("X (входы)", X)
        DebugLogger.log_data_shape("y_original", y_original)
        DebugLogger.log_data_shape("y_smoothed", y_smoothed)
        
        return X, y_original, y_smoothed, X.shape[2]

# ===================== ОБУЧЕНИЕ МОДЕЛЕЙ =====================

class ModelTrainer:
    """Обучение LSTM моделей"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Устройство для обучения: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def create_model(self, input_size: int) -> LSTMModel:
        """Создает модель LSTM"""
        model = LSTMModel(input_size, self.config).to(self.device)
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🧠 Модель создана:")
        print(f"   📊 Всего параметров: {total_params:,}")
        print(f"   🎯 Обучаемых параметров: {trainable_params:,}")
        
        return model
    
    def train_model(self, model: LSTMModel, X: np.ndarray, y: np.ndarray, model_name: str) -> LSTMModel:
        """Обучает одну модель"""
        print(f"\n🚀 Начало обучения модели: {model_name}")
        print(f"📊 Данные: {X.shape[0]} последовательностей")
        
        # Подготовка данных
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0  # Для стабильности
        )
        
        # Оптимизатор и функция потерь
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Обучение
        model.train()
        loss_history = []
        
        for epoch in range(self.config.EPOCHS):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Прямой проход
                outputs = model(batch_X)
                
                # Функция потерь (усредняем по выходным шагам)
                loss = criterion(outputs.squeeze(), batch_y.mean(dim=1))
                
                # Обратное распространение
                loss.backward()
                
                # Градиентный клиппинг
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            loss_history.append(avg_loss)
            
            # Логирование
            DebugLogger.log_training_progress(epoch, self.config.EPOCHS, avg_loss, model_name)
        
        print(f"✅ Обучение {model_name} завершено. Финальная потеря: {loss_history[-1]:.6f}")
        
        model.eval()
        return model
    
    def save_models(self, model_original: LSTMModel, model_smoothed: LSTMModel) -> None:
        """Сохраняет веса моделей"""
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        
        original_path = os.path.join(WEIGHTS_DIR, "model_weights_original.pth")
        smoothed_path = os.path.join(WEIGHTS_DIR, "model_weights_filtered.pth")
        
        torch.save(model_original.state_dict(), original_path)
        torch.save(model_smoothed.state_dict(), smoothed_path)
        
        print(f"💾 Модели сохранены:")
        print(f"   📁 Original: {original_path}")
        print(f"   📁 Smoothed: {smoothed_path}")

# ===================== ГЛАВНЫЙ КЛАСС =====================

class NDVIForecaster:
    """Главный класс для прогнозирования NDVI"""
    
    def __init__(self, config_path: str = CONFIG_FILE):
        print("🌿 Инициализация системы прогнозирования NDVI")
        print("=" * 60)
        
        # Загрузка конфигурации
        self.config = ConfigManager.load_config(config_path)
        ConfigManager.validate_config(self.config)
        
        # Извлечение параметров
        self.coordinates = [tuple(coord) for coord in self.config["coordinates"]]
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.n_steps_in = self.config["n_steps_in"]
        self.n_steps_out = self.config["n_steps_out"]
        self.percentile = self.config.get("percentile", 40)
        self.bimonthly_period = self.config.get("bimonthly_period", "2M")
        self.spline_smoothing = self.config.get("spline_smoothing", 0.7)
        
        # Инициализация компонентов
        self.data_manager = DataManager(self.coordinates)
        self.model_trainer = ModelTrainer(ModelConfig())
        
        # Данные и модели
        self.merged_df = None
        self.model_original = None
        self.model_filtered = None
        
        print("✅ Система инициализирована")
        print("=" * 60)
    
    def load_and_process_data(self) -> None:
        """Загружает и обрабатывает все данные"""
        print("\n📡 ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ")
        print("=" * 40)
        
        # Получение данных
        ndvi_df = self.data_manager.get_ndvi_data(self.start_date, self.end_date)
        weather_df = self.data_manager.get_weather_data(self.start_date, self.end_date)
        
        # Обработка и объединение
        self.merged_df = self.data_manager.process_data(
            ndvi_df, weather_df, 
            self.percentile, self.bimonthly_period, self.spline_smoothing
        )
        
        # Статистика финальных данных
        print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА ДАННЫХ:")
        print(f"   📅 Период: {self.merged_df['Date'].min()} - {self.merged_df['Date'].max()}")
        print(f"   📋 Записей: {len(self.merged_df)}")
        
        ndvi_stats = self.merged_df['NDVI'].describe()
        print(f"   🌿 NDVI: min={ndvi_stats['min']:.3f}, max={ndvi_stats['max']:.3f}, mean={ndvi_stats['mean']:.3f}")
        
        smoothed_stats = self.merged_df['NDVI_Smoothed'].describe()
        print(f"   🌿 NDVI (сглаж): min={smoothed_stats['min']:.3f}, max={smoothed_stats['max']:.3f}, mean={smoothed_stats['mean']:.3f}")
    
    def train_models(self) -> None:
        """Обучает модели"""
        if self.merged_df is None:
            raise ValueError("Сначала загрузите данные с помощью load_and_process_data()")
        
        print("\n🧠 ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("=" * 40)
        
        # Подготовка последовательностей
        X, y_original, y_smoothed, input_size = self.data_manager.prepare_sequences(
            self.merged_df, self.n_steps_in, self.n_steps_out
        )
        
        if len(X) == 0:
            raise ValueError("Недостаточно данных для создания последовательностей")
        
        # Создание и обучение моделей
        self.model_original = self.model_trainer.create_model(input_size)
        self.model_filtered = self.model_trainer.create_model(input_size)
        
        self.model_original = self.model_trainer.train_model(
            self.model_original, X, y_original, "Original"
        )
        self.model_filtered = self.model_trainer.train_model(
            self.model_filtered, X, y_smoothed, "Smoothed"
        )
        
        # Сохранение
        self.model_trainer.save_models(self.model_original, self.model_filtered)
    
    def run_training_pipeline(self) -> None:
        """Выполняет полный пайплайн обучения"""
        print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБУЧЕНИЯ")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            self.load_and_process_data()
            self.train_models()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            print(f"⏱️  Время выполнения: {duration}")
            print(f"📁 Веса сохранены в: {WEIGHTS_DIR}/")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ОШИБКА: {e}")
            raise

# ===================== ГЛАВНАЯ ФУНКЦИЯ =====================

def main():
    """Главная функция"""
    try:
        forecaster = NDVIForecaster()
        forecaster.run_training_pipeline()
        
    except KeyboardInterrupt:
        print("\n⏹️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main() 