"""
NDVI Time Series Analysis and Forecasting System
==============================================

Modular system for NDVI time series analysis using LSTM and multi-head attention.

Main components:
- DataManager: NDVI and weather data management
- ModelTrainer: neural network training
- NDVIPredictor: forecast generation
- ConfigManager: configuration management
- DebugLogger: debugging system
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

# ===================== CONSTANTS =====================
@dataclass
class ModelConfig:
    """Model configuration"""
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
    """Data configuration"""
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
RESULTS_DIR = "results"  # Changed from images to results
SERVICE_ACCOUNT_FILE = "key.json"

# ===================== HELPER CLASSES =====================

class DebugLogger:
    """Debugging and logging system"""
    
    @staticmethod
    def log_data_shape(name: str, data: Any) -> None:
        """Logs data shape"""
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
        """Logs NDVI statistics"""
        if not ndvi_values:
            print(f"⚠️  {source}: No NDVI data")
            return
            
        min_val, max_val = min(ndvi_values), max(ndvi_values)
        mean_val = np.mean(ndvi_values)
        
        print(f"🌿 {source} NDVI statistics:")
        print(f"   📈 Range: {min_val:.3f} - {max_val:.3f}")
        print(f"   📊 Mean: {mean_val:.3f}")
        print(f"   📋 Count: {len(ndvi_values)}")
        
        # Quality assessment
        if max_val < 0:
            print("   ⚠️  All negative values - problematic area!")
        elif max_val > 0.8:
            print("   ✅ High values - healthy vegetation")
        elif max_val > 0.3:
            print("   ✅ Moderate values - normal vegetation")
        else:
            print("   ⚠️  Low values - weak vegetation")
    
    @staticmethod
    def log_training_progress(epoch: int, total_epochs: int, loss: float, model_name: str) -> None:
        """Logs training progress"""
        if (epoch + 1) % 20 == 0 or epoch + 1 == total_epochs:
            progress = (epoch + 1) / total_epochs * 100
            print(f"🚀 {model_name}: Epoch [{epoch+1}/{total_epochs}] ({progress:.1f}%) - Loss: {loss:.6f}")

class ConfigManager:
    """Configuration management"""
    
    @staticmethod
    def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
        """Loads configuration from JSON file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            print(f"✅ Configuration loaded from {config_path}")
            print(f"📍 Coordinates: {len(config['coordinates'])} points")
            print(f"📅 Period: {config['start_date']} - {config['end_date']}")
            print(f"🔧 Parameters: n_steps_in={config['n_steps_in']}, n_steps_out={config['n_steps_out']}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file {config_path}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validates configuration"""
        required_keys = ['coordinates', 'start_date', 'end_date', 'n_steps_in', 'n_steps_out']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required parameters: {missing_keys}")
        
        if config['n_steps_in'] <= 0 or config['n_steps_out'] <= 0:
            raise ValueError("n_steps_in and n_steps_out must be positive")
        
        print("✅ Configuration is valid")

# ===================== LSTM MODEL =====================

class LSTMModel(nn.Module):
    """LSTM model with multi-head attention"""
    
    def __init__(self, input_size: int, config: ModelConfig):
        super(LSTMModel, self).__init__()
        self.hidden_size = config.LSTM_UNITS
        self.num_layers = config.NUM_LAYERS
        
        # LSTM layer
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
        
        # Normalization and output layer
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc = nn.Linear(self.hidden_size, 1)  # Output only 1 value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Initialize hidden states
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
        
        # Output layer (only last time step)
        return self.fc(output[:, -1, :])

# ===================== DATA MANAGEMENT =====================

class DataManager:
    """NDVI and weather data management"""
    
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
        """Initialize Google Earth Engine"""
        try:
            # Check if the key file exists
            if not os.path.exists(self.service_account_file):
                raise FileNotFoundError(
                    f"Key file {self.service_account_file} not found. "
                    f"Copy key.json.example to key.json and fill in your data."
                )
            
            # Load credentials from JSON file
            with open(self.service_account_file, 'r') as f:
                service_account_info = json.load(f)
            
            credentials = ee.ServiceAccountCredentials(
                service_account_info["client_email"], 
                self.service_account_file
            )
            ee.Initialize(credentials)
            print("✅ Google Earth Engine initialized with key from file")
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Create key.json on the basis of key.json.example")
            raise
        except Exception as e:
            print(f"❌ Error initializing GEE: {e}")
            print("💡 Check the correctness of the data in key.json")
            raise
    
    def _setup_weather_client(self) -> None:
        """Setup weather data client"""
        cache_session = requests_cache.CachedSession(CACHE_FILE, expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.weather_client = openmeteo_requests.Client(session=retry_session)
        print("✅ Weather data client set up")
    
    def get_ndvi_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get NDVI data through Google Earth Engine"""
        print(f"🛰️  Getting NDVI data: {start_date} - {end_date}")
        
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
        
        # Extract data
        dates, ndvi_values = [], []
        for feature in ndvi_info['features']:
            props = feature['properties']
            if props.get('date') and props.get('NDVI') is not None:
                dates.append(props['date'])
                ndvi_values.append(props['NDVI'])
        
        # Debug
        DebugLogger.log_ndvi_stats(ndvi_values, "GEE API")
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'NDVI': ndvi_values
        })
        
        print(f"✅ Got {len(df)} NDVI records")
        return df.sort_values('Date').reset_index(drop=True)
    
    def _mask_clouds(self, image):
        """Mask clouds"""
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
        """Calculate NDVI"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def get_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get weather data"""
        print(f"🌤️  Getting weather data: {start_date} - {end_date}")
        
        # Polygon center
        lats = [coord[0] for coord in self.coordinates]
        lons = [coord[1] for coord in self.coordinates]
        centroid_lat, centroid_lon = np.mean(lats), np.mean(lons)
        
        print(f"📍 Polygon center: {centroid_lat:.3f}, {centroid_lon:.3f}")
        
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
        
        # Data processing
        weather_df = self._process_weather_response(response)
        weather_df = self._validate_weather_data(weather_df)
        
        print(f"✅ Got {len(weather_df)} weather records")
        DebugLogger.log_data_shape("Weather data", weather_df)
        
        return weather_df
    
    def _process_weather_response(self, response) -> pd.DataFrame:
        """Process weather API response"""
        # Daily data
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
        
        # Hourly data
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
        
        # Daily aggregation
        daily_hourly = hourly_df.groupby(hourly_df["DateTime"].dt.date).agg({
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "precipitation": "sum"
        }).reset_index()
        daily_hourly["Date"] = pd.to_datetime(daily_hourly["DateTime"]).dt.tz_localize(None)
        
        # Merge
        weather_df = pd.merge(daily_df, daily_hourly, on="Date", how="inner")
        weather_df = weather_df.rename(columns={
            "relative_humidity_2m": "RelativeHumidity",
            "precipitation": "Precipitation"
        })
        
        return weather_df[["Date", "TempMin", "TempMax", "RelativeHumidity", "Precipitation"]]
    
    def _validate_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate weather data"""
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
            print(f"⚠️  Filtered out {filtered_count} incorrect weather records")
        
        return df
    
    def process_data(self, ndvi_df: pd.DataFrame, weather_df: pd.DataFrame, 
                    percentile: float, bimonthly_period: str, 
                    spline_smoothing: float) -> pd.DataFrame:
        """Process and merge NDVI and weather data"""
        print("🔄 Processing and merging data...")
        
        # Interpolation of NDVI
        ndvi_df = self._interpolate_data(ndvi_df)
        DebugLogger.log_data_shape("NDVI after interpolation", ndvi_df)
        
        # Outlier filtering
        ndvi_df = self._filter_outliers(ndvi_df, percentile, bimonthly_period)
        DebugLogger.log_data_shape("NDVI after filtering", ndvi_df)
        
        # Re-interpolation
        ndvi_df = self._interpolate_data(ndvi_df)
        
        # Merge with weather data
        merged_df = pd.merge_asof(
            ndvi_df.sort_values('Date'), 
            weather_df.sort_values('Date'), 
            on='Date', 
            direction='nearest'
        )
        
        # Smoothing of NDVI
        merged_df['NDVI_Smoothed'] = self._smooth_data(merged_df, spline_smoothing)
        
        print(f"✅ Merged {len(merged_df)} records")
        DebugLogger.log_data_shape("Final data", merged_df)
        
        return merged_df
    
    def _interpolate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolation of data with fixed step"""
        # Check for date duplicates
        duplicates_count = df['Date'].duplicated().sum()
        if duplicates_count > 0:
            print(f"🔍 Found {duplicates_count} date duplicates, removing...")
        
        # Remove date duplicates, keeping the first occurrence
        df_clean = df.drop_duplicates(subset=['Date'], keep='first')
        
        df_indexed = df_clean.set_index('Date')
        df_resampled = df_indexed.resample(DataConfig.RESAMPLE_FREQUENCY).interpolate(method='linear')
        return df_resampled.reset_index()
    
    def _filter_outliers(self, df: pd.DataFrame, percentile: float, period: str) -> pd.DataFrame:
        """Outlier filtering by percentile"""
        df = df.copy()
        df['Period'] = df['Date'].dt.to_period(period)
        
        thresholds = df.groupby('Period')['NDVI'].quantile(percentile/100).reset_index()
        df = df.merge(thresholds, on='Period', suffixes=('', '_threshold'))
        
        filtered_df = df[df['NDVI'] >= df['NDVI_threshold']]
        removed_count = len(df) - len(filtered_df)
        
        if removed_count > 0:
            print(f"🔍 Filtered out {removed_count} outliers ({percentile}% percentile)")
        
        return filtered_df.drop(columns=['Period', 'NDVI_threshold'])
    
    def _smooth_data(self, df: pd.DataFrame, smoothing: float) -> np.ndarray:
        """Smoothing data with spline"""
        if len(df) < 4:
            print("⚠️  Not enough data for smoothing")
            return df['NDVI'].values
        
        x_ordinal = df['Date'].map(pd.Timestamp.toordinal)
        spline = UnivariateSpline(x_ordinal, df['NDVI'], s=smoothing)
        return spline(x_ordinal)
    
    def prepare_sequences(self, df: pd.DataFrame, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Prepare sequences for training"""
        print(f"📦 Preparing sequences: {n_steps_in} inputs → {n_steps_out} outputs")
        
        # Scaling
        features = ['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']
        X_scaled = self.scaler_x.fit_transform(df[features])
        y_scaled = self.scaler_y.fit_transform(df[['NDVI']])
        y_smoothed_scaled = self.scaler_y_smoothed.fit_transform(df[['NDVI_Smoothed']])
        
        # Create sequences
        X, y_original, y_smoothed = [], [], []
        
        for i in range(len(df) - n_steps_in - n_steps_out + 1):
            # Input sequence (features + NDVI)
            input_seq = np.hstack([
                X_scaled[i:i+n_steps_in],
                y_scaled[i:i+n_steps_in]
            ])
            X.append(input_seq)
            
            # Output sequences
            y_original.append(y_scaled[i+n_steps_in:i+n_steps_in+n_steps_out].flatten())
            y_smoothed.append(y_smoothed_scaled[i+n_steps_in:i+n_steps_in+n_steps_out].flatten())
        
        X = np.array(X)
        y_original = np.array(y_original)
        y_smoothed = np.array(y_smoothed)
        
        print(f"✅ Created sequences: {len(X)}")
        DebugLogger.log_data_shape("X (inputs)", X)
        DebugLogger.log_data_shape("y_original", y_original)
        DebugLogger.log_data_shape("y_smoothed", y_smoothed)
        
        return X, y_original, y_smoothed, X.shape[2]

# ===================== MODEL TRAINING =====================

class ModelTrainer:
    """LSTM model training"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Training device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def create_model(self, input_size: int) -> LSTMModel:
        """Create LSTM model"""
        model = LSTMModel(input_size, self.config).to(self.device)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🧠 Model created:")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_model(self, model: LSTMModel, X: np.ndarray, y: np.ndarray, model_name: str) -> LSTMModel:
        """Train one model"""
        print(f"\n🚀 Starting model training: {model_name}")
        print(f"📊 Data: {X.shape[0]} sequences")
        
        # Data preparation
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0  # For stability
        )
        
        # Optimizer and loss function
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Training
        model.train()
        loss_history = []
        
        for epoch in range(self.config.EPOCHS):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                
                # Loss function (averaged over output steps)
                loss = criterion(outputs.squeeze(), batch_y.mean(dim=1))
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            loss_history.append(avg_loss)
            
            # Logging
            DebugLogger.log_training_progress(epoch, self.config.EPOCHS, avg_loss, model_name)
        
        print(f"✅ Training {model_name} completed. Final loss: {loss_history[-1]:.6f}")
        
        model.eval()
        return model
    
    def save_models(self, model_original: LSTMModel, model_smoothed: LSTMModel) -> None:
        """Save model weights"""
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        
        original_path = os.path.join(WEIGHTS_DIR, "model_weights_original.pth")
        smoothed_path = os.path.join(WEIGHTS_DIR, "model_weights_filtered.pth")
        
        torch.save(model_original.state_dict(), original_path)
        torch.save(model_smoothed.state_dict(), smoothed_path)
        
        print(f"💾 Models saved:")
        print(f"   📁 Original: {original_path}")
        print(f"   📁 Smoothed: {smoothed_path}")

# ===================== MAIN CLASS =====================

class NDVIForecaster:
    """Main class for NDVI forecasting"""
    
    def __init__(self, config_path: str = CONFIG_FILE):
        print("🌿 Initializing NDVI forecasting system")
        print("=" * 60)
        
        # Load configuration
        self.config = ConfigManager.load_config(config_path)
        ConfigManager.validate_config(self.config)
        
        # Extract parameters
        self.coordinates = [tuple(coord) for coord in self.config["coordinates"]]
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.n_steps_in = self.config["n_steps_in"]
        self.n_steps_out = self.config["n_steps_out"]
        self.percentile = self.config.get("percentile", 40)
        self.bimonthly_period = self.config.get("bimonthly_period", "2M")
        self.spline_smoothing = self.config.get("spline_smoothing", 0.7)
        
        # Initialize components
        self.data_manager = DataManager(self.coordinates)
        self.model_trainer = ModelTrainer(ModelConfig())
        
        # Data and models
        self.merged_df = None
        self.model_original = None
        self.model_filtered = None
        
        print("✅ System initialized")
        print("=" * 60)
    
    def load_and_process_data(self) -> None:
        """Load and process all data"""
        print("\n📡 DATA LOADING AND PROCESSING")
        print("=" * 40)
        
        # Get data
        ndvi_df = self.data_manager.get_ndvi_data(self.start_date, self.end_date)
        weather_df = self.data_manager.get_weather_data(self.start_date, self.end_date)
        
        # Processing and merging
        self.merged_df = self.data_manager.process_data(
            ndvi_df, weather_df, 
            self.percentile, self.bimonthly_period, self.spline_smoothing
        )
        
        # Final data statistics
        print(f"\n📊 FINAL DATA STATISTICS:")
        print(f"   📅 Period: {self.merged_df['Date'].min()} - {self.merged_df['Date'].max()}")
        print(f"   📋 Records: {len(self.merged_df)}")
        
        ndvi_stats = self.merged_df['NDVI'].describe()
        print(f"   🌿 NDVI: min={ndvi_stats['min']:.3f}, max={ndvi_stats['max']:.3f}, mean={ndvi_stats['mean']:.3f}")
        
        smoothed_stats = self.merged_df['NDVI_Smoothed'].describe()
        print(f"   🌿 NDVI (smoothed): min={smoothed_stats['min']:.3f}, max={smoothed_stats['max']:.3f}, mean={smoothed_stats['mean']:.3f}")
    
    def train_models(self) -> None:
        """Train models"""
        if self.merged_df is None:
            raise ValueError("First load data using load_and_process_data()")
        
        print("\n🧠 MODEL TRAINING")
        print("=" * 40)
        
        # Prepare sequences
        X, y_original, y_smoothed, input_size = self.data_manager.prepare_sequences(
            self.merged_df, self.n_steps_in, self.n_steps_out
        )
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Create and train models
        self.model_original = self.model_trainer.create_model(input_size)
        self.model_filtered = self.model_trainer.create_model(input_size)
        
        self.model_original = self.model_trainer.train_model(
            self.model_original, X, y_original, "Original"
        )
        self.model_filtered = self.model_trainer.train_model(
            self.model_filtered, X, y_smoothed, "Smoothed"
        )
        
        # Save
        self.model_trainer.save_models(self.model_original, self.model_filtered)
    
    def run_training_pipeline(self) -> None:
        """Run full training pipeline"""
        print("🚀 STARTING FULL TRAINING PIPELINE")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            self.load_and_process_data()
            self.train_models()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print("✅ TRAINING SUCCESSFULLY COMPLETED!")
            print(f"⏱️  Execution time: {duration}")
            print(f"📁 Models saved in: {WEIGHTS_DIR}/")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            raise

# ===================== MAIN FUNCTION =====================

def main():
    """Main function"""
    try:
        forecaster = NDVIForecaster()
        forecaster.run_training_pipeline()
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        raise

if __name__ == "__main__":
    main() 