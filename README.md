# 🌱 NDVI Time Series Analysis

## 📋 Description

Modular system for NDVI (Normalized Difference Vegetation Index) time series analysis using deep learning. The project is designed for vegetation condition forecasting based on Sentinel-2 satellite data and meteorological information.

## ✨ Features

- 🛰️ **Google Earth Engine integration** for Sentinel-2 data
- 🌤️ **Automatic weather data acquisition** through Open-Meteo API
- 🧠 **LSTM with Multi-Head Attention** for accurate forecasting
- 📊 **Interactive results visualization** using Plotly
- 🔧 **Modular architecture** with extensibility
- 🐛 **Detailed debugging system** with emoji logging
- ⚡ **Automatic data processing** (cloud filtering, interpolation)

## 🏗️ Architecture

### Main components:

- **`DebugLogger`** - Debugging system with informative messages
- **`ConfigManager`** - Project configuration management
- **`DataManager`** - NDVI and weather data acquisition and processing
- **`LSTMModel`** - Neural network with LSTM and attention mechanism
- **`ModelTrainer`** - Model training and saving
- **`NDVIForecaster`** - Main system orchestrator

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Earth Engine setup

Copy example file and fill with your data:

```bash
cp key.json.example key.json
```

Edit `key.json` file by adding your Google Earth Engine Service Account data:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

> ⚠️ **Important:** The `key.json` file contains secret data and should not be committed to version control!

### 3. Configuration setup

Create or edit `configs/config_ndvi.json` file:

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

### 4. Run training

```bash
python ndvi_ts_lstm.py
```

### 5. Test models

```bash
python test_models.py
```

## 📁 Project structure

```
NDVI-time-series-analysis/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 key.json.example             # Google Earth Engine key example
├── 📄 .gitignore                   # Ignored files
├── 🐍 ndvi_ts_lstm.py             # Main system module
├── 🐍 test_models.py              # Model testing
├── 📁 configs/                     # Configuration files
│   └── 📄 config_ndvi.json        # Analysis parameters
├── 📁 weights/                     # Saved model weights
│   ├── 📄 model_weights_original.pth
│   └── 📄 model_weights_filtered.pth
└── 📁 results/                     # Analysis results
    ├── 📄 forecast_metrics.json
    └── 🖼️ ndvi_forecast_comparison.png
```

## ⚙️ Configuration

### Main parameters:

- **`coordinates`** - Analysis area polygon (WGS84)
- **`start_date/end_date`** - Analysis time period
- **`n_steps_in`** - Number of time steps for training
- **`n_steps_out`** - Number of forecast steps
- **`percentile_filter`** - Outlier filter (in percentiles)
- **`spline_smoothing`** - Data smoothing parameter

### Model parameters (ModelConfig):

- **`LSTM_UNITS`** - Number of LSTM neurons (244)
- **`NUM_LAYERS`** - Number of layers (1)
- **`DROPOUT_RATE`** - Dropout coefficient (0.29)
- **`LEARNING_RATE`** - Learning rate (0.0018)
- **`BATCH_SIZE`** - Batch size (128)
- **`EPOCHS`** - Number of epochs (200)

## 📊 Results

The system generates:

1. **Graph images** - PNG files with results visualization
2. **Accuracy metrics** - JSON files with quality indicators
3. **Saved models** - PyTorch weights for reuse
4. **Debug information** - Detailed process logs

### Quality metrics:

- **MAE** (Mean Absolute Error) - Mean absolute error
- **MSE** (Mean Squared Error) - Mean squared error
- **RMSE** (Root MSE) - Root mean squared error
- **R²** - Coefficient of determination

## 🔒 Security

- 🔐 Secret keys are stored in `key.json` file
- 📝 `key.json` file is added to `.gitignore`
- 🗃️ Original keys are saved in `key.json.backup`
- 🚫 Never commit files with real keys!

## 🛠️ Development

### Code structure:

```python
# Debugging system
DebugLogger.log_ndvi_stats(ndvi_values, "API")

# Configuration management
config = ConfigManager.load_config()

# Data acquisition
data_manager = DataManager(coordinates)
ndvi_df = data_manager.get_ndvi_data(start_date, end_date)

# Model training
trainer = ModelTrainer(ModelConfig())
model = trainer.train_model(model, X, y, "Original")
```

### Adding new features:

1. Inherit from base classes
2. Use `DebugLogger` for debugging
3. Follow TypeHints typing
4. Document functions in English