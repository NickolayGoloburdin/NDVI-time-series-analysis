import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import NDVIForecaster class from main file
from ndvi_ts_lstm import NDVIForecaster, LSTMModel

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 12})
colors = sns.color_palette("Set2", 10)

def load_config(config_path="configs/config_ndvi.json"):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        # If config not found, create test config
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
        print(f"Created test configuration file: {config_path}")
        return test_config
    
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Configuration loaded from: {config_path}")
    return config

def test_ndvi_forecaster():
    """Main function for testing NDVI forecasting model"""
    # Load configuration
    config = load_config()
    
    # Create NDVIForecaster instance
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
    
    print("1. Initializing Google Earth Engine...")
    forecaster.initialize_ee()
    
    # STEP 1: Get and visualize NDVI data
    print("2. Getting NDVI data...")
    ndvi_timeseries = forecaster.get_ndvi_timeseries(config["start_date"], config["end_date"])
    ndvi_df = forecaster.extract_ndvi_data(ndvi_timeseries)
    
    # Visualize NDVI
    plt.figure(figsize=(14, 7))
    plt.plot(ndvi_df['Date'], ndvi_df['NDVI'], 'o-', color=colors[0], label='NDVI (after interpolation and filtering)')
    plt.title('NDVI Time Series', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('NDVI Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/ndvi_timeseries.png')
    print("NDVI graph saved to images/ndvi_timeseries.png")
    plt.close()
    
    # STEP 2: Get and visualize weather data
    print("3. Getting weather data...")
    weather_df = forecaster.get_weather_data(config["start_date"], config["end_date"])
    
    # Visualize weather data
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Temperature plot
    axs[0].plot(weather_df['Date'], weather_df['TempMax'], '-', color=colors[1], label='Maximum temperature')
    axs[0].plot(weather_df['Date'], weather_df['TempMin'], '-', color=colors[2], label='Minimum temperature')
    axs[0].set_ylabel('Temperature (°C)', fontsize=14)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Relative humidity plot
    axs[1].plot(weather_df['Date'], weather_df['RelativeHumidity'], '-', color=colors[3], label='Relative humidity')
    axs[1].set_ylabel('Humidity (%)', fontsize=14)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Precipitation plot
    axs[2].bar(weather_df['Date'], weather_df['Precipitation'], color=colors[4], label='Precipitation', alpha=0.7, width=3)
    axs[2].set_ylabel('Precipitation (mm)', fontsize=14)
    axs[2].set_xlabel('Date', fontsize=14)
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Weather Parameters', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/weather_data.png')
    print("Weather data graph saved to images/weather_data.png")
    plt.close()
    
    # STEP 3: Merge and prepare data
    print("4. Merging and preparing data...")
    forecaster.merge_data()
    forecaster.prepare_data()
    
    # Visualize merged data
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # NDVI and smoothed NDVI
    axs[0].plot(forecaster.train_df['Date'], forecaster.train_df['NDVI'], 'o-', color=colors[0], 
                label='NDVI (filter+interpolation)', alpha=0.7, markersize=4)
    axs[0].plot(forecaster.train_df['Date'], forecaster.train_df['NDVI_Smoothed'], '-', color=colors[5], 
                label='NDVI (smoothed)', linewidth=2)
    axs[0].set_ylabel('NDVI', fontsize=14)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Temperature and precipitation
    ax1 = axs[1]
    ax1.plot(forecaster.train_df['Date'], forecaster.train_df['TempMax'], '-', color=colors[1], label='Max. temperature')
    ax1.plot(forecaster.train_df['Date'], forecaster.train_df['TempMin'], '-', color=colors[2], label='Min. temperature')
    ax1.set_ylabel('Temperature (°C)', fontsize=14)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Overlay precipitation plot
    ax2 = ax1.twinx()
    ax2.bar(forecaster.train_df['Date'], forecaster.train_df['Precipitation'], 
            color=colors[4], label='Precipitation', alpha=0.4, width=3)
    ax2.set_ylabel('Precipitation (mm)', fontsize=14)
    ax2.legend(loc='upper right')
    
    plt.suptitle('Merged NDVI and Weather Data', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/merged_data.png')
    print("Merged data graph saved to images/merged_data.png")
    plt.close()
    
    # STEP 4: Train model or load weights
    # Check if model weights exist
    weights_path_original = "weights/model_weights_original.pth"
    weights_path_filtered = "weights/model_weights_filtered.pth"
    
    if os.path.exists(weights_path_original) and os.path.exists(weights_path_filtered):
        print("5. Loading pretrained model weights...")
        # Prepare data
        train_data, smoothed_data = forecaster.scale_data()
        X_train, y_train = forecaster.split_sequences(train_data, config["n_steps_in"], config["n_steps_out"])
        n_features = X_train.shape[2]
        
        # Create models
        forecaster.model_original = forecaster.create_model(n_features)
        forecaster.model_filtered = forecaster.create_model(n_features)
        
        # Load weights
        forecaster.load_model_weights(forecaster.model_original, "original")
        forecaster.load_model_weights(forecaster.model_filtered, "filtered")
    else:
        print("5. Pretrained models not found")
        raise Exception("Pretrained models not found")
    
    # STEP 5: Forecasting and evaluation
    print("6. Forecasting NDVI...")
    test_pred_original, test_pred_smoothed, forecast_pred_original, forecast_pred_smoothed = forecaster.forecast()
    
    # Create summary plot with forecast
    # Define dates for forecast
    if forecaster.case in [1, 2] and forecaster.test_df is not None:
        # For cases with test data
        test_dates = forecaster.test_df['Date'][:len(test_pred_smoothed)].values
    
    # Dates for future forecast
    forecast_dates = pd.date_range(start=forecaster.forecast_dates[0], periods=len(forecast_pred_smoothed), freq='5D')
    
    plt.figure(figsize=(14, 7))
    
    # Training data
    plt.plot(forecaster.train_df['Date'], forecaster.train_df['NDVI'], 'o-', color=colors[0], 
             label='NDVI (filter+interpolation)', alpha=0.7, markersize=4)
    plt.plot(forecaster.train_df['Date'], forecaster.train_df['NDVI_Smoothed'], '-', color=colors[5], 
             label='NDVI (smoothed)', linewidth=2)
    
    # Test data and forecasts
    if forecaster.case in [1, 2] and forecaster.test_df is not None and not forecaster.test_df.empty:
        plt.plot(forecaster.test_df['Date'], forecaster.test_df['NDVI'], 'o-', color='lightblue', 
                 label='Actual NDVI (test)', alpha=0.7, markersize=4)
        plt.plot(forecaster.test_df['Date'], forecaster.test_df['NDVI_Smoothed'], '-', color='blue', 
                 label='Actual NDVI (smoothed, test)', linewidth=2)
        
        if test_pred_smoothed is not None and len(test_pred_smoothed) > 0:
            plt.plot(test_dates, test_pred_smoothed, '--', color='red', 
                     label='LSTM forecast (smoothed)', linewidth=2)
    
    # Display future forecast
    if forecaster.case in [2, 3]:
        if forecast_pred_smoothed is not None and len(forecast_pred_smoothed) > 0:
            plt.plot(forecast_dates, forecast_pred_smoothed, '--', color='red', 
                     label='LSTM forecast (smoothed)', linewidth=2)
        
        # Display historical baseline forecast
        plt.plot(forecaster.baseline_df['Date'], forecaster.baseline_df['Historical_Avg_NDVI_Smoothed'], 
                 '--', color='purple', label='Historical baseline forecast', linewidth=2)
    
    # Add vertical line separating history from forecast
    current_date = forecaster.current_date
    plt.axvline(x=current_date, color='black', linestyle='--', label='Current date')
    
    plt.title(f'NDVI Forecast - Scenario {forecaster.case}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('NDVI', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images/ndvi_forecast_case_{forecaster.case}.png')
    print(f"NDVI forecast saved to images/ndvi_forecast_case_{forecaster.case}.png")
    plt.close()
    
    # STEP 6: Model accuracy evaluation (if test data exists)
    if forecaster.case in [1, 2] and forecaster.test_df is not None and not forecaster.test_df.empty:
        print("7. Model accuracy evaluation...")
        
        # Limit test data to forecast size
        test_actual = forecaster.test_df['NDVI_Smoothed'].values[:len(test_pred_smoothed)]
        
        if len(test_actual) > 0 and len(test_pred_smoothed) > 0:
            # Calculate metrics
            mae = mean_absolute_error(test_actual, test_pred_smoothed)
            rmse = np.sqrt(mean_squared_error(test_actual, test_pred_smoothed))
            r2 = r2_score(test_actual, test_pred_smoothed)
            
            # Create forecast comparison graph with actual data
            plt.figure(figsize=(14, 7))
            plt.plot(test_dates[:len(test_actual)], test_actual, 'o-', color='blue', 
                    label='Actual NDVI', alpha=0.7)
            plt.plot(test_dates[:len(test_pred_smoothed)], test_pred_smoothed, 'o--', color='red', 
                    label='NDVI Forecast', alpha=0.7)
            
            # Add metrics to title
            plt.title(f'Forecast Comparison with Actual Data\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}', 
                    fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('NDVI', fontsize=14)
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('images/forecast_accuracy.png')
            print("Forecast accuracy graph saved to images/forecast_accuracy.png")
            plt.close()
            
            # Create scatter plot
            plt.figure(figsize=(10, 10))
            plt.scatter(test_actual, test_pred_smoothed, color='blue', alpha=0.7)
            plt.plot([min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], 
                    'r--', linewidth=2)
            plt.title('Scatter Plot: Actual vs Forecasted NDVI', fontsize=16)
            plt.xlabel('Actual NDVI', fontsize=14)
            plt.ylabel('Forecasted NDVI', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('images/scatter_plot.png')
            print("Scatter plot saved to images/scatter_plot.png")
            plt.close()
        else:
            print("Insufficient data for model accuracy evaluation")
    
    # STEP 7: Visualize input sequences and attention mechanism
    print("8. Visualizing model input parameters...")
    
    # Create input sequence graph
    plt.figure(figsize=(14, 10))
    
    # Create example input sequence for visualization
    input_sequence = forecaster.train_df.tail(config["n_steps_in"])[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']]
    
    # Scale sequence
    input_scaled = forecaster.scaler_x.transform(input_sequence)
    
    # Visualize input functions
    x_range = np.arange(len(input_sequence))
    
    plt.subplot(4, 1, 1)
    plt.plot(x_range, input_sequence['TempMin'], 'o-', color=colors[2], label='Min. temperature')
    plt.plot(x_range, input_sequence['TempMax'], 'o-', color=colors[1], label='Max. temperature')
    plt.title('Input Sequence for LSTM Model', fontsize=16)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(4, 1, 2)
    plt.plot(x_range, input_sequence['RelativeHumidity'], 'o-', color=colors[3], label='Relative humidity')
    plt.ylabel('Humidity (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(4, 1, 3)
    plt.bar(x_range, input_sequence['Precipitation'], color=colors[4], label='Precipitation', alpha=0.7, width=0.8)
    plt.ylabel('Precipitation (mm)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(4, 1, 4)
    plt.plot(x_range, input_sequence.index.map(lambda x: forecaster.train_df.loc[x, 'NDVI']), 
             'o-', color=colors[0], label='NDVI')
    plt.plot(x_range, input_sequence.index.map(lambda x: forecaster.train_df.loc[x, 'NDVI_Smoothed']), 
             'o-', color=colors[5], label='NDVI (smoothed)')
    plt.ylabel('NDVI', fontsize=12)
    plt.xlabel('Sequence Step', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/input_sequence.png')
    print("Input sequence graph saved to images/input_sequence.png")
    plt.close()
    
    print("NDVI forecasting test completed!")
    print("All graphs saved to directory 'images/'")

def explain_model():
    """Creates information graph with model scheme"""
    # Create schematic LSTM model with attention mechanism image
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Hide axes
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'LSTM Model Architecture with Attention Mechanism', 
            fontsize=18, weight='bold', ha='center', va='top')
    
    # Add model description
    description = """
    NDVI forecasting model working principle:
    
    1. Input data:
       • Weather parameters: min./max. temperature, humidity, precipitation
       • Historical NDVI values with 5-day interval
    
    2. Data preprocessing:
       • Outliers filtering using percentiles for 2-month periods
       • Linear interpolation for filling missing values
       • Smoothing using splines to reduce noise
       • Scaling all values to range [0,1]
    
    3. LSTM architecture:
       • Input layer: weather parameters + NDVI values
       • LSTM layers with dropout for preventing overfitting
       • Multi-head attention mechanism for determining input data significance
       • Fully connected layer for NDVI output values
    
    4. Training:
       • Time series formation method "sliding window"
       • Optimization using Adam with gradient clipping
       • Loss function: Mean Squared Error (MSE)
    
    5. Forecasting:
       • Input sequence: n_steps_in steps (usually 72 points = 360 days)
       • Output sequence: n_steps_out steps (usually 18 points = 90 days)
       • Forecast generated with 5-day step
    
    6. Check and validation:
       • Comparison with actual NDVI values
       • Accuracy evaluation: MAE, RMSE, R²
       • Comparison with historical average (baseline)
    """
    
    ax.text(0.5, 0.5, description, fontsize=14, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('images/model_explanation.png', dpi=300, bbox_inches='tight')
    print("Model scheme saved to images/model_explanation.png")
    plt.close()

if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Create model explanation
    explain_model()
    
    # Run test
    test_ndvi_forecaster() 