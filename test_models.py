#!/usr/bin/env python3
"""
NDVI Model Testing System
========================

Tests trained models on real data and compares predictions with actual values.
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

# ===================== CONSTANTS =====================
DEFAULT_CONFIG_PATH = "configs/config_ndvi.json"
OUTPUT_DIR = "results"
ORIGINAL_WEIGHTS_PATH = "weights/model_weights_original.pth"
FILTERED_WEIGHTS_PATH = "weights/model_weights_filtered.pth" 
FORECAST_STEP_DAYS = 5

# ===================== MAIN TESTING CLASS =====================

class NDVIModelTester:
    """Class for testing and validating NDVI model."""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize model tester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.forecaster: Optional[NDVIForecaster] = None
        self.data_manager: Optional[DataManager] = None
        self._ensure_output_dir()
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file {config_path}")
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _validate_config(self) -> None:
        """Check and validate configuration parameters."""
        required_params = [
            "coordinates", "n_steps_in", "n_steps_out", "test_start_date", "test_end_date"
        ]
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter in configuration: {param}")
        
        # Check test parameters
        test_start = self.config["test_start_date"]
        test_end = self.config["test_end_date"]
        
        print(f"✓ Test parameters: {test_start} - {test_end}")
        
        total_forecast_days = self.config['n_steps_out'] * FORECAST_STEP_DAYS
        print(f"✓ Forecast: {self.config['n_steps_out']} steps × {FORECAST_STEP_DAYS} days = {total_forecast_days} days")
    
    def setup_forecaster(self) -> NDVIForecaster:
        """
        Create and setup forecaster with data.
        
        Returns:
            Configured NDVIForecaster object
        """
        print("Initializing forecaster...")
        
        # Create test configuration
        test_config = self.config.copy()
        test_config["start_date"] = self.config["test_start_date"]
        test_config["end_date"] = self.config["test_end_date"]
        
        # Save temporary configuration
        temp_config_path = "temp_test_config.json"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        
        try:
            # Create forecaster with temporary configuration
            self.forecaster = NDVIForecaster(temp_config_path)
            
            print("Loading and processing data...")
            self.forecaster.load_and_process_data()
            
            # Create separate data_manager for getting forecast data
            coordinates = [tuple(coord) for coord in self.config["coordinates"]]
            self.data_manager = DataManager(coordinates)
            
            return self.forecaster
            
        finally:
            # Remove temporary file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def load_trained_models(self) -> None:
        """Load trained model weights."""
        if self.forecaster is None:
            raise ValueError("Forecaster must be initialized before loading models")
        
        # Check if weight files exist
        if not (os.path.exists(ORIGINAL_WEIGHTS_PATH) and os.path.exists(FILTERED_WEIGHTS_PATH)):
            raise FileNotFoundError(
                "Model weights not found. First train the model by running ndvi_ts_lstm.py"
            )
        
        print("Loading model weights...")
        
        # Prepare data to determine input dimension
        X, _, _, input_size = self.forecaster.data_manager.prepare_sequences(
            self.forecaster.merged_df, 
            self.forecaster.n_steps_in, 
            self.forecaster.n_steps_out
        )
        
        # Create models
        model_config = ModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_original = LSTMModel(input_size, model_config).to(device)
        model_filtered = LSTMModel(input_size, model_config).to(device)
        
        # Load weights
        model_original.load_state_dict(
            torch.load(ORIGINAL_WEIGHTS_PATH, map_location=device)
        )
        model_filtered.load_state_dict(
            torch.load(FILTERED_WEIGHTS_PATH, map_location=device)
        )
        
        # Set to evaluation mode
        model_original.eval()
        model_filtered.eval()
        
        self.forecaster.model_original = model_original
        self.forecaster.model_filtered = model_filtered
        
        print("✅ Model weights successfully loaded!")
    
    def generate_forecast(self) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Generate NDVI forecast for specified period.
        
        Returns:
            Tuple of forecast dates and predicted NDVI values
        """
        if self.forecaster is None:
            raise ValueError("Forecaster must be initialized")
        
        print("🔮 Generating forecast...")
        
        # Create forecast dates
        forecast_start = pd.Timestamp(self.config["test_end_date"]) + pd.Timedelta(days=FORECAST_STEP_DAYS)
        forecast_dates = pd.date_range(
            start=forecast_start,
            periods=self.config["n_steps_out"],
            freq=f'{FORECAST_STEP_DAYS}D'
        )
        
        print(f"📅 Forecast: {forecast_dates[0]} - {forecast_dates[-1]}")
        
        # Use recent data for forecasting
        recent_data = self.forecaster.merged_df.tail(self.forecaster.n_steps_in).copy()
        
        # Scale recent data
        features = ['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation', 'NDVI_Smoothed']
        recent_scaled = self.forecaster.data_manager.scaler_x.fit_transform(
            recent_data[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']]
        )
        recent_ndvi_scaled = self.forecaster.data_manager.scaler_y_smoothed.transform(
            recent_data[['NDVI_Smoothed']]
        )
        
        # Combine features and NDVI
        input_sequence = np.hstack([recent_scaled, recent_ndvi_scaled])
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  # Add batch dimension
        
        # Generate forecast
        with torch.no_grad():
            forecast_tensor = self.forecaster.model_filtered(input_tensor)
            forecast_scaled = forecast_tensor.cpu().numpy().flatten()
        
        # Create forecast for all steps (simple repetition of one prediction)
        forecast_predictions = np.repeat(forecast_scaled[0], self.config["n_steps_out"])
        
        # Reverse scaling
        forecast_ndvi = self.forecaster.data_manager.scaler_y_smoothed.inverse_transform(
            forecast_predictions.reshape(-1, 1)
        ).flatten()
        
        # Forecast diagnostics
        print(f"📊 Forecast NDVI diagnostics:")
        print(f"   📈 Range: {forecast_ndvi.min():.3f} - {forecast_ndvi.max():.3f}")
        print(f"   📊 Mean: {forecast_ndvi.mean():.3f}")
        print(f"   📋 Values: {[f'{v:.3f}' for v in forecast_ndvi]}")
        
        if forecast_ndvi.max() < 0:
            print("   ⚠️  WARNING: All forecast NDVI values are negative!")
        elif forecast_ndvi.max() > 0.3:
            print("   ✅ Good values - vegetation is predicted")
        else:
            print("   ⚠️  Low NDVI values")
        
        return forecast_dates, forecast_ndvi
    
    def get_real_ndvi_for_dates(self, forecast_dates: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Get daily actual NDVI values in extended range for better comparison.
        
        Args:
            forecast_dates: Forecast dates for determining range
            
        Returns:
            Tuple of actual data dates and NDVI values array
        """
        if self.data_manager is None:
            raise ValueError("DataManager must be initialized")
        
        print("🛰️  Getting actual NDVI values...")
        
        # Extend range: 30 days before forecast + entire forecast period
        start_date = forecast_dates[0] - pd.Timedelta(days=30)
        end_date = forecast_dates[-1]
        
        print(f"📅 Requesting actual data: {start_date.date()} - {end_date.date()}")
        
        # Get actual NDVI data for forecast period
        real_ndvi_df = self.data_manager.get_ndvi_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if real_ndvi_df.empty:
            print("⚠️  No actual data NDVI for comparison period")
            return pd.DatetimeIndex([]), np.array([])
        
        print(f"✅ {len(real_ndvi_df)} actual data points NDVI obtained")
        print(f"📊 Actual NDVI values range: {real_ndvi_df['NDVI'].min():.3f} - {real_ndvi_df['NDVI'].max():.3f}")
        
        return real_ndvi_df['Date'], real_ndvi_df['NDVI'].values
    
    def calculate_metrics(self, forecast_dates: pd.DatetimeIndex, forecast_values: np.ndarray,
                         real_dates: pd.DatetimeIndex, real_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate forecast quality metrics.
        
        Args:
            forecast_dates: Forecast dates
            forecast_values: Predicted NDVI values
            real_dates: Actual data dates
            real_values: Actual NDVI values
            
        Returns:
            Dictionary with quality metrics
        """
        print("📊 Calculating forecast quality metrics...")
        
        # Create DataFrame for convenient work
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'NDVI_Forecast': forecast_values
        })
        
        real_df = pd.DataFrame({
            'Date': real_dates,
            'NDVI_Real': real_values
        })
        
        # Merge data by dates
        merged_df = pd.merge(forecast_df, real_df, on='Date', how='inner')
        
        if len(merged_df) == 0:
            print("⚠️  No overlapping dates for metrics calculation")
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
        
        # Extract matched values
        forecast_matched = merged_df['NDVI_Forecast'].values
        real_matched = merged_df['NDVI_Real'].values
        
        # Calculate metrics
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
        
        print(f"✅ Metrics calculated for {len(merged_df)} matched points:")
        print(f"   🎯 MAE: {mae:.4f}")
        print(f"   🎯 RMSE: {rmse:.4f}")
        print(f"   🎯 R²: {r2:.4f}")
        
        return metrics
    
    def plot_comparison(self, forecast_dates: pd.DatetimeIndex, forecast: np.ndarray, 
                       real_dates: pd.DatetimeIndex, real: np.ndarray) -> None:
        """
        Create and save forecast comparison graph as PNG image.
        
        Args:
            forecast_dates: Forecast dates
            forecast: Predicted NDVI values
            real_dates: Actual data dates
            real: Actual NDVI values
        """
        print("📈 Creating forecast comparison graph...")
        
        fig = go.Figure()
        
        # Separate actual data into historical and comparison periods
        forecast_start = forecast_dates[0]
        
        historical_mask = real_dates < forecast_start
        comparison_mask = real_dates >= forecast_start
        
        # Historical data (gray color)
        if np.any(historical_mask):
            fig.add_trace(go.Scatter(
                x=real_dates[historical_mask],
                y=real[historical_mask],
                mode='markers+lines',
                name='Historical data',
                line=dict(color='gray', width=2),
                marker=dict(size=4, color='gray'),
                opacity=0.7
            ))
        
        # Actual data in forecast period (blue)
        if np.any(comparison_mask):
            fig.add_trace(go.Scatter(
                x=real_dates[comparison_mask],
                y=real[comparison_mask],
                mode='markers+lines',
                name='Actual data (for comparison)',
                line=dict(color='blue', width=3),
                marker=dict(size=6, color='blue')
            ))
        
        # Forecast data (red)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='markers+lines',
            name='Forecast model',
            line=dict(color='red', width=3),
            marker=dict(size=8, color='red', symbol='diamond')
        ))
        
        # Layout settings
        fig.update_layout(
            title={
                'text': 'Forecast NDVI comparison with actual data',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Date',
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
        
        # Add horizontal lines for reference
        fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.3, 
                     annotation_text="Healthy vegetation threshold", annotation_position="left")
        
        # Save graph as PNG image
        output_path = os.path.join(OUTPUT_DIR, "ndvi_forecast_comparison.png")
        fig.write_image(output_path, width=1200, height=600, scale=2)
        
        print(f"✅ Graph saved as image: {output_path}")
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary with metrics
        """
        output_path = os.path.join(OUTPUT_DIR, "forecast_metrics.json")
        
        # Add metadata
        full_metrics = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_metrics, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Metrics saved: {output_path}")
    
    def run_full_test(self) -> None:
        """Perform full model testing cycle."""
        print("🚀 STARTING FULL MODEL TESTING")
        print("=" * 60)
        
        try:
            # Initialize and setup
            self.setup_forecaster()
            self.load_trained_models()
            
            # Generate forecast
            forecast_dates, forecast_values = self.generate_forecast()
            
            # Get actual data
            real_dates, real_values = self.get_real_ndvi_for_dates(forecast_dates)
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                forecast_dates, forecast_values,
                real_dates, real_values
            )
            
            # Create graph
            self.plot_comparison(
                forecast_dates, forecast_values,
                real_dates, real_values
            )
            
            # Save results
            self.save_metrics(metrics)
            
            print("\n" + "=" * 60)
            print("✅ FULL TESTING SUCCESSFULLY COMPLETED!")
            print(f"📁 Results saved in: {OUTPUT_DIR}/")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            raise

# ===================== MAIN FUNCTION =====================

def main():
    """Main function to run testing."""
    print("Starting NDVI model testing...")
    
    try:
        tester = NDVIModelTester()
        tester.run_full_test()
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\nError during model testing: {e}")
        raise

if __name__ == "__main__":
    main() 