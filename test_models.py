import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ndvi_ts_lstm import NDVIForecaster

def load_config(config_path="configs/config_ndvi.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_forecaster_and_data(config):
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
    forecaster.initialize_ee()
    s = pd.Timestamp(config["start_date"])
    e = pd.Timestamp(config["end_date"])
    ndvi_timeseries = forecaster.get_ndvi_timeseries(s, e)
    forecaster.extract_ndvi_data(ndvi_timeseries)
    forecaster.get_weather_data(s, e)
    forecaster.merge_data()
    forecaster.prepare_data()
    return forecaster

def load_model_weights(forecaster, weights_path_original, weights_path_filtered):
    train_data, smoothed_data = forecaster.scale_data()
    X_train, y_train = forecaster.split_sequences(train_data, forecaster.n_steps_in, forecaster.n_steps_out)
    n_features = X_train.shape[2]
    forecaster.model_original = forecaster.create_model(n_features)
    forecaster.model_filtered = forecaster.create_model(n_features)
    forecaster.model_original.load_state_dict(torch.load(weights_path_original, map_location=forecaster.device))
    forecaster.model_filtered.load_state_dict(torch.load(weights_path_filtered, map_location=forecaster.device))
    forecaster.model_original.eval()
    forecaster.model_filtered.eval()

def forecast_next_n_steps(forecaster, end_date, n_steps_out):
    # Прогнозируем n_steps_out шагов вперёд с шагом 5 дней
    forecast_dates = pd.date_range(start=pd.Timestamp(end_date) + pd.Timedelta(days=5), periods=n_steps_out, freq='5D')
    # Генерируем погоду на эти даты
    future_weather = forecaster.get_future_weather(forecast_dates)
    future_X = forecaster.scaler_x.transform(future_weather[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']])
    last_known_data = forecaster.train_df[['TempMin', 'TempMax', 'RelativeHumidity', 'Precipitation']].tail(forecaster.n_steps_in).values
    last_known_X = forecaster.scaler_x.transform(last_known_data)
    forecast_input_sequence = np.vstack([last_known_X, future_X])
    forecast_pred_smoothed = forecaster.predict_future(forecaster.model_filtered, forecast_input_sequence)
    forecast_pred_smoothed = np.array(forecast_pred_smoothed)
    forecast_pred_smoothed = forecaster.scaler_y_smoothed.inverse_transform(forecast_pred_smoothed.reshape(-1, 1)).flatten()
    return forecast_dates, forecast_pred_smoothed

def get_real_ndvi_for_period(config, start_date, end_date):
    """Получить реальные NDVI за указанный диапазон дат (без привязки к прогнозу)"""
    forecaster_real = NDVIForecaster(
        coordinates=config["coordinates"],
        start_date=str(start_date),
        end_date=str(end_date),
        n_steps_in=config["n_steps_in"],
        n_steps_out=config["n_steps_out"],
        percentile=config["percentile"],
        bimonthly_period=config["bimonthly_period"],
        spline_smoothing=config["spline_smoothing"]
    )
    forecaster_real.initialize_ee()
    ndvi_timeseries = forecaster_real.get_ndvi_timeseries(str(start_date), str(end_date))
    ndvi_df = forecaster_real.extract_ndvi_data(ndvi_timeseries)
    return ndvi_df[['Date', 'NDVI']]

def plot_and_metrics(forecast_dates, forecast_pred, real_ndvi):
    plt.figure(figsize=(14, 7))
    plt.plot(forecast_dates, forecast_pred, label='Прогноз NDVI (сглаженный)', color='red')
    plt.plot(forecast_dates, real_ndvi, label='Реальный NDVI', color='blue')
    plt.xlabel('Дата')
    plt.ylabel('NDVI')
    plt.title('Сравнение прогноза и реального NDVI')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('images/ndvi_forecast_vs_real.png')
    plt.close()
    # Метрики
    mask = ~np.isnan(real_ndvi)
    if np.sum(mask) == 0:
        metrics_text = "Нет валидных точек для сравнения."
    else:
        mae = mean_absolute_error(real_ndvi[mask], forecast_pred[mask])
        rmse = np.sqrt(mean_squared_error(real_ndvi[mask], forecast_pred[mask]))
        r2 = r2_score(real_ndvi[mask], forecast_pred[mask])
        metrics_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\n"
    with open('images/ndvi_forecast_vs_real_metrics.txt', 'w') as f:
        f.write(metrics_text)

def match_forecast_with_real_ndvi(forecast_dates, ndvi_real_df, tolerance_days=10):
    """Для каждой даты прогноза ищет ближайший реальный NDVI (или NaN, если нет в окне tolerance)"""
    if ndvi_real_df.empty:
        return np.full(len(forecast_dates), np.nan)
    ndvi_real_df = ndvi_real_df.sort_values('Date')
    forecast_dates_pd = pd.to_datetime(forecast_dates)
    merged = pd.merge_asof(
        pd.DataFrame({'Date': forecast_dates_pd}),
        ndvi_real_df[['Date', 'NDVI']],
        on='Date',
        direction='nearest',
        tolerance=pd.Timedelta(f'{tolerance_days}D')
    )
    return merged['NDVI'].values

def main():
    config = load_config()
    forecaster = load_forecaster_and_data(config)
    weights_path_original = "weights/model_weights_original.pth"
    weights_path_filtered = "weights/model_weights_filtered.pth"
    if not (os.path.exists(weights_path_original) and os.path.exists(weights_path_filtered)):
        print("Веса моделей не найдены. Сначала обучите модель.")
        return
    load_model_weights(forecaster, weights_path_original, weights_path_filtered)
    print("Веса моделей успешно загружены.")
    # Прогноз на n_steps_out шагов вперёд
    forecast_dates, forecast_pred = forecast_next_n_steps(forecaster, config["end_date"], config["n_steps_out"])
    # Приводим forecast_dates к строковому формату YYYY-MM-DD для диапазона
    forecast_start_str = pd.to_datetime(forecast_dates[0]).strftime('%Y-%m-%d')
    forecast_end_str = pd.to_datetime(forecast_dates[-1]).strftime('%Y-%m-%d')
    # Получаем реальные NDVI на весь диапазон прогнозирования
    ndvi_real_df = get_real_ndvi_for_period(config, forecast_start_str, forecast_end_str)
    # Сопоставляем прогнозные даты с реальными NDVI
    real_ndvi = match_forecast_with_real_ndvi(forecast_dates, ndvi_real_df)
    # Сравнение и график
    plot_and_metrics(forecast_dates, forecast_pred, real_ndvi)

if __name__ == "__main__":
    main() 