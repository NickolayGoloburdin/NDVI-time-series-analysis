# NDVI Model Usage Instructions

## 1. Data Preparation

### Configuration File
All parameters for training and testing are set in the `configs/config_ndvi.json` file. Example structure:

```
{
    "coordinates": [
        [latitude1, longitude1],
        [latitude2, longitude2],
        ... (closed polygon)
    ],
    "start_date": "YYYY-MM-DD",  # analysis start date
    "end_date": "YYYY-MM-DD",    # analysis end date
    "n_steps_in": 21,             # LSTM input sequence length
    "n_steps_out": 7,             # forecast length
    "percentile": 40,             # percentile for NDVI outlier filtering
    "bimonthly_period": "1M",    # filtering period (e.g., "1M" or "2M")
    "spline_smoothing": 0.7       # NDVI smoothing degree
}
```

- **coordinates** — list of polygon points (latitude, longitude), minimum 3, first and last match.
- **start_date**, **end_date** — analysis range (YYYY-MM-DD format).
- Other parameters — model and filtering hyperparameters.

### Data
- NDVI and weather data are downloaded automatically via Google Earth Engine and open-meteo.com using coordinates and dates from config.
- No additional data files need to be prepared.

## 2. Model Training

1. Make sure all dependencies are installed (see `requirements.txt`).
2. Run the main file for training:
   ```
   python ndvi_ts_lstm.py
   ```
3. After training completion, model weights will be saved to `weights/` folder:
   - `weights/model_weights_original.pth`
   - `weights/model_weights_filtered.pth`

## 3. Testing and Predictions

1. Run the file for testing and inference:
   ```
   python test_models.py
   ```
2. The script automatically:
   - loads model weights
   - downloads NDVI and weather data from config
   - performs predictions on test and future periods
   - outputs quality metrics (MAE, RMSE, R2) and prediction examples

## 4. Using Model for New Data

- Change coordinates and/or dates in `configs/config_ndvi.json`.
- Repeat training and testing steps.
- If you only want predictions for new data — just run `test_models.py` (if weights exist).

## 5. Important Notes

- Working with NDVI requires Google Earth Engine (GEE) access and service account key (`key.json`).
- Weather data uses open-meteo.com (internet required).
- All results and graphs are saved to `images/` folder (if visualization is used).
- For proper operation, Python 3.9–3.12 and all dependencies from `requirements.txt` are recommended.

## 6. Example Run (Linux)

```bash
python ndvi_ts_lstm.py   # training and saving weights
python test_models.py    # testing and predictions
```

---

If you have questions about data format, setup, or running — contact the developer or see comments in source files. 