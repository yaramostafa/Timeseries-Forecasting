import json
import os
import pandas as pd
import joblib
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import zscore
from statsmodels.tsa.stattools import pacf
from scipy.stats import stats

class TimeSeriesPreprocessor:
    def __init__(self, outlier_threshold: float = 3.0):
        self.selected_lags = {}
        self.outlier_threshold = outlier_threshold
    def detect_outliers(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers using z-score method.
        Returns a boolean series where True indicates an outlier.
        """
        z_scores = np.abs(zscore(series, nan_policy='omit'))
        return z_scores > self.outlier_threshold

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in the dataset, either using explicit anomaly labels
        or detecting them using statistical methods.
        """
        df = df.copy()

        if 'anomaly' in df.columns:
            # Use explicit anomaly labels
            df.loc[df['anomaly'], 'value'] = np.nan
        else:
            # Detect outliers using z-score method
            outliers = self.detect_outliers(df['value'])
            df.loc[outliers, 'value'] = np.nan

        return df

    def standardize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # First try linear interpolation
        df = df.interpolate(method='linear')

        # For any remaining NaNs at the edges, use forward and backward fill
        df = df.ffill()
        df = df.bfill()

        return df

    def select_significant_lags(self, series: pd.Series, max_lags: int = 50, threshold: float = 0.1) -> List[int]:
        try:
            # Calculate PACF values
            pacf_values, confidence_intervals = pacf(series.dropna(), nlags=max_lags, alpha=0.05)

            # Find significant lags (excluding lag 0)
            significant_lags = [
                i for i in range(1, len(pacf_values))
                if abs(pacf_values[i]) > threshold
            ]

            # If no significant lags found, return at least lag 1
            if not significant_lags:
                return [1]

            return sorted(significant_lags)

        except Exception as e:
            print(f"Error in PACF calculation: {str(e)}")
            # Fallback to default lag 1 if PACF calculation fails
            return [1]

    def create_features(self, df: pd.DataFrame, dataset_id: str, n_lags: Optional[int] = None, drop_na: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        features = []

        # Use PACF to select significant lags if n_lags not provided
        if n_lags is None:
            max_possible_lags = min(int(len(df) * 0.2), 50)  # Consider up to 20% of data points or max 50
            significant_lags = self.select_significant_lags(
                df['value'],
                max_lags=max_possible_lags,
                threshold=0.1
            )
            self.selected_lags[dataset_id] = significant_lags
            print(f"Selected significant lags for dataset {dataset_id}: {significant_lags}")
        else:
            significant_lags = list(range(1, n_lags + 1))
            self.selected_lags[dataset_id] = significant_lags

        # Create lag features only for significant lags
        for lag in significant_lags:
            lag_col = f'lag_{lag}'
            df[lag_col] = df['value'].shift(lag)
            features.append(lag_col)

        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
        features.extend(['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend'])

        if drop_na:
            df = df.dropna()

        return df, features

    def get_dynamic_window_size(self, df: pd.DataFrame, min_window_size: int = 5, max_window_size: int = 50, window_percentage: float = 0.05) -> int:
        window_size = int(len(df) * window_percentage)
        window_size = max(min_window_size, window_size)
        window_size = min(max_window_size, window_size)
        return window_size

    def add_rolling_features(self, df: pd.DataFrame, min_window_size: int = 5, max_window_size: int = 50, window_percentage: float = 0.05) -> pd.DataFrame:
        window_size = self.get_dynamic_window_size(df, min_window_size, max_window_size, window_percentage)

        df[f'rolling_mean_{window_size}'] = df['value'].rolling(window=window_size).mean()
        df[f'rolling_std_{window_size}'] = df['value'].rolling(window=window_size).std()
        df[f'rolling_min_{window_size}'] = df['value'].rolling(window=window_size).min()
        df[f'rolling_max_{window_size}'] = df['value'].rolling(window=window_size).max()
        df[f'rolling_median_{window_size}'] = df['value'].rolling(window=window_size).median()
        df[f'rolling_sum_{window_size}'] = df['value'].rolling(window=window_size).sum()

        df = df.bfill().ffill()

        return df

    def preprocess(self, df: pd.DataFrame, dataset_id: str, n_lags: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        # My preprocessing pipeline
        df = self.standardize_timestamp(df)
        df = self.handle_outliers(df)  # New step for outlier handling
        df = self.handle_missing_values(df)
        df, features = self.create_features(df, dataset_id, n_lags)
        df = self.add_rolling_features(df)

        return df, features

class ModelTester:
    def __init__(self, models_folder: str):
        self.models_folder = models_folder
        self.preprocessor = TimeSeriesPreprocessor()

    def get_base_dataset_id(self, dataset_id: str) -> str:
        """Strip 'test_' prefix from dataset ID to match with trained models."""
        if dataset_id.startswith('test_'):
            return dataset_id[5:]  # Remove 'test_' prefix
        return dataset_id
        
    def load_model_data(self, dataset_id: str) -> dict:
        """Load the model and its metadata."""
        base_id = self.get_base_dataset_id(dataset_id)
        model_files = [f for f in os.listdir(self.models_folder) 
                      if f.startswith(f'train_{base_id}_')]
        
        if not model_files:
            raise ValueError(f"No model found for dataset {base_id}")
            
        model_path = os.path.join(self.models_folder, model_files[0])
        print(f"Loading model from: {model_path}")
        model_data = joblib.load(model_path)
        return model_data
        
    def test_model(self, json_data: dict) -> dict:
        """Test a single model on new JSON data."""
        
        dataset_id = json_data['dataset_id']
        values = json_data['values']
        
        df = pd.DataFrame(values)
        df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.drop(columns=['time'])
        
        model_data = self.load_model_data(dataset_id)
        model = model_data['model']
        features = model_data['features']
        
        base_id = self.get_base_dataset_id(dataset_id)
        
        processed_df, _ = self.preprocessor.preprocess(
            df=df,
            dataset_id=base_id,
            n_lags=max([int(f.split('_')[1]) for f in features if f.startswith('lag_')], default=1)
        )
        
        X_test = processed_df[features]
        
        y_pred = model.predict(X_test)

        return {"prediction": y_pred[0]}  
