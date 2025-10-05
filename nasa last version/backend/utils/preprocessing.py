"""
Data preprocessing utilities for exoplanet classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from config import settings, FEATURE_MEDIANS


def preprocess_row(row_dict: Dict) -> List[float]:
    """
    Preprocess a single row of exoplanet data
    
    Args:
        row_dict: Dictionary with feature names as keys
        
    Returns:
        List of preprocessed feature values in correct order
    """
    processed_values = []
    
    for feature_name in settings.FEATURE_NAMES:
        value = row_dict.get(feature_name)
        
        # handle missing values with median imputation
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = FEATURE_MEDIANS[feature_name]
        
        processed_values.append(float(value))
    
    return processed_values


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a dataframe of exoplanet data
    
    Args:
        df: Input dataframe with exoplanet features
        
    Returns:
        Preprocessed dataframe ready for model prediction
    """
    # create copy to avoid modifying original
    df_processed = df.copy()
    
    # ensure all required features are present
    missing_features = set(settings.FEATURE_NAMES) - set(df_processed.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # select only required features in correct order
    df_processed = df_processed[settings.FEATURE_NAMES]
    
    # fill missing values with medians
    for feature_name in settings.FEATURE_NAMES:
        df_processed[feature_name].fillna(FEATURE_MEDIANS[feature_name], inplace=True)
    
    # convert to numeric (handle any string values)
    for feature_name in settings.FEATURE_NAMES:
        df_processed[feature_name] = pd.to_numeric(
            df_processed[feature_name], 
            errors='coerce'
        )
    
    # final check: fill any remaining NaN with medians
    for feature_name in settings.FEATURE_NAMES:
        df_processed[feature_name].fillna(FEATURE_MEDIANS[feature_name], inplace=True)
    
    return df_processed


def validate_feature_ranges(row_dict: Dict) -> bool:
    """
    Validate that feature values are within reasonable ranges
    
    Args:
        row_dict: Dictionary with feature values
        
    Returns:
        True if all values are valid, raises ValueError otherwise
    """
    # define reasonable ranges (based on Kepler data)
    ranges = {
        "koi_period": (0.1, 10000),  # days
        "koi_time0bk": (0, 2000),  # BKJD
        "koi_impact": (0, 2),  # dimensionless
        "koi_duration": (0.1, 100),  # hours
        "koi_depth": (1, 100000),  # ppm
        "koi_prad": (0.1, 100),  # Earth radii
        "koi_teq": (100, 5000),  # Kelvin
        "koi_insol": (0.01, 10000),  # Earth flux
        "koi_model_snr": (1, 1000),  # dimensionless
        "koi_steff": (2000, 10000),  # Kelvin
        "koi_slogg": (2, 6),  # log10(cm/s²)
        "koi_srad": (0.1, 50),  # solar radii
        "koi_kepmag": (5, 20)  # magnitude
    }
    
    for feature_name, (min_val, max_val) in ranges.items():
        value = row_dict.get(feature_name)
        if value is not None and not np.isnan(value):
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Feature '{feature_name}' value {value} is outside "
                    f"reasonable range [{min_val}, {max_val}]"
                )
    
    return True


def normalize_features(df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    """
    Normalize features using standard scaling (optional)
    
    Note: If model was trained on normalized data, apply same scaler
    
    Args:
        df: Input dataframe
        scaler: Pre-fitted scaler object (sklearn)
        
    Returns:
        Normalized dataframe
    """
    if scaler is None:
        # if no scaler provided, return as-is
        # (assumes model was trained on unnormalized data)
        return df
    
    df_normalized = df.copy()
    df_normalized[settings.FEATURE_NAMES] = scaler.transform(df[settings.FEATURE_NAMES])
    
    return df_normalized


def extract_features_from_row(row: pd.Series) -> Dict:
    """
    Extract feature dictionary from pandas Series
    
    Args:
        row: Pandas Series with feature values
        
    Returns:
        Dictionary with feature names and values
    """
    return {
        feature_name: row[feature_name]
        for feature_name in settings.FEATURE_NAMES
        if feature_name in row.index
    }