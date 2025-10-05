"""
Input validation utilities
"""

import os
from typing import Dict
from config import settings


def validate_file_type(filename: str) -> bool:
    """
    Validate file extension
    
    Args:
        filename: Name of uploaded file
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid file type: {file_extension}. "
            f"Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    return True


def validate_file_size(file_size: int) -> bool:
    """
    Validate file size
    
    Args:
        file_size: Size of file in bytes
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if file_size > settings.MAX_UPLOAD_SIZE:
        max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        actual_size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File size ({actual_size_mb:.2f} MB) exceeds maximum "
            f"allowed size ({max_size_mb:.2f} MB)"
        )
    
    if file_size == 0:
        raise ValueError("File is empty")
    
    return True


def validate_row_data(row_dict: Dict) -> bool:
    """
    Validate single row of exoplanet data
    
    Args:
        row_dict: Dictionary with feature values
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # check all required features are present
    missing_features = set(settings.FEATURE_NAMES) - set(row_dict.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # check for valid numeric values
    for feature_name in settings.FEATURE_NAMES:
        value = row_dict[feature_name]
        if value is not None:
            try:
                float(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Feature '{feature_name}' has invalid value: {value}. "
                    "Must be numeric."
                )
    
    return True