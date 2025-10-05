"""
Configuration settings for WorldAway backend
"""


# Backend Setup (FastAPI):
# To start the backend server, make sure you have all dependencies installed, then run:
# # cd backend
# python -m uvicorn main:app --


# Frontend Setup (React):
# To start the frontend (React) application, open the frontend folder and run:
# cd frontend
# npm install
# npm start

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "WorldAway - Exoplanet Seeker API"
    API_VERSION: str = "1.0.0"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ]
    
    # file Upload Settings
    MAX_UPLOAD_SIZE: int = 209715200  # 200MB in bytes
    MAX_ROWS_PER_FILE: int = 10000
    ALLOWED_EXTENSIONS: List[str] = [".xlsx", ".xls", ".csv"]
    
    # processing Settings
    BATCH_SIZE: int = 1000
    
    # model Settings
    MODEL_PATH: str = "models/exoplanet_model.pkl"
    
    # feature Names (must match training data)
    FEATURE_NAMES: List[str] = [
        "koi_period",
        "koi_time0bk",
        "koi_impact",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_teq",
        "koi_insol",
        "koi_model_snr",
        "koi_steff",
        "koi_slogg",
        "koi_srad",
        "koi_kepmag"
    ]
    
    # storage Directories
    UPLOADS_DIR: str = "storage/uploads"
    JOBS_DIR: str = "storage/jobs"
    
    # redis (optional)
    REDIS_URL: str = "redis://localhost:6379/0"
    USE_CELERY: bool = False
    
    # rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# create settings instance
settings = Settings()


# feature median values for imputation (calculated from training data)
FEATURE_MEDIANS = {
    "koi_period": 10.5,
    "koi_time0bk": 134.5,
    "koi_impact": 0.3,
    "koi_duration": 5.2,
    "koi_depth": 150.0,
    "koi_prad": 1.1,
    "koi_teq": 550.0,
    "koi_insol": 200.0,
    "koi_model_snr": 12.5,
    "koi_steff": 5700.0,
    "koi_slogg": 4.4,
    "koi_srad": 1.0,
    "koi_kepmag": 13.5
}


# class names:
# CLASS_NAMES = ["FALSE_POSITIVE", "CANDIDATE", "CONFIRMED"]
CLASS_NAMES = [0, 1, 2]

