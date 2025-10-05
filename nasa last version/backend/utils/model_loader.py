"""
Model loader utility - Singleton pattern for model management
"""

import joblib
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton class to load and manage the ML model
    Ensures model is loaded only once at application startup
    """
    _instance = None
    _model = None
    
    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.model_path = model_path
        return cls._instance
    
    def load_model(self):
        """Load the model from disk"""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    "Please ensure the model file exists."
                )
            
            try:
                logger.info(f"Loading model from {self.model_path}")
                self._model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully. Type: {type(self._model).__name__}")
                
                # validate model has required methods
                if not hasattr(self._model, 'predict'):
                    raise ValueError("Loaded model does not have 'predict' method")
                if not hasattr(self._model, 'predict_proba'):
                    raise ValueError("Loaded model does not have 'predict_proba' method")
                
                # log model info
                if hasattr(self._model, 'classes_'):
                    logger.info(f"Model classes: {self._model.classes_}")
                if hasattr(self._model, 'n_features_in_'):
                    logger.info(f"Model expects {self._model.n_features_in_} features")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
        
        return self._model
    
    def get_model(self):
        """Get the loaded model (loads if not already loaded)"""
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def model(self):
        """Property to access model"""
        return self._model
    
    def reload_model(self):
        """Reload the model from disk (useful for model updates)"""
        logger.info("Reloading model...")
        self._model = None
        return self.load_model()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self._model is None:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "model_type": type(self._model).__name__,
            "model_path": self.model_path
        }
        
        # add optional attributes if available
        if hasattr(self._model, 'classes_'):
            info["classes"] = self._model.classes_.tolist()
        if hasattr(self._model, 'n_features_in_'):
            info["n_features"] = self._model.n_features_in_
        if hasattr(self._model, 'feature_names_in_'):
            info["feature_names"] = self._model.feature_names_in_.tolist()
        
        return info