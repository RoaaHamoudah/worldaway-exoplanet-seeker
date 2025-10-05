"""
Model explainability utilities using SHAP
"""

import shap
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def get_shap_values(model, X, background_samples: int = 100):
    """
    Calculate SHAP values for model predictions
    
    Args:
        model: Trained model
        X: Input data (list or array)
        background_samples: Number of background samples for TreeExplainer
        
    Returns:
        SHAP values
    """
    try:
        # convert to numpy array if needed
        if isinstance(X, list):
            X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
        
        # try TreeExplainer first (faster for tree-based models)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
        except Exception:
            # fall back to KernelExplainer for other model types
            logger.info("TreeExplainer failed, using KernelExplainer")
            
            # create background dataset (sample from training data)
            # for now, use zeros as background (in production, use actual training data)
            background = np.zeros((background_samples, X.shape[1]))
            
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X)
        
        return shap_values
        
    except Exception as e:
        logger.error(f"SHAP calculation failed: {str(e)}")
        # return None to indicate failure
        return None


def get_top_features(model, X, feature_names: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Get top N features that contributed most to the prediction
    
    Args:
        model: Trained model
        X: Input data (single row)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        List of tuples (feature_name, importance_score)
    """
    try:
        # get SHAP values
        shap_values = get_shap_values(model, [X])
        
        if shap_values is None:
            # fallback to feature importances if SHAP fails
            return get_feature_importances_fallback(model, feature_names, top_n)
        
        # for multi-class, use the predicted class SHAP values
        prediction = model.predict([X])[0]
        class_idx = list(model.classes_).index(prediction)
        
        if isinstance(shap_values, list):
            # multi-class model returns list of arrays
            class_shap_values = shap_values[class_idx][0]
        else:
            # binary or single output
            class_shap_values = shap_values[0]
        
        # get absolute values for importance ranking
        abs_shap_values = np.abs(class_shap_values)
        
        # get top N indices
        top_indices = np.argsort(abs_shap_values)[-top_n:][::-1]
        
        # create list of (feature_name, importance) tuples
        top_features = [
            (feature_names[idx], float(abs_shap_values[idx]))
            for idx in top_indices
        ]
        
        return top_features
        
    except Exception as e:
        logger.error(f"Failed to get top features: {str(e)}")
        return get_feature_importances_fallback(model, feature_names, top_n)


def get_feature_importances_fallback(model, feature_names: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Fallback method using model's feature_importances_ if available
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        List of tuples (feature_name, importance_score)
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # get top N indices
            top_indices = np.argsort(importances)[-top_n:][::-1]
            
            # create list of (feature_name, importance) tuples
            top_features = [
                (feature_names[idx], float(importances[idx]))
                for idx in top_indices
            ]
            
            return top_features
    except Exception as e:
        logger.error(f"Fallback feature importance failed: {str(e)}")
    
    # if all else fails, return uniform importance
    return [(feature_names[i], 1.0/len(feature_names)) for i in range(min(top_n, len(feature_names)))]


def generate_shap_summary(model, X, feature_names: List[str]) -> dict:
    """
    Generate comprehensive SHAP summary for visualization
    
    Args:
        model: Trained model
        X: Input data
        feature_names: List of feature names
        
    Returns:
        Dictionary with SHAP summary data
    """
    try:
        shap_values = get_shap_values(model, X)
        
        if shap_values is None:
            return {"error": "SHAP calculation failed"}
        
        # get prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # format for each class
        class_summaries = {}
        
        for class_idx, class_name in enumerate(model.classes_):
            if isinstance(shap_values, list):
                class_shap = shap_values[class_idx]
            else:
                class_shap = shap_values
            
            # create feature importance list
            feature_contributions = []
            for i, feature_name in enumerate(feature_names):
                feature_contributions.append({
                    "feature": feature_name,
                    "shap_value": float(class_shap[0][i]),
                    "feature_value": float(X[0][i]) if hasattr(X[0], '__getitem__') else float(X[i])
                })
            
            # sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            
            class_summaries[class_name] = {
                "probability": float(probabilities[class_idx]),
                "feature_contributions": feature_contributions
            }
        
        return {
            "predicted_class": prediction,
            "class_summaries": class_summaries
        }
        
    except Exception as e:
        logger.error(f"SHAP summary generation failed: {str(e)}")
        return {"error": str(e)}


def get_shap_force_plot_data(model, X, feature_names: List[str]) -> dict:
    """
    Get data for SHAP force plot visualization
    
    Args:
        model: Trained model
        X: Input data (single instance)
        feature_names: List of feature names
        
    Returns:
        Dictionary with force plot data
    """
    try:
        shap_values = get_shap_values(model, [X])
        
        if shap_values is None:
            return {"error": "SHAP calculation failed"}
        
        prediction = model.predict([X])[0]
        class_idx = list(model.classes_).index(prediction)
        
        if isinstance(shap_values, list):
            class_shap_values = shap_values[class_idx][0]
        else:
            class_shap_values = shap_values[0]
        
        # separate positive and negative contributions
        positive_features = []
        negative_features = []
        
        for i, feature_name in enumerate(feature_names):
            contribution = float(class_shap_values[i])
            feature_value = float(X[i])
            
            feature_data = {
                "feature": feature_name,
                "value": feature_value,
                "contribution": contribution
            }
            
            if contribution > 0:
                positive_features.append(feature_data)
            else:
                negative_features.append(feature_data)
        
        # sort by absolute contribution
        positive_features.sort(key=lambda x: x["contribution"], reverse=True)
        negative_features.sort(key=lambda x: x["contribution"])
        
        return {
            "predicted_class": prediction,
            "base_value": 1.0 / len(model.classes_),  # Uniform prior
            "positive_contributions": positive_features,
            "negative_contributions": negative_features,
            "final_prediction": float(model.predict_proba([X])[0][class_idx])
        }
        
    except Exception as e:
        logger.error(f"Force plot data generation failed: {str(e)}")
        return {"error": str(e)}