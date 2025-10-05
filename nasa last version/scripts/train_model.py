"""
Exoplanet Model Training Script (Reference)
This script demonstrates how to train the classification model

Note: This is a reference script. The actual model should be pre-trained
and placed in backend/models/exoplanet_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


# feature names (must match application config)
FEATURE_NAMES = [
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_insol",
    "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]


def load_kepler_data(filepath: str) -> pd.DataFrame:
    """
    Load Kepler exoplanet data
    
    Args:
        filepath: Path to CSV file with Kepler data
        
    Returns:
        DataFrame with cleaned data
    """
    print("Loading Kepler data...")
    df = pd.read_csv(filepath)
    
    # select relevant columns
    columns_to_keep = FEATURE_NAMES + ['koi_disposition']
    df = df[columns_to_keep]
    
    # map disposition to simplified labels
    disposition_map = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE_POSITIVE'
    }
    df['label'] = df['koi_disposition'].map(disposition_map)
    
    # remove rows with unknown disposition
    df = df[df['label'].notna()]
    
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess data for training
    
    Args:
        df: Input DataFrame
        
    Returns:
        X, y arrays
    """
    print("\nPreprocessing data...")
    
    # calculate feature medians for later use
    feature_medians = {}
    for feature in FEATURE_NAMES:
        feature_medians[feature] = df[feature].median()
    
    print("Feature medians (for imputation):")
    for feature, median in feature_medians.items():
        print(f"  {feature}: {median:.2f}")
    
    # fill missing values with medians
    df_clean = df.copy()
    for feature in FEATURE_NAMES:
        df_clean[feature].fillna(feature_medians[feature], inplace=True)
    
    # prepare X and y
    X = df_clean[FEATURE_NAMES].values
    y = df_clean['label'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, feature_medians


def train_model(X, y, model_type='random_forest'):
    """
    Train classification model
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model to train
        
    Returns:
        Trained model
    """
    print(f"\nTraining {model_type} model...")
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
           
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # train model
    model.fit(X_train, y_train)
    
    # evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nTop 10 Feature Importances:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(FEATURE_NAMES))):
            idx = indices[i]
            print(f"  {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")
    
    # cross-validation
    print("\nCross-validation (5-fold):")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    print(f"CV F1-Scores: {cv_scores}")
    print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model


def save_model(model, filepath: str):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        filepath: Path to save model
    """
    print(f"\nSaving model to {filepath}...")
    joblib.dump(model, filepath)
    print("✓ Model saved successfully")


def test_saved_model(filepath: str):
    """
    Test loading and using the saved model
    
    Args:
        filepath: Path to saved model
    """
    print(f"\nTesting saved model from {filepath}...")
    
    # load model
    model = joblib.load(filepath)
    print("✓ Model loaded successfully")
    
    # test prediction with sample data
    test_sample = [[
        10.5, 134.5, 0.3, 5.2, 150.0, 1.1, 550.0,
        200.0, 12.5, 5700.0, 4.4, 1.0, 13.5
    ]]
    
    prediction = model.predict(test_sample)
    probabilities = model.predict_proba(test_sample)
    
    print(f"\nTest Prediction:")
    print(f"  Predicted class: {prediction[0]}")
    print(f"  Probabilities: {dict(zip(model.classes_, probabilities[0]))}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("WorldAway - Exoplanet Model Training")
    print("=" * 60)
    
    # note: You need to download Kepler data first
    # Dataset available at: https://exoplanetarchive.ipac.caltech.edu/
    DATA_PATH = "data/kepler_data.csv"  # update with actual path
    MODEL_PATH = "../backend/models/exoplanet_model.pkl"
    
    try:
        # load data
        df = load_kepler_data(DATA_PATH)
        
        # preprocess
        X, y, feature_medians = preprocess_data(df)
        
        # train model
        model = train_model(X, y, model_type='random_forest')
        
        # save model
        save_model(model, MODEL_PATH)
        
        # test saved model
        test_saved_model(MODEL_PATH)
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"\nModel saved at: {MODEL_PATH}")
        print("You can now use this model in the WorldAway application.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo train the model, you need to:")
        print("1. Download Kepler exoplanet data from NASA")
        print("2. Update DATA_PATH in this script")
        print("3. Run this script again")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()