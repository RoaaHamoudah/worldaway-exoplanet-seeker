import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 13)
y = np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'], 100)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/exoplanet_model.pkl')
print("Model created successfully!")