import joblib
import pandas as pd

model = joblib.load("model.pkl")

sample = {
    'mean radius': [14.5],
    'mean texture': [20.5],
    'mean perimeter': [96.2],
    'mean area': [700],
    'mean smoothness': [0.09],
    # Add other required features...
}

df = pd.DataFrame(sample)
prediction = model.predict(df)[0]
print("✅ Tumor Type:", "Malignant" if prediction == 0 else "Benign")
