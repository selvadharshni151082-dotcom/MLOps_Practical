import joblib
from sklearn.datasets import load_breast_cancer

# Load the trained model
model = joblib.load("models/model.pkl")

# Load one sample for prediction
data = load_breast_cancer(as_frame=True)
sample = data.data.iloc[[0]]

# Make prediction
prediction = model.predict(sample)[0]

if prediction == 0:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")