# Cancer Prediction System using MLOps

## Project Overview
This project uses Machine Learning and MLOps tools to train a classification model and deploy it as a REST API.

The model uses the Scikit-learn Breast Cancer dataset.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- MLflow
- FastAPI
- Docker
- Git

## Project Workflow
1. Load the dataset.
2. Split the data into training and testing sets.
3. Train a Random Forest classification model.
4. Evaluate using Accuracy, Precision, and Recall.
5. Track experiments using MLflow.
6. Save the trained model.
7. Create a prediction API using FastAPI.
8. Build and run a Docker container.
9. Track files using Git.

## Model Evaluation
- Accuracy: 0.9649
- Precision: 0.9589
- Recall: 0.9859

## Run the API
Install dependencies:

    pip install -r requirements.txt

Start the API:

    uvicorn src.app:app --reload

Open: http://127.0.0.1:8000/docs

## Run with Docker
Build:

    docker build -t cancer-prediction-api .

Run:

    docker run -d --name cancer-api -p 8000:8000 cancer-prediction-api

## Disclaimer
This project is for educational purposes only. It is not a medical diagnostic tool.
