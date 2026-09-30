from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
import os


app = FastAPI(
    title="Linear Regression API",
    description="Salary prediction using Years of Experience",
    version="1.0"
)


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "model.pkl"
DATA_PATH = "train.csv"


# Request data format
class PredictionRequest(BaseModel):
    years_experience: float


# Train model if model.pkl does not exist
def train_model():

    data = pd.read_csv(DATA_PATH)

    X = data[["YearsExperience"]]
    y = data["Salary"]

    model = LinearRegression()

    model.fit(X, y)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return model


# Load model
def load_model():

    if os.path.exists(MODEL_PATH):

        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        return model

    return train_model()


model = load_model()


@app.get("/")
def home():

    return {
        "message": "Linear Regression API is running",
        "endpoint": "/predict"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    prediction = model.predict(
        [[request.years_experience]]
    )

    predicted_salary = prediction[0]

    return {
        "years_experience": request.years_experience,
        "predicted_salary": round(float(predicted_salary), 2)
    }