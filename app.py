import os
import zipfile
import gdown

MODEL_FILE_ID = "17I0sKjrGgW4EGhtv2Q0eB5rVp9x85nX0"
ZIP_PATH = "models.zip"

if not os.path.exists(ZIP_PATH):
    print("Downloading models from Google Drive...")
    gdown.download(
        url=f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
        output=ZIP_PATH,
        quiet=False
    )

if not os.path.exists("models"):
    print("Extracting models...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI(title="Fraud Detection REST API")


expert_procurement = joblib.load("models/expert_procurement_xgb.pkl")
expert_spending = joblib.load("models/expert_spending_iso.pkl")
expert_visual = joblib.load("models/expert_visual_rf.pkl")
meta_learner = joblib.load("models/meta_learner_lr.pkl")


class FraudRequest(BaseModel):
    data: list[float]


@app.get("/")
def health_check():
    return {
        "status": "API running",
        "message": "Fraud Detection REST API is live"
    }

@app.post("/predict")
def predict_fraud(request: FraudRequest):

    data = request.data

    if len(data) == 0:
        raise


