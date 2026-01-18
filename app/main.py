from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pickle
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "dev", "models")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "income(1).csv")

# Load models and feature names
knn_model = None
logistic_model = None
scaler = None
feature_names = None

def load_models():
    global knn_model, logistic_model, scaler, feature_names
    try:
        with open(KNN_MODEL_PATH, 'rb') as f:
            knn_model = pickle.load(f)
        
        logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on startup
load_models()

# Load dataset to get unique values
def get_unique_values():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset file not found at: {DATA_PATH}")
        
        data = pd.read_csv(DATA_PATH, na_values=[" ?"])
        data = data.dropna(axis=0)
        
        if data.empty:
            raise ValueError("Dataset is empty after dropping null values")
        
        unique_values = {
            "JobType": sorted(data['JobType'].dropna().unique().tolist()),
            "EdType": sorted(data['EdType'].dropna().unique().tolist()),
            "maritalstatus": sorted(data['maritalstatus'].dropna().unique().tolist()),
            "occupation": sorted(data['occupation'].dropna().unique().tolist()),
            "relationship": sorted(data['relationship'].dropna().unique().tolist()),
            "race": sorted(data['race'].dropna().unique().tolist()),
            "gender": sorted(data['gender'].dropna().unique().tolist()),
            "nativecountry": sorted(data['nativecountry'].dropna().unique().tolist())
        }
        return unique_values
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise  # Re-raise to be caught by the endpoint handler

# Request models
class PredictionRequest(BaseModel):
    age: int
    JobType: str
    EdType: str
    maritalstatus: str
    occupation: str
    relationship: str
    race: str
    gender: str
    capitalgain: int
    capitalloss: int
    hoursperweek: int
    nativecountry: str
    model_type: str  # "knn" or "logistic"

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors"""
    return Response(status_code=204)

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/api/unique-values")
async def get_unique_values_endpoint():
    """Get unique values for categorical fields"""
    try:
        unique_vals = get_unique_values()
        if not unique_vals or len(unique_vals) == 0:
            raise HTTPException(status_code=500, detail="Failed to load unique values from dataset")
        return unique_vals
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Dataset file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading unique values: {str(e)}")

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Make prediction using selected model"""
    try:
        # Create a DataFrame with the input data
        input_data = {
            'age': [request.age],
            'JobType': [request.JobType],
            'EdType': [request.EdType],
            'maritalstatus': [request.maritalstatus],
            'occupation': [request.occupation],
            'relationship': [request.relationship],
            'race': [request.race],
            'gender': [request.gender],
            'capitalgain': [request.capitalgain],
            'capitalloss': [request.capitalloss],
            'hoursperweek': [request.hoursperweek],
            'nativecountry': [request.nativecountry]
        }
        
        df = pd.DataFrame(input_data)
        
        # One-hot encode the categorical variables (same as training)
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Get the feature names that the model expects
        if feature_names is None:
            raise HTTPException(status_code=500, detail="Feature names not loaded")
        
        # Create a feature vector with zeros for all features
        feature_vector = np.zeros(len(feature_names))
        
        # Create a dictionary for quick lookup of encoded values
        encoded_dict = {col: df_encoded[col].values[0] for col in df_encoded.columns}
        
        # Map the input features to the feature vector
        for i, feature_name in enumerate(feature_names):
            if feature_name in encoded_dict:
                feature_vector[i] = encoded_dict[feature_name]
            elif feature_name in ['age', 'capitalgain', 'capitalloss', 'hoursperweek']:
                feature_vector[i] = input_data[feature_name][0]
            # For one-hot encoded features that don't match, they remain 0 (which is correct)
        
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Make prediction based on model type
        if request.model_type.lower() == "knn":
            if knn_model is None:
                raise HTTPException(status_code=500, detail="KNN model not loaded")
            prediction = knn_model.predict(feature_vector)[0]
        elif request.model_type.lower() == "logistic":
            if logistic_model is None or scaler is None:
                raise HTTPException(status_code=500, detail="Logistic regression model or scaler not loaded")
            # Scale the features for logistic regression
            feature_vector_scaled = scaler.transform(feature_vector)
            prediction = logistic_model.predict(feature_vector_scaled)[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'knn' or 'logistic'")
        
        # Convert prediction to human-readable format
        result = "greater than 50,000" if prediction == 1 else "less than or equal to 50,000"
        
        return {
            "prediction": int(prediction),
            "result": result,
            "model_used": request.model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
