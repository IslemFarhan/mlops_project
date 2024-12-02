from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import mlflow
import pandas as pd
from fastapi import FastAPI
import dagshub
app = FastAPI()
import json


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon application FastAPI"}
# Charger le modèle


# Initialize DagsHub and MLflow integration
dagshub.init(repo_owner='RimMth', repo_name='mlops_project', mlflow=True)

mlflow.set_experiment("mlops")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/RimMth/mlops_project.mlflow") 

reports_path = "models/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)
model_name = run_info['model_name'] 


try:
    # Create an MlflowClient to interact with the MLflow server
    client = mlflow.tracking.MlflowClient()

    # Get the latest version of the model in the Production stage
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if versions:
        latest_version = versions[0].version
        run_id = versions[0].run_id  # Fetching the run ID from the latest version
        print(f"Latest version in Production: {latest_version}, Run ID: {run_id}")

        # Construct the logged_model string
        logged_model = f'runs:/{run_id}/{model_name}'
        print("Logged Model:", logged_model)

        # Load the model using the logged_model variable
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print(f"Model loaded from {logged_model}")

    else:
        print("No model found in the 'Production' stage.")

except Exception as e:
    print(f"Error fetching model: {e}")

def preprocess_health_data(df):
    # Mapping for Diabetes
    diabetes_mapping = {
        'No': 0, 
        'No, pre-diabetes or borderline diabetes': 0, 
        'Yes, but female told only during pregnancy': 1,
        'Yes': 1
    }
    df['Diabetes'] = df['Diabetes'].map(diabetes_mapping)

    # Encoding Sex: 0 for female, 1 for male
    df['Sex'] = df['Sex'].astype(str).str.strip()

    df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})
    # Convert remaining categorical variables with "Yes" and "No" values to binary format for correlation computation
    binary_columns = [ 'Skin_Cancer', 'Other_Cancer', 'Arthritis', 'Smoking_History', 'Exercise']
    
    for column in binary_columns:
        df[column] = df[column].map({'Yes': 1, 'No': 0})

    # Ordinal encoding for General_Health, Age_Category, BMI_Category
    general_health_mapping = {
        'Poor': 0,
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Excellent': 4
    }
    df['General_Health'] = df['General_Health'].map(general_health_mapping)

   

    age_category_mapping = {
        '18-24': 0,
        '25-29': 1,
        '30-34': 2,
        '35-39': 3,
        '40-44': 4,
        '45-49': 5,
        '50-54': 6,
        '55-59': 7,
        '60-64': 8,
        '65-69': 9,
        '70-74': 10,
        '75-79': 11,
        '80+': 12
    }
    df['Age_Category'] = df['Age_Category'].map(age_category_mapping)
     # Ordinal encoding for Checkup
    checkup_mapping = {
        'Never': 0,
        '5 or more years ago': 1,
        'Within the past 5 years': 2,
        'Within the past 2 years': 3,
        'Within the past year': 4
    }
    df['Checkup'] = df['Checkup'].map(checkup_mapping)


    return df

def scale_data(df):
    with open('models/standard_scaler.pkl', 'rb') as file:
        scalar = pickle.load(file)
    df = scalar.transform(df)

    return df

class HealthDataInput(BaseModel):
    general_health: int
    checkup: int
    exercise: float
    skin_cancer: int
    other_cancer: int
    diabetes: int
    arthritis: int
    sex: int
    age_category: int
    height_cm: float
    weight_kg: float
    bmi: float
    smoking_history: int
    alcohol_consumption: float
    fruit_consumption: int
    green_vegetables_consumption: int

    class Config:
        json_schema_extra = {
            "example": {
                "general_health": 3,
                "checkup": "Yes",
                "exercise": "Yes",
                "skin_cancer": "Yes",
                "other_cancer": 0,
                "diabetes": "Yes",
                "arthritis": 0,
                "sex": "Yes",
                "age_category": 5,
                "height_cm": 170.5,
                "weight_kg": 75.3,
                "bmi": 25.9,
                "smoking_history": 2,
                "alcohol_consumption": 1.3,
                "fruit_consumption": 3,
                "green_vegetables_consumption": 4
            }
        }


# Endpoint pour les prédictions
@app.post("/predict")
def predict_health_risk(data: HealthDataInput):
    # Préparer les données pour le modèle
    data = pd.DataFrame(data.dict(), index=[0])

    features = ['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer',
                'Other_Cancer', 'Diabetes', 'Arthritis', 'Sex',
                'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History',
                'Alcohol_Consumption', 'Fruit_Consumption',
                'Green_Vegetables_Consumption']

    # Conversion en DataFrame
    df = pd.DataFrame(data, columns=features)

    df=preprocess_health_data(df)
    df=scale_data(df)
    df = pd.DataFrame(data, columns=features)


    # Obtenir la prédiction et la probabilité
    prediction = loaded_model.predict(df)[0]

    # Return results as JSON-compatible
    return {"prediction": int(prediction)}
