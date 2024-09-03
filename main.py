import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

model = joblib.load("./MSD_RFR_PredictiveModel.pkl")
scaler = joblib.load("./MSD_RFR_Scaler.pkl")

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://beatprophet.wuiquique.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SongModel(BaseModel):
    year: int
    duration: float
    key: int
    loudness: float
    mode: int
    tempo: float
    signature: int

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(data: SongModel):
    df = pd.DataFrame([data.dict()])
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scaled)

    response = {
        "prediction": prediction.tolist()[0],
        "shap_values": {
            "year": shap_values.tolist()[0][0],
            "duration": shap_values.tolist()[0][1],
            "key": shap_values.tolist()[0][2],
            "loudness": shap_values.tolist()[0][3],
            "mode": shap_values.tolist()[0][4],
            "tempo": shap_values.tolist()[0][5],
            "signature": shap_values.tolist()[0][6]
        }
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
