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

def decade_classify(year):
    if year >= 1922 and year <= 1929:
        return 1
    elif year >= 1930 and year <= 1939:
        return 2
    elif year >= 1940 and year <= 1949:
        return 3
    elif year >= 1950 and year <= 1959:
        return 4
    elif year >= 1960 and year <= 1969:
        return 5
    elif year >= 1970 and year <= 1979:
        return 6
    elif year >= 1980 and year <= 1989:
        return 7
    elif year >= 1990 and year <= 1999:
        return 8
    elif year >= 2000 and year <= 2011:
        return 9
    else:
        return -1

def tempo_transform(tempo):
    if tempo >= 200:
        return tempo / 2
    else:
        return tempo

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
    df = pd.DataFrame([data.model_dump()])
    df["year"] = df["year"].apply(decade_classify)
    df["tempo"] = df["tempo"].apply(tempo_transform)
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
