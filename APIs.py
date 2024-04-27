from fastapi import FastAPI, status, HTTPException, Body, Response
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import json

filename = 'Data/retencion-por-mes.pkl'
model = joblib.load(filename)

app = FastAPI(
    title="APIs retención clientes",
    version="0.0.1"
 )

#API para ingresar información de un cliente en particular y obtener una predicción

@app.post("/api/v1/predict/")
async def predict(data: dict = Body(...)):
    try:
        df_data = pd.DataFrame([data])
        predictions = model.predict(df_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

