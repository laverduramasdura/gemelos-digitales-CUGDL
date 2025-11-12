from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("modelo_tdah.pkl")

# Aquí defines el BaseModel
class DatosInput(BaseModel):
    atencion: float
    impulsividad: float
    cortisol: float
    sueno: float

@app.get("/")
def home():
    return {"mensaje": "Gemelo Digital IA activo"}

@app.post("/predecir/")
def predecir(datos: DatosInput):
    valores = np.array([[datos.atencion, datos.impulsividad, datos.cortisol, datos.sueno]])
    resultado = model.predict(valores)
    return {"ansiedad_predicha": int(resultado[0])}

