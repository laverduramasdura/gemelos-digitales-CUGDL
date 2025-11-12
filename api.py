
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "https://gemelo-digital-fipiqq.flutterflow.app", 
    "app://flutterflow.io",                        
    "*"                                           
]

app.add_middleware(  
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

