from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


import warnings
from sklearn.exceptions import UserWarning


app = FastAPI()


origins = [
    "https://gemelo-digital-fipiqq.flutterflow.app", # Tu app publicada
    "app://flutterflow.io",                          # El Test Mode de FlutterFlow
    "*"                                              # Comodín
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("modelo_tdah.pkl")


class DatosInput(BaseModel):
    atencion: float
    impulsividad: float
    cortisol: float
    sueno: float

@app.get("/")
def home():
    return {"mensaje": "Gemelo Digital IA activo"}


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# -------------------------------------------------------------

@app.post("/predecir/")
def predecir(datos: DatosInput):
   
    valores = np.array([[datos.atencion, datos.impulsividad, datos.cortisol, datos.sueno]])
    
    
    resultado = model.predict(valores)
    
 
    return {"ansiedad_predicha": int(resultado[0])}
