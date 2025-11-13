# ==============================================
# 🧠 Gemelo Digital IA - TDAH Mixto (Emocional o Disfórico)
# Archivo: main.py — versión para Render + FlutterFlow
# ==============================================

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# ==============================================
# Inicialización de la aplicación
# ==============================================
app = FastAPI(title="Gemelo Digital TDAH", version="2.1")

# ==============================================
# Configuración CORS (para conexión FlutterFlow)
# ==============================================
origins = [
    "https://gemelo-digital-fipiqq.flutterflow.app",  # Tu dominio FlutterFlow
    "https://app.flutterflow.io",
    "*",  # Para pruebas locales o con Ngrok
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# Carga del modelo de IA entrenado
# ==============================================
try:
    model = joblib.load("modelo_tdah.pkl")
except Exception as e:
    model = None
    print("⚠️ Error al cargar modelo_tdah.pkl:", e)

# ==============================================
# Definición del esquema de entrada
# ==============================================
class DatosInput(BaseModel):
    atencion: float
    impulsividad: float
    cortisol: float
    dopamina: float
    sueno: float

# ==============================================
# Endpoint base (prueba de conexión)
# ==============================================
@app.get("/")
def home():
    return {
        "mensaje": "🧠 API Gemelo Digital TDAH Mixto / Emocional activa",
        "version": "2.1",
        "modelo": "Cargado correctamente" if model else "Error al cargar modelo",
    }

# ==============================================
# Endpoint principal de predicción
# ==============================================
@app.post("/predecir/")
def predecir(datos: DatosInput):
    """
    Recibe parámetros neuroconductuales y devuelve
    un diagnóstico predictivo del nivel de disregulación emocional.
    """

    if model is None:
        return {"error": "⚠️ Modelo no cargado en Render"}

    # Convertir datos a matriz NumPy
    entrada = np.array([[datos.atencion, datos.impulsividad, datos.cortisol, datos.dopamina, datos.sueno]])

    # Predicción
    prediccion = model.predict(entrada)[0]
    probabilidad = None

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(entrada)[0]
        probabilidad = round(float(probas[1] * 100), 2)

    # Clasificación según nivel
    if prediccion == 0:
        estado = "Regulado"
        mensaje = "Buen control emocional y atencional. Continúa tus hábitos saludables."
        color = "🟢 Verde"
    elif prediccion == 1:
        estado = "Disregulación Moderada"
        mensaje = "Leve desbalance emocional. Requiere pausas activas o ejercicios de mindfulness."
        color = "🟡 Amarillo"
    else:
        estado = "Disregulación Alta"
        mensaje = "Desbalance severo en dopamina/cortisol. Considerar apoyo psicológico o ajuste conductual."
        color = "🔴 Rojo"

    # Respuesta estructurada
    return {
        "nivel_tdah_emocional": int(prediccion),
        "estado": estado,
        "mensaje": mensaje,
        "color_referencia": color,
        "probabilidad": probabilidad,
    }
