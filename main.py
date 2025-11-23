# ==============================================
# 🧠 Gemelo Digital IA - TDAH Mixto (Emocional o Disfórico)
# Archivo: main.py — Versión optimizada para Render + FlutterFlow
# ==============================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

# ==============================================
# Inicialización de la aplicación
# ==============================================
app = FastAPI(title="Gemelo Digital TDAH", version="2.2")

# ==============================================
# Configuración CORS (Crítico para FlutterFlow)
# ==============================================
origins = [
    "https://gemelo-digital-fipiqq.flutterflow.app",  # Tu dominio FlutterFlow
    "https://app.flutterflow.io",
    "http://localhost:3000", # Útil para pruebas locales
    "*" # Puedes mantener esto para desarrollo, pero idealmente restriingelo en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Esto permite GET, POST, HEAD, OPTIONS, etc.
    allow_headers=["*"],
)

# ==============================================
# Carga del modelo de IA entrenado
# ==============================================
# Usamos una ruta relativa segura para evitar errores de ruta en Linux/Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_tdah.pkl")

try:
    # Intenta cargar el modelo. Si falla, no rompe el servidor, pero avisa.
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️ Error CRÍTICO al cargar el modelo: {e}")
    # Intento de carga alternativa si el archivo está en la raíz directa
    try:
        model = joblib.load("modelo_tdah.pkl")
        print("✅ Modelo cargado desde ruta raíz (fallback).")
    except:
        pass

# ==============================================
# Definición del esquema de entrada (Pydantic)
# ==============================================
class DatosInput(BaseModel):
    atencion: float
    impulsividad: float
    cortisol: float
    dopamina: float
    sueno: float

# ==============================================
# Endpoint Base + Health Check (Solución al error 405)
# ==============================================
@app.get("/")
@app.head("/") # <--- ESTO ARREGLA EL ERROR EN TUS LOGS DE RENDER
def home():
    """
    Endpoint de salud. Responde a GET y HEAD.
    Render usa HEAD para verificar que la app no se ha congelado.
    """
    return {
        "mensaje": "🧠 API Gemelo Digital TDAH Mixto / Emocional activa",
        "estado_servidor": "Online",
        "modelo_cargado": model is not None,
        "version": "2.2"
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
        # Retorna un error 503 (Servicio no disponible) si no hay modelo
        return {
            "error": "El modelo de IA no está disponible en este momento.",
            "detalle": "Verifica que modelo_tdah.pkl esté subido en Render."
        }

    try:
        # Convertir datos a matriz NumPy para el modelo
        entrada = np.array([[
            datos.atencion, 
            datos.impulsividad, 
            datos.cortisol, 
            datos.dopamina, 
            datos.sueno
        ]])

        # Predicción directa
        prediccion = model.predict(entrada)[0]
        
        # Cálculo de probabilidad (Confianza del modelo)
        probabilidad = 0.0
        if hasattr(model, "predict_proba"):
            # Obtenemos la probabilidad de la clase predicha
            probas = model.predict_proba(entrada)[0]
            # Asumimos que la clase 1 o 2 son las "activas", tomamos la más alta
            probabilidad = round(float(np.max(probas) * 100), 2)

        # Mapeo de resultados a lenguaje natural
        # Ajusta estos mensajes según la lógica de tu entrenamiento
        if prediccion == 0:
            estado = "Regulado"
            mensaje = "Buen control emocional y atencional. Continúa tus hábitos saludables."
            color = "#4CAF50" # Verde Hex
        elif prediccion == 1:
            estado = "Disregulación Moderada"
            mensaje = "Leve desbalance emocional. Se sugiere pausa activa o mindfulness."
            color = "#FFC107" # Amarillo Hex
        else: # Asumiendo 2 o más
            estado = "Disregulación Alta"
            mensaje = "Posible crisis de dopamina/cortisol. Considerar estrategia de contención."
            color = "#F44336" # Rojo Hex

        # Respuesta estructurada JSON
        return {
            "resultado_numerico": int(prediccion),
            "estado_texto": estado,
            "mensaje_recomendacion": mensaje,
            "color_alerta": color,
            "nivel_confianza": probabilidad,
            "input_recibido": datos.dict() # Útil para depurar si envías algo mal
        }

    except Exception as e:
        return {"error": f"Ocurrió un error al procesar la predicción: {str(e)}"}