# ======================================
# Entrenamiento del modelo TDAH Mixto / Emocional
# ======================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Simulación de datos de ejemplo
np.random.seed(42)
n = 200

data = pd.DataFrame({
    "atencion": np.random.uniform(1, 10, n),
    "impulsividad": np.random.uniform(1, 10, n),
    "cortisol": np.random.uniform(1, 10, n),
    "dopamina": np.random.uniform(1, 10, n),
    "sueno": np.random.uniform(1, 10, n)
})

# Regla simple para crear niveles simulados (solo ejemplo)
data["nivel_disregulacion_emocional"] = np.where(
    (data["impulsividad"] > 6) & (data["cortisol"] > 6),
    2,  # Alta
    np.where((data["impulsividad"] > 4), 1, 0)  # Moderada o Baja
)

# Separar variables y etiquetas
X = data[["atencion", "impulsividad", "cortisol", "dopamina", "sueno"]]
y = data["nivel_disregulacion_emocional"]

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Guardar modelo
joblib.dump(model, "modelo_tdah.pkl")

print("✅ Modelo entrenado y guardado correctamente como modelo_tdah.pkl")
