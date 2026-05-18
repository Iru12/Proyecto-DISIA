# Recogemos las instancias que han pasado por el sistema de inferencia.
from datetime import datetime
import os
import shutil
import time
import pandas as pd
import json
import subprocess
import re
from pathlib import Path

def obtener_siguiente_version(model_dir="/app/modelos"):
    path = Path(model_dir)

    versiones = []

    for fichero in path.glob("RF_v*.joblib"):
        match = re.search(r"RF_v(\d+)\.joblib", fichero.name)

        if match:
            versiones.append(int(match.group(1)))

    siguiente = max(versiones, default=0) + 1

    return f"RF_v{siguiente}"

logs = []

with open(os.getenv("PREDICTIONS_LOG_PATH", "data_output/predicciones_inferencia.log"), "r", encoding="utf-8") as f:
    for linea in f:
        logs.append(json.loads(linea))

train_antiguo_path = os.getenv("TRAIN_ANTIGUO_PATH", "data_output/X_train_final.csv")
train_antiguo_path_y = os.getenv("TRAIN_ANTIGUO_PATH_Y", "data_output/y_train_class1.csv")

df_antiguo = pd.read_csv(train_antiguo_path)
df_antiguo_y = pd.read_csv(train_antiguo_path_y)

for log in logs:
    input = log["input"]
    prediccion = log["prediction"]

    df_nuevo = pd.DataFrame(input)
    df_nuevo["class1"] = prediccion

    df_antiguo = pd.concat([df_antiguo, df_nuevo.drop(columns=["class1"])], ignore_index=True)
    df_antiguo_y = pd.concat([df_antiguo_y, df_nuevo[["class1"]]], ignore_index=True)

df_antiguo.to_csv(os.getenv("TRAIN_ANTIGUO_PATH", "data_output/X_train_final.csv"), index=False)
df_antiguo_y.to_csv(os.getenv("TRAIN_ANTIGUO_PATH_Y", "data_output/y_train_class1.csv"), index=False)

version = obtener_siguiente_version()
print(f"Iniciando proceso de re-entrenamiento con la nueva versión: {version}...")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Para no perder aquellas instancias ya utilizadas para el entrenamiento, las movemos a un nuevo archivo con timestamp para auditoría
shutil.move(os.getenv("PREDICTIONS_LOG_PATH", "data_output/predicciones_inferencia.log"), f"/app/datos/preprocesados/predicciones_inferencia_{timestamp}.log") 
# Reseteamos los logs para evitar que se vuelvan a procesar en el siguiente ciclo
open(os.getenv("PREDICTIONS_LOG_PATH", "data_output/predicciones_inferencia.log"), "w").close()


subprocess.run(["python", "train.py", "--input", "/app/datos/preprocesados", "--output", "/app/modelos", "--model_version", version])
