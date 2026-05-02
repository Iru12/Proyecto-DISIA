import argparse
import pandas as pd
import os
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

def entrenar_modelo(input_dir, output_dir):

    # Comprobación de la carpeta de entrada
    if not os.path.isdir(input_dir):
        print(f"Error: El directorio de entrada no existe: {input_dir}")
        return False
    
    # Comprobación de la carpeta de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta de salida creada: {output_dir}")

    print(f"Iniciando el entrenamiento del modelo con los datos en: {input_dir}")

    try:
        X_train = pd.read_csv(f"{input_dir}/X_train_final.csv").astype(np.float32)
        y_train_type = pd.read_csv(f"{input_dir}/y_train_class1.csv")["class1"].astype(str)

        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train_type)

        X_train_np = X_train.to_numpy(dtype=np.float32)

        # Modelo a entrenar (Random Forest)
        model_name = "Random Forest"
        spec = {
            "builder": lambda: RandomForestClassifier(
                n_estimators=120, random_state=42, n_jobs=1, min_samples_leaf=2, class_weight="balanced_subsample"
            )
        }

        print(f"Entrenando {model_name}...")
        start_time = time.time()

        model = spec["builder"]()
        model.fit(X_train_np, y_train_enc)

        fit_seconds = time.time() - start_time
        print(f"{model_name} entrenado en {fit_seconds:.2f} segundos.")

        # Guardar el modelo entrenado
        joblib.dump(model, f"{output_dir}/RF_model.joblib")
        print("Entrenamiento completado exitosamente.")
    except Exception as e:
        print(f"Error durante el entrenamiento del modelo: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo")
    parser.add_argument(
        "--input",
        type=str, 
        default = "datos/preprocesados", 
        help = "Ruta del directorio con los datos preprocesados"
    )
    parser.add_argument(
        "--output",
        type=str,
        default = "modelos",
        help = "Ruta del directorio donde se guardará el modelo entrenado"
    )

    args = parser.parse_args()
    print(f"Ruta de los datos preprocesados: {args.input}")
    print(f"Ruta donde se guardará el modelo entrenado: {args.output}")
    entrenar_modelo(args.input, args.output)