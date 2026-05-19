import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


def entrenar_modelo(
    input_dir,
    output_dir,
    model_version,
    n_estimators,
    min_samples_leaf,
    random_state,
    class_weight,
):
    if not os.path.isdir(input_dir):
        print(f"Error: El directorio de entrada no existe: {input_dir}")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta de salida creada: {output_dir}")

    print(f"Iniciando el entrenamiento del modelo con los datos en: {input_dir}")

    try:
        X_train = pd.read_csv(f"{input_dir}/X_train_final.csv").astype(np.float32)
        X_val = pd.read_csv(f"{input_dir}/X_val_final.csv").astype(np.float32)
        y_train_type = pd.read_csv(f"{input_dir}/y_train_class1.csv")["class1"].astype(str)
        y_val_type = pd.read_csv(f"{input_dir}/y_val_class1.csv")["class1"].astype(str)

        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train_type)
        y_val_enc = label_encoder.transform(y_val_type)

        X_train_np = X_train.to_numpy(dtype=np.float32)
        X_val_np = X_val.to_numpy(dtype=np.float32)

        model_name = "Random Forest"
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )

        print(f"Entrenando {model_name}...")
        start_time = time.time()
        model.fit(X_train_np, y_train_enc)
        y_val_pred = model.predict(X_val_np)
        fit_seconds = time.time() - start_time
        print(f"{model_name} entrenado en {fit_seconds:.2f} segundos.")

        metrics = {
            "accuracy": float(accuracy_score(y_val_enc, y_val_pred)),
            "f1_macro": float(f1_score(y_val_enc, y_val_pred, average="macro")),
            "f1_weighted": float(f1_score(y_val_enc, y_val_pred, average="weighted")),
            "classification_report": classification_report(
                y_val_enc,
                y_val_pred,
                labels=list(range(len(label_encoder.classes_))),
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        }

        created_at = datetime.now(timezone.utc).isoformat()
        model_filename = f"{model_version}.joblib"
        model_path = os.path.join(output_dir, model_filename)
        default_model_path = os.path.join(output_dir, "RF_model.joblib")

        metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "model_file": model_filename,
            "default_model_file": "RF_model.joblib",
            "created_at": created_at,
            "fit_seconds": fit_seconds,
            "features_count": int(X_train.shape[1]),
            "classes": label_encoder.classes_.tolist(),
            "params": {
                "n_estimators": n_estimators,
                "min_samples_leaf": min_samples_leaf,
                "random_state": random_state,
                "class_weight": class_weight,
            },
            "metrics": {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
            },
        }

        joblib.dump(model, model_path)
        joblib.dump(model, default_model_path)
        joblib.dump(label_encoder, os.path.join(output_dir, f"{model_version}_label_encoder.joblib"))

        with open(os.path.join(output_dir, f"{model_version}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(output_dir, f"{model_version}_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(output_dir, "current_model_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("Entrenamiento completado exitosamente.")
        print(f"Modelo versionado guardado en: {model_path}")
        print(f"Alias activo guardado en: {default_model_path}")
        return True
    except Exception as e:
        print(f"Error durante el entrenamiento del modelo: {e}")
        return False


if __name__ == "__main__":
    app_root = os.getenv("APP_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir = os.getenv("DATA_DIR", os.path.join(app_root, "datos", "preprocesados"))
    model_dir = os.getenv("MODEL_DIR", os.path.join(app_root, "modelos"))

    parser = argparse.ArgumentParser(description="Entrenamiento del modelo")
    parser.add_argument(
        "--input",
        type=str,
        default=data_dir,
        help="Ruta del directorio con los datos preprocesados",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=model_dir,
        help="Ruta del directorio donde se guardara el modelo entrenado",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=os.getenv("MODEL_VERSION", "RF_v1"),
        help="Version del modelo que se guardara en el directorio de salida",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=int(os.getenv("N_ESTIMATORS", "120")),
        help="Numero de arboles del Random Forest",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=int(os.getenv("MIN_SAMPLES_LEAF", "2")),
        help="Minimo de muestras por hoja",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
        help="Semilla para reproducibilidad",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default=os.getenv("CLASS_WEIGHT", "balanced_subsample"),
        help="Peso de clases usado por RandomForestClassifier",
    )

    args = parser.parse_args()
    print(f"Ruta de los datos preprocesados: {args.input}")
    print(f"Ruta donde se guardara el modelo entrenado: {args.output}")
    print(f"Version del modelo: {args.model_version}")

    if not entrenar_modelo(
        input_dir=args.input,
        output_dir=args.output,
        model_version=args.model_version,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight=args.class_weight,
    ):
        sys.exit(1)
