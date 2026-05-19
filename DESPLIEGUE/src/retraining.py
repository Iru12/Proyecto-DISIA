from datetime import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


SKIPPED_NOT_ENOUGH_VALIDATED = 2


def obtener_siguiente_version(model_dir):
    path = Path(model_dir)
    versiones = []

    for fichero in path.glob("RF_v*.joblib"):
        match = re.search(r"RF_v(\d+)\.joblib", fichero.name)
        if match:
            versiones.append(int(match.group(1)))

    siguiente = max(versiones, default=0) + 1
    return f"RF_v{siguiente}"


def leer_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def escribir_jsonl(path, registros):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def main():
    app_root = os.getenv("APP_ROOT", str(Path(__file__).resolve().parents[1]))
    data_dir = os.getenv(
        "RETRAINING_DATA_DIR",
        os.getenv("DATA_DIR", os.path.join(app_root, "datos", "preprocesados")),
    )
    model_dir = os.getenv(
        "RETRAINING_MODEL_DIR",
        os.getenv("MODEL_DIR", os.path.join(app_root, "modelos")),
    )
    validated_path = os.getenv(
        "RETRAINING_VALIDATED_PATH",
        os.path.join(data_dir, "retraining_validated.jsonl"),
    )
    min_validated_samples = int(os.getenv("RETRAINING_MIN_VALIDATED_SAMPLES", "50"))

    train_antiguo_path = os.getenv("TRAIN_ANTIGUO_PATH", os.path.join(data_dir, "X_train_final.csv"))
    train_antiguo_path_y = os.getenv("TRAIN_ANTIGUO_PATH_Y", os.path.join(data_dir, "y_train_class1.csv"))

    registros = leer_jsonl(validated_path)
    registros_aprobados = [
        registro
        for registro in registros
        if registro.get("review_status") == "approved"
        and registro.get("validated_label")
        and isinstance(registro.get("input"), dict)
    ]

    if len(registros_aprobados) < min_validated_samples:
        print(
            "Re-entrenamiento omitido: "
            f"{len(registros_aprobados)} muestras validadas disponibles; "
            f"minimo requerido {min_validated_samples}."
        )
        return SKIPPED_NOT_ENOUGH_VALIDATED

    df_antiguo = pd.read_csv(train_antiguo_path)
    df_antiguo_y = pd.read_csv(train_antiguo_path_y)

    nuevas_x = pd.DataFrame([registro["input"] for registro in registros_aprobados])
    nuevas_x = nuevas_x.reindex(columns=df_antiguo.columns, fill_value=0)
    nuevas_y = pd.DataFrame(
        {"class1": [registro["validated_label"] for registro in registros_aprobados]}
    )

    df_entrenamiento = pd.concat([df_antiguo, nuevas_x], ignore_index=True)
    df_entrenamiento_y = pd.concat([df_antiguo_y, nuevas_y], ignore_index=True)

    version = obtener_siguiente_version(model_dir)
    print(f"Iniciando proceso de re-entrenamiento con etiquetas validadas: {version}...")

    try:
        df_entrenamiento.to_csv(train_antiguo_path, index=False)
        df_entrenamiento_y.to_csv(train_antiguo_path_y, index=False)

        subprocess.run(
            [
                "python",
                "train.py",
                "--input",
                data_dir,
                "--output",
                model_dir,
                "--model_version",
                version,
            ],
            check=True,
        )
    except Exception:
        df_antiguo.to_csv(train_antiguo_path, index=False)
        df_antiguo_y.to_csv(train_antiguo_path_y, index=False)
        raise

    ids_procesados = {registro.get("sample_id") for registro in registros_aprobados}
    registros_restantes = [
        registro
        for registro in leer_jsonl(validated_path)
        if registro.get("sample_id") not in ids_procesados
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archivo_procesados = Path(validated_path).with_name(
        f"retraining_validated_processed_{timestamp}.jsonl"
    )
    escribir_jsonl(archivo_procesados, registros_aprobados)
    escribir_jsonl(validated_path, registros_restantes)

    print(f"Re-entrenamiento completado con {len(registros_aprobados)} muestras validadas.")
    print(f"Etiquetas procesadas archivadas en: {archivo_procesados}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
