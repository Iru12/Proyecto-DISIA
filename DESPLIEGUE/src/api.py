import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from preprocesamiento import preprocesamiento_inferencia
from inferencia import modelo_deteccion_anomalias
import pandas as pd
from typing import List
from threading import Lock
from datetime import datetime, timezone

app = FastAPI(
    title="API de Inferencia para Detección de Anomalías en sistemas IIoT",
    version="1.0",
    description="Esta API recibe datos de tráfico de red, lo preprocesa y devuelve una predicción de si el tráfico es normal o anómalo." \
    "En el caso de ser anómalo, muestra el tipo de ataque detectado."
)

modelo_path = os.getenv("MODELO_PATH", "modelos/RF_model.joblib")
artefactos_path = os.getenv("ARTEFACTOS_PATH", "artefactos/preprocesamiento_artifacts.joblib")
metadata_path = os.getenv("MODEL_METADATA_PATH", "/app/modelos/current_model_metadata.json")
modelos_dir = os.path.dirname(modelo_path) or "."
selection_history_path = os.getenv("MODEL_SELECTION_HISTORY_PATH", os.path.join(modelos_dir, "model_selection_history.jsonl"))
modelo_lock = Lock()

if not os.path.isfile(artefactos_path):
    raise FileNotFoundError(f"El archivo de artefactos no existe: {artefactos_path}")

modelo = modelo_deteccion_anomalias(modelo_path=modelo_path)


def cargar_metadata_modelo():
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "model_file": os.path.basename(modelo_path),
        "model_path": modelo_path,
        "metadata_available": False,
    }


def ruta_modelo_version(model_version):
    nombre_archivo = f"{model_version}.joblib"
    return os.path.join(modelos_dir, nombre_archivo)


def ruta_metadata_version(model_version):
    return os.path.join(modelos_dir, f"{model_version}_metadata.json")


def ruta_metricas_version(model_version):
    return os.path.join(modelos_dir, f"{model_version}_metrics.json")


def cargar_json_si_existe(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def listar_modelos_disponibles():
    modelos = []

    if not os.path.isdir(modelos_dir):
        return modelos

    for archivo in sorted(os.listdir(modelos_dir)):
        if not archivo.endswith(".joblib"):
            continue
        if archivo == "RF_model.joblib" or archivo.endswith("_label_encoder.joblib"):
            continue

        version = archivo[:-7]
        metadata_file = f"{version}_metadata.json"
        metrics_file = f"{version}_metrics.json"
        metadata = cargar_json_si_existe(os.path.join(modelos_dir, metadata_file)) or {}

        modelos.append({
            "version": version,
            "model_file": archivo,
            "metadata_file": metadata_file if os.path.isfile(os.path.join(modelos_dir, metadata_file)) else None,
            "metrics_file": metrics_file if os.path.isfile(os.path.join(modelos_dir, metrics_file)) else None,
            "created_at": metadata.get("created_at"),
        })

    return modelos


def registrar_cambio_modelo(previous_model, new_model, status, detail=None):
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_model": previous_model,
        "new_model": new_model,
        "status": status,
        "detail": detail,
    }

    os.makedirs(os.path.dirname(selection_history_path), exist_ok=True)
    with open(selection_history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro) + "\n")

    return registro


def leer_historial_cambios(limit=50):
    if not os.path.isfile(selection_history_path):
        return []

    with open(selection_history_path, "r", encoding="utf-8") as f:
        lineas = [line.strip() for line in f if line.strip()]

    registros = [json.loads(linea) for linea in lineas[-limit:]]
    return list(reversed(registros))


class SeleccionModelo(BaseModel):
    model_version: str = Field(
        ...,
        description="Version del modelo que se quiere activar",
        json_schema_extra={"example": "RF_v1"}
    )


class InputParaElModelo(BaseModel):
    model_config = {
        "populate_by_name": True
    }

    Date: str = Field(
        ...,
        description="Fecha y hora del tráfico de red",
        json_schema_extra={"example": "9/01/2020"}
    )

    Timestamp: int = Field(
        ...,
        description="Timestamp Unix del evento",
        json_schema_extra={"example": 1578522486}
    )

    Scr_IP: str = Field(
        ...,
        description="IP origen",
        json_schema_extra={"example": "172.24.1.80"}
    )

    Scr_port: int = Field(
        ...,
        description="Puerto origen",
        json_schema_extra={"example": 59050}
    )

    Des_IP: str = Field(
        ...,
        description="IP destino",
        json_schema_extra={"example": "172.24.1.1"}
    )

    Des_port: int = Field(
        ...,
        description="Puerto destino",
        json_schema_extra={"example": 53}
    )

    Protocol: str = Field(
        ...,
        description="Protocolo de red",
        json_schema_extra={"example": "udp"}
    )

    Service: str = Field(
        ...,
        description="Servicio detectado",
        json_schema_extra={"example": "dns"}
    )

    Duration: float = Field(
        ...,
        description="Duración de la conexión",
        json_schema_extra={"example": 0.000132}
    )

    Scr_bytes: int = Field(
        ...,
        description="Bytes enviados por el origen",
        json_schema_extra={"example": 38}
    )

    Des_bytes: int = Field(
        ...,
        description="Bytes enviados por el destino",
        json_schema_extra={"example": 38}
    )

    Conn_state: int = Field(
        ...,
        description="Estado de la conexión",
        json_schema_extra={"example": 1}
    )

    missed_bytes: int = Field(
        ...,
        description="Bytes perdidos",
        json_schema_extra={"example": 0}
    )

    is_syn_only: bool = Field(
        ...,
        description="Solo SYN",
        json_schema_extra={"example": False}
    )

    Is_SYN_ACK: bool = Field(
        ...,
        description="SYN ACK detectado",
        json_schema_extra={"example": False}
    )

    is_pure_ack: bool = Field(
        ...,
        description="ACK puro",
        json_schema_extra={"example": False}
    )

    is_with_payload: bool = Field(
        ...,
        description="Tiene payload",
        json_schema_extra={"example": True}
    )

    FIN_or_RST: bool = Field(
        ...,
        description="FIN o RST en la conexión",
        json_schema_extra={"example": False}
    )

    Bad_checksum: bool = Field(
        ...,
        description="Checksum incorrecto",
        json_schema_extra={"example": False}
    )

    is_SYN_with_RST: bool = Field(
        ...,
        description="SYN con RST",
        json_schema_extra={"example": False}
    )

    Scr_pkts: int = Field(..., json_schema_extra={"example": 1})
    Scr_ip_bytes: int = Field(..., json_schema_extra={"example": 66})
    Des_pkts: int = Field(..., json_schema_extra={"example": 1})
    Des_ip_bytes: int = Field(..., json_schema_extra={"example": 66})

    anomaly_alert: bool = Field(..., json_schema_extra={"example": False})

    total_bytes: int = Field(..., json_schema_extra={"example": 208})
    total_packet: int = Field(..., json_schema_extra={"example": 2})

    paket_rate: float = Field(..., json_schema_extra={"example": 15151.51515})
    byte_rate: float = Field(..., json_schema_extra={"example": 1575757576})

    Scr_packts_ratio: float = Field(..., json_schema_extra={"example": 0.5})
    Des_pkts_ratio: float = Field(..., json_schema_extra={"example": 0.5})
    Scr_bytes_ratio: float = Field(..., json_schema_extra={"example": 0.5})
    Des_bytes_ratio: float = Field(..., json_schema_extra={"example": 0.5})

    Avg_user_time: float = Field(..., json_schema_extra={"example": 6931})
    Std_user_time: float = Field(..., json_schema_extra={"example": 6.416007248})

    Avg_nice_time: float = Field(..., json_schema_extra={"example": 706})
    Std_nice_time: float = Field(..., json_schema_extra={"example": 0.408905857})

    Avg_system_time: float = Field(..., json_schema_extra={"example": 1693})
    Std_system_time: float = Field(..., json_schema_extra={"example": 0.771635277})

    Avg_iowait_time: float = Field(..., json_schema_extra={"example": 2423})
    Std_iowait_time: float = Field(..., json_schema_extra={"example": 3.829809525})

    Avg_ideal_time: float = Field(..., json_schema_extra={"example": 88245})
    Std_ideal_time: float = Field(..., json_schema_extra={"example": 7.112108337})

    Avg_tps: float = Field(..., json_schema_extra={"example": 37.4})
    Std_tps: float = Field(..., json_schema_extra={"example": 40.19004852})

    Avg_rtps: float = Field(..., json_schema_extra={"example": 30.1})
    Std_rtps: float = Field(..., json_schema_extra={"example": 39.79811553})

    Avg_wtps: float = Field(..., json_schema_extra={"example": 7.3})
    Std_wtps: float = Field(..., json_schema_extra={"example": 3.1})

    Avg_ldavg_1: float = Field(..., json_schema_extra={"example": 0.55})
    Std_ldavg_1: float = Field(..., json_schema_extra={"example": 0.02})

    Avg_kbmemused: float = Field(..., json_schema_extra={"example": 921020.4})
    Std_kbmemused: float = Field(..., json_schema_extra={"example": 2139.652645})

    Avg_num_Proc_s: float = Field(..., json_schema_extra={"example": 1})
    Std_num_proc_s: float = Field(..., json_schema_extra={"example": 0})

    Avg_num_cswch_s: float = Field(..., json_schema_extra={"example": 1603.3})
    std_num_cswch_s: float = Field(..., json_schema_extra={"example": 294.1390997})

    OSSEC_alert: int = Field(..., json_schema_extra={"example": 0})
    OSSEC_alert_level: int = Field(..., json_schema_extra={"example": 0})

    Login_attempt: int = Field(..., json_schema_extra={"example": 0})
    Succesful_login: int = Field(..., json_schema_extra={"example": 0})

    File_activity: int = Field(..., json_schema_extra={"example": 0})
    Process_activity: int = Field(..., json_schema_extra={"example": 0})

    read_write_physical_process: int = Field(..., json_schema_extra={"example": 0})
    is_privileged: int = Field(..., json_schema_extra={"example": 0})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": modelo.modelo is not None,
        "model_path": modelo_path,
        "artefactos_path": artefactos_path,
    }


@app.get("/model-info")
def model_info():
    metadata = cargar_metadata_modelo()
    metadata["model_path"] = modelo_path
    metadata["artefactos_path"] = artefactos_path
    return metadata


@app.get("/models")
def models():
    metadata = cargar_metadata_modelo()
    return {
        "active_model": metadata.get("model_version", os.path.basename(modelo_path).replace(".joblib", "")),
        "active_model_path": modelo_path,
        "models_dir": modelos_dir,
        "available_models": listar_modelos_disponibles(),
    }


@app.get("/models/compare")
def compare_models():
    comparacion = []

    for modelo_disponible in listar_modelos_disponibles():
        metadata = cargar_json_si_existe(os.path.join(modelos_dir, modelo_disponible["metadata_file"])) if modelo_disponible["metadata_file"] else {}
        metrics = (metadata or {}).get("metrics", {})
        params = (metadata or {}).get("params", {})

        comparacion.append({
            "version": modelo_disponible["version"],
            "created_at": (metadata or {}).get("created_at"),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "params": params,
        })

    return comparacion


@app.get("/admin/models/history")
def model_selection_history(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit debe estar entre 1 y 500")

    return {
        "history_file": selection_history_path,
        "changes": leer_historial_cambios(limit=limit),
    }


@app.post("/admin/models/select")
def select_model(selection: SeleccionModelo):
    global modelo, modelo_path, metadata_path

    nuevo_modelo_path = ruta_modelo_version(selection.model_version)
    nuevo_metadata_path = ruta_metadata_version(selection.model_version)
    previous_metadata = cargar_metadata_modelo()
    previous_model = previous_metadata.get("model_version", os.path.basename(modelo_path).replace(".joblib", ""))

    if not os.path.isfile(nuevo_modelo_path):
        registrar_cambio_modelo(
            previous_model=previous_model,
            new_model=selection.model_version,
            status="failed",
            detail=f"No existe el modelo: {nuevo_modelo_path}",
        )
        raise HTTPException(status_code=404, detail=f"No existe el modelo: {nuevo_modelo_path}")

    try:
        nuevo_modelo = modelo_deteccion_anomalias(modelo_path=nuevo_modelo_path)
    except Exception as e:
        registrar_cambio_modelo(
            previous_model=previous_model,
            new_model=selection.model_version,
            status="failed",
            detail=str(e),
        )
        raise HTTPException(status_code=400, detail=f"No se pudo cargar el modelo solicitado: {e}")

    with modelo_lock:
        modelo = nuevo_modelo
        modelo_path = nuevo_modelo_path
        metadata_path = nuevo_metadata_path

    metadata = cargar_metadata_modelo()
    registro = registrar_cambio_modelo(
        previous_model=previous_model,
        new_model=selection.model_version,
        status="success",
        detail=f"Modelo activo actualizado a {selection.model_version}",
    )

    return {
        "status": "model_selected",
        "active_model": selection.model_version,
        "model_path": modelo_path,
        "metadata": metadata,
        "audit": registro,
        "note": "Endpoint admin de simulacion. En produccion debe protegerse con autenticacion/autorizacion.",
    }


@app.post("/predict")
def predict(input_data: List[InputParaElModelo]):
    try:
        # Convertir el modelo de entrada a un DataFrame
        df_input = pd.DataFrame([input_data.model_dump() for input_data in input_data])

        df_input.rename(columns={"Avg_num_cswch_s": "Avg_num_cswch/s",
                                 "read_write_physical_process": "read_write_physical.process",
                                 "std_num_cswch_s": "std_num_cswch/s"
                                }, 
                                inplace=True
                        )

        # Preprocesar los datos de entrada
        X_preprocesado= preprocesamiento_inferencia(df_input, artefactos_path=artefactos_path)

        # Realizar la predicción
        with modelo_lock:
            modelo_activo = modelo

        prediccion = modelo_activo.predecir(X_preprocesado)
        print(prediccion)

        return {"prediccion": prediccion.tolist() }

    except Exception as e:
        columnas = df_input.columns if "df_input" in locals() else []
        print(columnas)
        raise HTTPException(status_code=500, detail=str(columnas) + " - " + str(e))
