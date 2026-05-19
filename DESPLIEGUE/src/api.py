import os
import json
import subprocess
import threading
import joblib
import time
import uuid
from collections import deque
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from preprocesamiento import preprocesamiento_inferencia
from inferencia import modelo_deteccion_anomalias
from drift import DriftMonitor
from alerting import AlertManager
import pandas as pd
from typing import List, Optional
from threading import Lock
from datetime import datetime, timezone
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

app = FastAPI(
    title="API de Inferencia para Detección de Anomalías en sistemas IIoT",
    version="1.0",
    description="Esta API recibe datos de tráfico de red, lo preprocesa y devuelve una predicción de si el tráfico es normal o anómalo." \
    "En el caso de ser anómalo, muestra el tipo de ataque detectado."
)

app_root = os.getenv("APP_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
data_dir = os.getenv("DATA_DIR", os.path.join(app_root, "datos", "preprocesados"))
model_dir = os.getenv("MODEL_DIR", os.path.join(app_root, "modelos"))

modelo_path = os.getenv("MODELO_PATH", os.path.join(model_dir, "RF_model.joblib"))
artefactos_path = os.getenv("ARTEFACTOS_PATH", os.path.join(data_dir, "artefactos_inferencia.joblib"))
metadata_path = os.getenv("MODEL_METADATA_PATH", os.path.join(model_dir, "current_model_metadata.json"))
drift_reference_path = os.getenv("DRIFT_REFERENCE_PATH", os.path.join(data_dir, "drift_reference.json"))
modelos_dir = os.getenv("MODEL_DIR", os.path.dirname(modelo_path) or model_dir)
modelo_alias_path = os.path.join(modelos_dir, "RF_model.joblib")
metadata_alias_path = metadata_path
selection_history_path = os.getenv("MODEL_SELECTION_HISTORY_PATH", os.path.join(modelos_dir, "model_selection_history.jsonl"))
alert_history_path = os.getenv("ALERT_HISTORY_PATH", os.path.join(modelos_dir, "alerts_history.jsonl"))
retraining_review_queue_path = os.getenv(
    "RETRAINING_REVIEW_QUEUE_PATH",
    os.path.join(data_dir, "retraining_review_queue.jsonl"),
)
retraining_validated_path = os.getenv(
    "RETRAINING_VALIDATED_PATH",
    os.path.join(data_dir, "retraining_validated.jsonl"),
)
operational_window_size = int(os.getenv("ALERT_OPERATIONAL_WINDOW_SIZE", "50"))
operational_min_window_size = int(os.getenv("ALERT_OPERATIONAL_MIN_WINDOW_SIZE", "10"))
operational_error_rate_threshold = float(os.getenv("ALERT_ERROR_RATE_THRESHOLD", "0.30"))
operational_latency_p95_threshold = float(os.getenv("ALERT_LATENCY_P95_THRESHOLD", "3.0"))
model_f1_macro_min = float(os.getenv("ALERT_MODEL_F1_MACRO_MIN", "0.90"))
modelo_lock = Lock()
drift_alert_previously_active = False
request_window = deque(maxlen=operational_window_size)

CANTIDAD_MAX_LOGS_REENTRENAMIENTO = 1000
CANTIDAD_LOGS_ACTUALES_REENTRENAMIENTO = 0
RETRAINING_IN_PROGRESS = False
ULTIMO_REENTRENAMIENTO_TIMESTAMP = time.time()
REENTRENAMIENTO_INTERVALO = 3600 # Cada hora
lock = threading.Lock()

HTTP_REQUESTS_TOTAL = Counter(
    "api_http_requests_total",
    "Numero total de peticiones HTTP recibidas por la API",
    ["method", "endpoint", "http_status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "api_http_request_duration_seconds",
    "Tiempo de respuesta de las peticiones HTTP en segundos",
    ["method", "endpoint"],
)
PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Numero total de predicciones generadas por la API",
)
PREDICTIONS_BY_CLASS_TOTAL = Counter(
    "api_predictions_by_class_total",
    "Numero total de predicciones generadas por clase",
    ["predicted_class"],
)
ACTIVE_MODEL_INFO = Gauge(
    "api_active_model_info",
    "Informacion del modelo activo. El valor 1 indica el modelo cargado actualmente",
    ["model_version"],
)
MODEL_METRIC = Gauge(
    "api_model_metric",
    "Metricas de validacion del modelo activo",
    ["model_version", "metric"],
)
DRIFT_SCORE = Gauge(
    "api_drift_score",
    "Puntuacion de deriva de datos calculada sobre las peticiones de inferencia",
    ["score_type"],
)
DRIFT_ALERT_ACTIVE = Gauge(
    "api_drift_alert_active",
    "Indica si la alerta de deriva de datos esta activa",
)
DRIFT_WINDOW_OBSERVATIONS = Gauge(
    "api_drift_window_observations",
    "Numero de observaciones disponibles en la ventana de deriva",
)
DRIFT_ALERTS_TOTAL = Counter(
    "api_drift_alerts_total",
    "Numero total de activaciones de alerta de deriva de datos",
)
ALERTS_TOTAL = Counter(
    "api_alerts_total",
    "Numero total de alertas generadas por la API",
    ["category", "severity", "key"],
)
ALERT_ACTIVE = Gauge(
    "api_alert_active",
    "Indica si una alerta esta activa",
    ["category", "severity", "key"],
)
RETRAINING_REVIEW_PENDING = Gauge(
    "api_retraining_review_pending",
    "Numero de muestras pendientes de validacion humana para reentrenamiento",
)
RETRAINING_VALIDATED_SAMPLES = Gauge(
    "api_retraining_validated_samples",
    "Numero de muestras validadas disponibles para reentrenamiento",
)
RETRAINING_MIN_VALIDATED_SAMPLES = Gauge(
    "api_retraining_min_validated_samples",
    "Minimo de muestras validadas requerido para reentrenar",
)
RETRAINING_IN_PROGRESS_GAUGE = Gauge(
    "api_retraining_in_progress",
    "Indica si hay un reentrenamiento en curso",
)
RETRAINING_LAST_STATUS = Gauge(
    "api_retraining_last_status",
    "Estado del ultimo intento de reentrenamiento. El estado activo vale 1",
    ["status"],
)


drift_monitor = DriftMonitor(drift_reference_path)
alert_manager = AlertManager.from_env(alert_history_path)


def registrar_alerta(key, category, severity, title, detail, metadata=None):
    event = alert_manager.trigger(
        key=key,
        category=category,
        severity=severity,
        title=title,
        detail=detail,
        metadata=metadata or {},
    )
    ALERT_ACTIVE.labels(category=category, severity=severity, key=key).set(1)
    if event is not None:
        ALERTS_TOTAL.labels(category=category, severity=severity, key=key).inc()
    return event


def resolver_alerta(key, category, severity):
    alert_manager.resolve(key)
    ALERT_ACTIVE.labels(category=category, severity=severity, key=key).set(0)


def actualizar_metrica_modelo_activo():
    MODEL_METRIC.clear()
    metadata = cargar_metadata_modelo()
    model_version = obtener_version_modelo_activo(metadata)
    versiones_modelos = {
        modelo_disponible["version"]
        for modelo_disponible in listar_modelos_disponibles()
    }
    versiones_modelos.add(model_version)

    for version in versiones_modelos:
        ACTIVE_MODEL_INFO.labels(model_version=version).set(1 if version == model_version else 0)

    metricas = cargar_metricas_modelo_activo()
    metrics = metricas["metrics"]
    classification_report = metrics.get("classification_report", {})
    metricas_resumen = {
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "precision_macro": classification_report.get("macro avg", {}).get("precision"),
        "recall_macro": classification_report.get("macro avg", {}).get("recall"),
    }

    for nombre_metrica, valor in metricas_resumen.items():
        if valor is not None:
            MODEL_METRIC.labels(model_version=model_version, metric=nombre_metrica).set(float(valor))


def actualizar_metricas_deriva(status):
    global drift_alert_previously_active

    if not status.get("enabled"):
        DRIFT_SCORE.labels(score_type="last_sample").set(0)
        DRIFT_SCORE.labels(score_type="rolling_window").set(0)
        DRIFT_ALERT_ACTIVE.set(0)
        DRIFT_WINDOW_OBSERVATIONS.set(0)
        drift_alert_previously_active = False
        return

    DRIFT_SCORE.labels(score_type="last_sample").set(float(status.get("last_sample_score", 0)))
    DRIFT_SCORE.labels(score_type="rolling_window").set(float(status.get("rolling_drift_score", 0)))
    DRIFT_WINDOW_OBSERVATIONS.set(float(status.get("window_observations", 0)))

    alert_active = bool(status.get("alert_active", False))
    DRIFT_ALERT_ACTIVE.set(1 if alert_active else 0)
    if alert_active and not drift_alert_previously_active:
        DRIFT_ALERTS_TOTAL.inc()

    if alert_active:
        registrar_alerta(
            key="model_data_drift",
            category="model",
            severity="warning",
            title="Deriva de datos activa",
            detail="La ventana reciente de peticiones se sale del perfil normal aprendido.",
            metadata={
                "rolling_drift_score": status.get("rolling_drift_score"),
                "last_sample_score": status.get("last_sample_score"),
                "alert_reason": status.get("alert_reason"),
                "top_features": status.get("top_features", []),
            },
        )
    else:
        resolver_alerta("model_data_drift", "model", "warning")

    drift_alert_previously_active = alert_active


def es_endpoint_interno(endpoint):
    return endpoint in {"/metrics", "/health"}

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


def ruta_label_encoder_version(model_version):
    return os.path.join(modelos_dir, f"{model_version}_label_encoder.joblib")


def cargar_json_si_existe(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def obtener_version_modelo_activo(metadata):
    return metadata.get("model_version", os.path.basename(modelo_path).replace(".joblib", ""))


def cargar_metricas_modelo_activo():
    metadata = cargar_metadata_modelo()
    model_version = obtener_version_modelo_activo(metadata)
    metrics_path = ruta_metricas_version(model_version)
    metrics = cargar_json_si_existe(metrics_path) or metadata.get("metrics", {})

    return {
        "model_version": model_version,
        "metrics_file": metrics_path if os.path.isfile(metrics_path) else None,
        "metrics_available": bool(metrics),
        "metrics": metrics,
    }


def percentil(valores, p):
    if not valores:
        return 0
    ordenados = sorted(valores)
    indice = round((len(ordenados) - 1) * p)
    return float(ordenados[indice])


def evaluar_alertas_operativas(endpoint, http_status, duracion_segundos):
    request_window.append(
        {
            "endpoint": endpoint,
            "http_status": int(http_status),
            "duration": float(duracion_segundos),
        }
    )

    if len(request_window) < operational_min_window_size:
        return

    total = len(request_window)
    errores = sum(1 for item in request_window if item["http_status"] >= 400)
    error_rate = errores / total
    p95_latency = percentil([item["duration"] for item in request_window], 0.95)

    if error_rate >= operational_error_rate_threshold:
        registrar_alerta(
            key="operational_http_error_rate",
            category="operational",
            severity="warning",
            title="Tasa elevada de errores HTTP",
            detail=f"La tasa de errores HTTP en la ventana reciente es {error_rate:.2%}.",
            metadata={
                "window_size": total,
                "errors": errores,
                "error_rate": error_rate,
                "threshold": operational_error_rate_threshold,
            },
        )
    else:
        resolver_alerta("operational_http_error_rate", "operational", "warning")

    if p95_latency >= operational_latency_p95_threshold:
        registrar_alerta(
            key="operational_high_latency",
            category="operational",
            severity="warning",
            title="Latencia elevada en la API",
            detail=f"La latencia p95 en la ventana reciente es {p95_latency:.3f}s.",
            metadata={
                "window_size": total,
                "p95_latency_seconds": p95_latency,
                "threshold_seconds": operational_latency_p95_threshold,
            },
        )
    else:
        resolver_alerta("operational_high_latency", "operational", "warning")


def evaluar_alerta_metrica_modelo():
    metricas = cargar_metricas_modelo_activo()
    metrics = metricas["metrics"]
    f1_macro = metrics.get("f1_macro")

    if f1_macro is None:
        return

    if float(f1_macro) < model_f1_macro_min:
        registrar_alerta(
            key="model_f1_macro_low",
            category="model",
            severity="critical",
            title="F1 macro del modelo por debajo del umbral",
            detail=f"El modelo activo tiene f1_macro={float(f1_macro):.4f}.",
            metadata={
                "model_version": metricas["model_version"],
                "f1_macro": float(f1_macro),
                "threshold": model_f1_macro_min,
            },
        )
    else:
        resolver_alerta("model_f1_macro_low", "model", "critical")


def cargar_label_encoder_modelo(metadata_path_activo=None, modelo_path_activo=None):
    metadata_activa_path = metadata_path_activo or metadata_path
    modelo_activo_path = modelo_path_activo or modelo_path
    metadata = cargar_json_si_existe(metadata_activa_path) or {}

    candidatos = []
    model_version = metadata.get("model_version")
    if model_version:
        candidatos.append(ruta_label_encoder_version(model_version))

    version_desde_archivo = os.path.basename(modelo_activo_path).replace(".joblib", "")
    if version_desde_archivo != "RF_model":
        candidatos.append(ruta_label_encoder_version(version_desde_archivo))

    for candidato in candidatos:
        if os.path.isfile(candidato):
            return joblib.load(candidato)

    return None


def decodificar_predicciones(predicciones, label_encoder_activo):
    if label_encoder_activo is not None:
        return label_encoder_activo.inverse_transform(predicciones).tolist()

    metadata = cargar_metadata_modelo()
    clases = metadata.get("classes", [])
    return [
        clases[int(prediccion)] if int(prediccion) < len(clases) else str(prediccion)
        for prediccion in predicciones
    ]


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


def escribir_jsonl(path, registro):
    directorio = os.path.dirname(path)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def leer_jsonl(path, limit=None):
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        lineas = [line.strip() for line in f if line.strip()]

    if limit is not None:
        lineas = lineas[-limit:]

    return [json.loads(linea) for linea in lineas]


def buscar_muestra_revision(sample_id):
    for registro in leer_jsonl(retraining_review_queue_path):
        if registro.get("sample_id") == sample_id:
            return registro

    return None


def ids_muestras_validadas():
    return {
        registro.get("sample_id")
        for registro in leer_jsonl(retraining_validated_path)
        if registro.get("sample_id")
    }


def listar_muestras_pendientes(limit=50):
    validadas = ids_muestras_validadas()
    pendientes = [
        registro
        for registro in leer_jsonl(retraining_review_queue_path)
        if registro.get("sample_id") not in validadas
    ]
    return list(reversed(pendientes[-limit:]))


def contar_muestras_pendientes_revision():
    validadas = ids_muestras_validadas()
    return sum(
        1
        for registro in leer_jsonl(retraining_review_queue_path)
        if registro.get("sample_id") not in validadas
    )


def contar_muestras_validadas_retraining():
    return sum(
        1
        for registro in leer_jsonl(retraining_validated_path)
        if registro.get("review_status") == "approved"
        and registro.get("validated_label")
        and isinstance(registro.get("input"), dict)
    )


def actualizar_metricas_retraining():
    RETRAINING_REVIEW_PENDING.set(contar_muestras_pendientes_revision())
    RETRAINING_VALIDATED_SAMPLES.set(contar_muestras_validadas_retraining())
    RETRAINING_MIN_VALIDATED_SAMPLES.set(float(os.getenv("RETRAINING_MIN_VALIDATED_SAMPLES", "50")))
    RETRAINING_IN_PROGRESS_GAUGE.set(1 if RETRAINING_IN_PROGRESS else 0)


def actualizar_ultimo_estado_retraining(status):
    for estado in ["success", "skipped", "failed"]:
        RETRAINING_LAST_STATUS.labels(status=estado).set(1 if status == estado else 0)


def recargar_modelo_reentrenado(previous_model=None):
    global modelo, modelo_path, metadata_path, label_encoder

    metadata_reentrenada = cargar_json_si_existe(metadata_alias_path) or {}
    version_reentrenada = metadata_reentrenada.get(
        "model_version",
        os.path.basename(modelo_alias_path).replace(".joblib", ""),
    )
    modelo_reentrenado = modelo_deteccion_anomalias(modelo_path=modelo_alias_path)
    label_encoder_reentrenado = cargar_label_encoder_modelo(
        metadata_path_activo=metadata_alias_path,
        modelo_path_activo=modelo_alias_path,
    )

    with modelo_lock:
        if previous_model is None:
            metadata_anterior = cargar_metadata_modelo()
            previous_model = obtener_version_modelo_activo(metadata_anterior)
        modelo = modelo_reentrenado
        modelo_path = modelo_alias_path
        metadata_path = metadata_alias_path
        label_encoder = label_encoder_reentrenado
        actualizar_metrica_modelo_activo()
        evaluar_alerta_metrica_modelo()

    registrar_cambio_modelo(
        previous_model=previous_model,
        new_model=version_reentrenada,
        status="success",
        detail="Modelo reentrenado cargado automaticamente en memoria",
    )
    print(f"Modelo reentrenado cargado en memoria: {version_reentrenada}")


if not os.path.isfile(artefactos_path):
    raise FileNotFoundError(f"El archivo de artefactos no existe: {artefactos_path}")

modelo = modelo_deteccion_anomalias(modelo_path=modelo_path)
label_encoder = cargar_label_encoder_modelo()
actualizar_metrica_modelo_activo()
evaluar_alerta_metrica_modelo()
actualizar_metricas_deriva(drift_monitor.status())
actualizar_metricas_retraining()
actualizar_ultimo_estado_retraining(None)


class SeleccionModelo(BaseModel):
    model_version: str = Field(
        ...,
        description="Version del modelo que se quiere activar",
        json_schema_extra={"example": "RF_v1"}
    )


class AlertaPrueba(BaseModel):
    category: str = Field("operational", json_schema_extra={"example": "operational"})
    severity: str = Field("warning", json_schema_extra={"example": "warning"})
    title: str = Field("Alerta de prueba", json_schema_extra={"example": "Alerta de prueba"})
    detail: str = Field(
        "Validacion manual del canal de alertas",
        json_schema_extra={"example": "Validacion manual del canal de alertas"}
    )


class EtiquetaValidada(BaseModel):
    sample_id: str = Field(..., json_schema_extra={"example": "2026-05-19T15:30:10.123456+00:00_a1b2c3d4"})
    validated_label: str = Field("normal", json_schema_extra={"example": "normal"})
    reviewer: Optional[str] = Field(None, json_schema_extra={"example": "rafa"})
    notes: Optional[str] = Field(None, json_schema_extra={"example": "Validado manualmente"})


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


@app.middleware("http")
async def registrar_metricas_http(request: Request, call_next):
    inicio = time.perf_counter()
    endpoint = request.url.path
    http_status = 500
    response = None

    try:
        response = await call_next(request)
        http_status = response.status_code
    finally:
        route = request.scope.get("route")
        if route is not None and getattr(route, "path", None):
            endpoint = route.path

        if not es_endpoint_interno(endpoint):
            duracion = time.perf_counter() - inicio
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                http_status=str(http_status),
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duracion)
            evaluar_alertas_operativas(endpoint, http_status, duracion)

    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": modelo.modelo is not None,
        "label_encoder_loaded": label_encoder is not None,
        "drift_monitor_enabled": drift_monitor.status().get("enabled", False),
        "model_path": modelo_path,
        "artefactos_path": artefactos_path,
        "drift_reference_path": drift_reference_path,
        "alert_channels": alert_manager.channels(),
    }


@app.get("/drift/status")
def drift_status():
    return drift_monitor.status()


@app.post("/admin/drift/reset")
def reset_drift_monitor():
    status = drift_monitor.reset()
    actualizar_metricas_deriva(status)
    return status


@app.get("/alerts/status")
def alerts_status(limit: int = 20):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit debe estar entre 1 y 200")

    return alert_manager.status(limit=limit)


@app.post("/admin/alerts/reset")
def reset_alerts_runtime():
    alert_manager.reset_runtime()
    for key, category, severity in [
        ("operational_http_error_rate", "operational", "warning"),
        ("operational_high_latency", "operational", "warning"),
        ("model_f1_macro_low", "model", "critical"),
        ("model_data_drift", "model", "warning"),
    ]:
        ALERT_ACTIVE.labels(category=category, severity=severity, key=key).set(0)

    return alert_manager.status()


@app.post("/admin/alerts/test")
def test_alert(alerta: AlertaPrueba):
    event = registrar_alerta(
        key="manual_test_alert",
        category=alerta.category,
        severity=alerta.severity,
        title=alerta.title,
        detail=alerta.detail,
        metadata={"source": "admin_test_endpoint"},
    )
    return {
        "status": "sent" if event is not None else "cooldown_active",
        "event": event,
        "alerts": alert_manager.status(limit=5),
    }


@app.get("/model-info")
def model_info():
    metadata = cargar_metadata_modelo()
    metadata["model_path"] = modelo_path
    metadata["artefactos_path"] = artefactos_path
    return metadata


@app.get("/model/metrics")
def model_metrics():
    metricas = cargar_metricas_modelo_activo()
    metrics = metricas["metrics"]
    classification_report = metrics.get("classification_report", {})

    return {
        "model_version": metricas["model_version"],
        "metrics_file": metricas["metrics_file"],
        "metrics_available": metricas["metrics_available"],
        "summary": {
            "accuracy": metrics.get("accuracy"),
            "precision_macro": classification_report.get("macro avg", {}).get("precision"),
            "recall_macro": classification_report.get("macro avg", {}).get("recall"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "auc": metrics.get("auc"),
        },
        "classification_report": classification_report,
    }


@app.get("/metrics")
def prometheus_metrics():
    actualizar_metricas_retraining()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


@app.get("/admin/retraining/review-queue")
def retraining_review_queue(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit debe estar entre 1 y 500")

    pendientes = listar_muestras_pendientes(limit=limit)
    return {
        "review_queue_file": retraining_review_queue_path,
        "validated_file": retraining_validated_path,
        "pending_count_returned": len(pendientes),
        "pending_samples": pendientes,
    }


@app.post("/admin/retraining/labels")
def validate_retraining_label(etiqueta: EtiquetaValidada):
    muestra = buscar_muestra_revision(etiqueta.sample_id)
    if muestra is None:
        raise HTTPException(status_code=404, detail=f"No existe sample_id pendiente: {etiqueta.sample_id}")

    if etiqueta.sample_id in ids_muestras_validadas():
        raise HTTPException(status_code=409, detail=f"sample_id ya validado: {etiqueta.sample_id}")

    validated_label = etiqueta.validated_label.strip().lower()
    clases_validas = cargar_metadata_modelo().get("classes", [])
    if clases_validas and validated_label not in clases_validas:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "validated_label no existe entre las clases conocidas del modelo",
                "allowed_labels": clases_validas,
            },
        )

    registro_validado = {
        **muestra,
        "review_status": "approved",
        "validated_label": validated_label,
        "label_source": "human",
        "reviewer": etiqueta.reviewer,
        "review_notes": etiqueta.notes,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    escribir_jsonl(retraining_validated_path, registro_validado)

    return {
        "status": "validated",
        "sample_id": etiqueta.sample_id,
        "validated_label": validated_label,
        "validated_file": retraining_validated_path,
    }


@app.post("/admin/models/select")
def select_model(selection: SeleccionModelo):
    global modelo, modelo_path, metadata_path, label_encoder

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
        nuevo_label_encoder = cargar_label_encoder_modelo(
            metadata_path_activo=nuevo_metadata_path,
            modelo_path_activo=nuevo_modelo_path,
        )
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
        label_encoder = nuevo_label_encoder
        actualizar_metrica_modelo_activo()
        evaluar_alerta_metrica_modelo()

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

def retraining():

    global RETRAINING_IN_PROGRESS

    def run():

        global RETRAINING_IN_PROGRESS
        global CANTIDAD_LOGS_ACTUALES_REENTRENAMIENTO
        global ULTIMO_REENTRENAMIENTO_TIMESTAMP

        previous_model = None
        try:
            previous_model = obtener_version_modelo_activo(cargar_metadata_modelo())
            resultado = subprocess.run(["python", "retraining.py"])
            if resultado.returncode == 0:
                recargar_modelo_reentrenado(previous_model=previous_model)
                actualizar_ultimo_estado_retraining("success")
            elif resultado.returncode == 2:
                registrar_cambio_modelo(
                    previous_model=previous_model,
                    new_model=None,
                    status="skipped",
                    detail="Reentrenamiento omitido: no hay suficientes muestras validadas",
                )
                actualizar_ultimo_estado_retraining("skipped")
            else:
                registrar_cambio_modelo(
                    previous_model=previous_model,
                    new_model=None,
                    status="failed",
                    detail=f"Reentrenamiento fallido con codigo de salida {resultado.returncode}",
                )
                actualizar_ultimo_estado_retraining("failed")
        except Exception as e:
            registrar_cambio_modelo(
                previous_model=previous_model or obtener_version_modelo_activo(cargar_metadata_modelo()),
                new_model=None,
                status="failed",
                detail=f"No se pudo recargar el modelo reentrenado: {e}",
            )
            actualizar_ultimo_estado_retraining("failed")
            print(f"No se pudo recargar el modelo reentrenado: {e}")
        finally:
            RETRAINING_IN_PROGRESS = False
            CANTIDAD_LOGS_ACTUALES_REENTRENAMIENTO = 0
            ULTIMO_REENTRENAMIENTO_TIMESTAMP = time.time()

    with lock:
        if RETRAINING_IN_PROGRESS:
            print("Re-entrenamiento ya en progreso. Se ignora esta solicitud.")
            return
    
        RETRAINING_IN_PROGRESS = True
            
    threading.Thread(target=run).start()

@app.post("/predict")
def predict(input_data: List[InputParaElModelo]):

    global CANTIDAD_LOGS_ACTUALES_REENTRENAMIENTO
    global CANTIDAD_MAX_LOGS_REENTRENAMIENTO
    global RETRAINING_IN_PROGRESS
    global ULTIMO_REENTRENAMIENTO_TIMESTAMP
    global REENTRENAMIENTO_INTERVALO

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
        drift_status_actual = drift_monitor.observe(X_preprocesado)
        actualizar_metricas_deriva(drift_status_actual)

        # Realizar la predicción
        with modelo_lock:
            modelo_activo = modelo
            label_encoder_activo = label_encoder

        prediccion = modelo_activo.predecir(X_preprocesado)
        prediccion_decodificada = decodificar_predicciones(prediccion, label_encoder_activo)
        PREDICTIONS_TOTAL.inc(len(prediccion_decodificada))
        for clase_predicha in prediccion_decodificada:
            PREDICTIONS_BY_CLASS_TOTAL.labels(predicted_class=str(clase_predicha)).inc()
        print(prediccion)

        metadata_activa = cargar_metadata_modelo()
        model_version_activa = obtener_version_modelo_activo(metadata_activa)
        timestamp_prediccion = datetime.now(timezone.utc).isoformat()
        registros_preprocesados = X_preprocesado.to_dict(orient="records")
        muestras_revision = []
        sample_score_threshold = float(drift_status_actual.get("sample_alert_threshold", 0))
        last_sample_score = float(drift_status_actual.get("last_sample_score", 0))
        enviar_a_revision = bool(drift_status_actual.get("alert_active", False)) or (
            sample_score_threshold > 0 and last_sample_score >= sample_score_threshold
        )

        if enviar_a_revision:
            for index, registro_preprocesado in enumerate(registros_preprocesados):
                sample_id = f"{timestamp_prediccion}_{uuid.uuid4().hex[:8]}"
                muestra_revision = {
                    "sample_id": sample_id,
                    "timestamp": timestamp_prediccion,
                    "model_version": model_version_activa,
                    "input": registro_preprocesado,
                    "prediction": prediccion_decodificada[index],
                    "prediction_code": int(prediccion[index]),
                    "drift_status": drift_status_actual,
                    "review_reason": drift_status_actual.get("alert_reason") or "sample_out_of_profile",
                    "review_status": "pending",
                }
                escribir_jsonl(retraining_review_queue_path, muestra_revision)
                muestras_revision.append({
                    "sample_id": sample_id,
                    "predicted_label": prediccion_decodificada[index],
                    "review_reason": muestra_revision["review_reason"],
                    "review_status": "pending",
                })

        # Guardamos la instancia para re-entrenamiento futuro en caso de deriva, junto con su predicción y el resultado de la monitorización de deriva
        predicciones_log_path = os.getenv("PREDICTIONS_LOG_PATH", os.path.join(data_dir, "predicciones_inferencia.log"))

        registro_drift = {
            "timestamp": timestamp_prediccion,
            "model_version": model_version_activa,
            "input": registros_preprocesados,
            "prediction": prediccion_decodificada,
            "queued_for_review": enviar_a_revision,
            "review_samples": muestras_revision,
            "drift_status": drift_status_actual,
        }
        escribir_jsonl(predicciones_log_path, registro_drift)
        
        with lock:
            CANTIDAD_LOGS_ACTUALES_REENTRENAMIENTO += 1
        
        # Re-entrenamos el modelo según ciertas restricciones
        if RETRAINING_IN_PROGRESS:
            print("Re-entrenamiento ya en progreso. Se evaluará la necesidad de iniciar otro re-entrenamiento al finalizar el actual.")
            pass
        elif CANTIDAD_LOGS_ACTUALES_REENTRENAMIENTO >= CANTIDAD_MAX_LOGS_REENTRENAMIENTO:
            print("Iniciando proceso de re-entrenamiento por cantidad de logs alcanzada...")
            retraining()

        # Si la deriva está activa, damos prioridad al re-entrenamiento para intentar mitigarla lo antes posible
        elif drift_status_actual.get("alert_active", False):
            print("Iniciando proceso de re-entrenamiento por alerta de deriva activa...")
            retraining()

        # Si ha pasado un tiempo considerable desde el último re-entrenamiento se inicia uno nuevo
        elif time.time() - ULTIMO_REENTRENAMIENTO_TIMESTAMP >= REENTRENAMIENTO_INTERVALO:
            print("Iniciando proceso de re-entrenamiento por intervalo de tiempo alcanzado...")
            retraining()

        
        return {
            "prediccion": prediccion_decodificada,
            "prediccion_codificada": prediccion.tolist(),
            "drift": {
                "alert_active": drift_status_actual.get("alert_active", False),
                "last_sample_score": drift_status_actual.get("last_sample_score", 0),
                "rolling_drift_score": drift_status_actual.get("rolling_drift_score", 0),
                "alert_reason": drift_status_actual.get("alert_reason"),
            },
            "queued_for_review": enviar_a_revision,
            "review_samples": muestras_revision,
        }

    except Exception as e:
        columnas = df_input.columns if "df_input" in locals() else []
        print(columnas)
        raise HTTPException(status_code=500, detail=str(columnas) + " - " + str(e))

