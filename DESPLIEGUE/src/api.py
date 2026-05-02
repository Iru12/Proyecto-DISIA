import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from preprocesamiento import preprocesamiento_inferencia
from inferencia import modelo_deteccion_anomalias
import pandas as pd
from typing import List

app = FastAPI(
    title="API de Inferencia para Detección de Anomalías en sistemas IIoT",
    version="1.0",
    description="Esta API recibe datos de tráfico de red, lo preprocesa y devuelve una predicción de si el tráfico es normal o anómalo." \
    "En el caso de ser anómalo, muestra el tipo de ataque detectado."
)

modelo = modelo_deteccion_anomalias(modelo_path=os.getenv("MODELO_PATH", "modelos/RF_model.joblib"))

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
        X_preprocesado= preprocesamiento_inferencia(df_input, artefactos_path=os.getenv("ARTEFACTOS_PATH", "artefactos/preprocesamiento_artifacts.joblib"))

        # Realizar la predicción
        prediccion = modelo.predecir(X_preprocesado)
        print(prediccion)

        return {"prediccion": prediccion.tolist() }

    except Exception as e:
        print(df_input.columns)
        raise HTTPException(status_code=500, detail=str(df_input.columns) + " - " + str(e))
