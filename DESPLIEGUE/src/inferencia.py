import joblib
import numpy as np
import pandas as pd
import os
import argparse
from preprocesamiento import preprocesamiento_inferencia

class modelo_deteccion_anomalias:
    def __init__(self, modelo_path):
        
        # Comprobación de la existencia del archivo del modelo
        if not os.path.isfile(modelo_path):
            raise FileNotFoundError(f"El archivo del modelo no existe: {modelo_path}")
        
        # Cargar el modelo entrenado
        self.modelo = joblib.load(modelo_path)

        print("Modelo de detección de anomalías cargado exitosamente.")

    def predecir(self, X):
        if self.modelo is None:
            print("Error: No se ha cargado un modelo válido.")
            return None
        
        X_np = X.to_numpy(dtype=np.float32)
        return self.modelo.predict(X_np)


if __name__ == "__main__":
    # Ejemplo de datos de prueba para la inferencia
    """
    dato_prueba = [
    {
        "Date": "9/01/2020",
        "Timestamp": 1578522486,
        "Scr_IP": "172.24.1.80",
        "Scr_port": 59050,
        "Des_IP": "172.24.1.1",
        "Des_port": 53,
        "Protocol": "udp",
        "Service": "dns",
        "Duration": 0.000132,
        "Scr_bytes": 38,
        "Des_bytes": 38,
        "Conn_state": 1,
        "missed_bytes": 0,
        "is_syn_only": False,
        "Is_SYN_ACK": False,
        "is_pure_ack": False,
        "is_with_payload": True,
        "FIN or RST": False,
        "Bad_checksum": False,
        "is_SYN_with_RST": False,
        "Scr_pkts": 1,
        "Scr_ip_bytes": 66,
        "Des_pkts": 1,
        "Des_ip_bytes": 66,
        "anomaly_alert": False,
        "total_bytes": 208,
        "total_packet": 2,
        "paket_rate": 15151.51515,
        "byte_rate": 1575757576,
        "Scr_packts_ratio": 0.5,
        "Des_pkts_ratio": 0.5,
        "Scr_bytes_ratio": 0.5,
        "Des_bytes_ratio": 0.5,
        "Avg_user_time": 6931,
        "Std_user_time": 6.416007248,
        "Avg_nice_time": 706,
        "Std_nice_time": 0.408905857,
        "Avg_system_time": 1693,
        "Std_system_time": 0.771635277,
        "Avg_iowait_time": 2423,
        "Std_iowait_time": 3.829809525,
        "Avg_ideal_time": 88245,
        "Std_ideal_time": 7.112108337,
        "Avg_tps": 37.4,
        "Std_tps": 40.19004852,
        "Avg_rtps": 30.1,
        "Std_rtps": 39.79811553,
        "Avg_wtps": 7.3,
        "Std_wtps": 3.1,
        "Avg_ldavg_1": 0.55,
        "Std_ldavg_1": 0.02,
        "Avg_kbmemused": 921020.4,
        "Std_kbmemused": 2139.652645,
        "Avg_num_Proc/s": 1,
        "Std_num_proc/s": 0,
        "Avg_num_cswch/s": 1603.3,
        "std_num_cswch/s": 294.1390997,
        "OSSEC_alert": 0,
        "OSSEC_alert_level": 0,
        "Login_attempt": 0,
        "Succesful_login": 0,
        "File_activity": 0,
        "Process_activity": 0,
        "read_write_physical.process": 0,
        "is_privileged": 0,
        "class1": "Normal",
        "class2": "Normal",
        "class3": "Normal"
    }
    ]
    """
    dato_prueba = [
    {
        "Date": "9/01/2020",
        "Timestamp": 1578522486,
        "Scr_IP": "172.24.1.80",
        "Scr_port": 59050,
        "Des_IP": "172.24.1.1",
        "Des_port": 53,
        "Protocol": "udp",
        "Service": "dns",
        "Duration": 0.000132,
        "Scr_bytes": 38,
        "Des_bytes": 38,
        "Conn_state": 1,
        "missed_bytes": 0,
        "is_syn_only": False,
        "Is_SYN_ACK": False,
        "is_pure_ack": False,
        "is_with_payload": True,
        "FIN or RST": False,
        "Bad_checksum": False,
        "is_SYN_with_RST": False,
        "Scr_pkts": 1,
        "Scr_ip_bytes": 66,
        "Des_pkts": 1,
        "Des_ip_bytes": 66,
        "anomaly_alert": False,
        "total_bytes": 208,
        "total_packet": 2,
        "paket_rate": 15151.51515,
        "byte_rate": 1575757576,
        "Scr_packts_ratio": 0.5,
        "Des_pkts_ratio": 0.5,
        "Scr_bytes_ratio": 0.5,
        "Des_bytes_ratio": 0.5,
        "Avg_user_time": 6931,
        "Std_user_time": 6.416007248,
        "Avg_nice_time": 706,
        "Std_nice_time": 0.408905857,
        "Avg_system_time": 1693,
        "Std_system_time": 0.771635277,
        "Avg_iowait_time": 2423,
        "Std_iowait_time": 3.829809525,
        "Avg_ideal_time": 88245,
        "Std_ideal_time": 7.112108337,
        "Avg_tps": 37.4,
        "Std_tps": 40.19004852,
        "Avg_rtps": 30.1,
        "Std_rtps": 39.79811553,
        "Avg_wtps": 7.3,
        "Std_wtps": 3.1,
        "Avg_ldavg_1": 0.55,
        "Std_ldavg_1": 0.02,
        "Avg_kbmemused": 921020.4,
        "Std_kbmemused": 2139.652645,
        "Avg_num_Proc/s": 1,
        "Std_num_proc/s": 0,
        "Avg_num_cswch/s": 1603.3,
        "std_num_cswch/s": 294.1390997,
        "OSSEC_alert": 0,
        "OSSEC_alert_level": 0,
        "Login_attempt": 0,
        "Succesful_login": 0,
        "File_activity": 0,
        "Process_activity": 0,
        "read_write_physical.process": 0,
        "is_privileged": 0,
        "class1": "Normal",
        "class2": "Normal",
        "class3": "Normal"
    },
    {
        "Date": "9/01/2020",
        "Timestamp": 1578540956,
        "Scr_IP": "192.168.2.199",
        "Scr_port": 49278,
        "Des_IP": "192.168.2.10",
        "Des_port": 80,
        "Protocol": "tcp",
        "Service": "http",
        "Duration": 0.67369,
        "Scr_bytes": 13437,
        "Des_bytes": 34924,
        "Conn_state": 1,
        "missed_bytes": 0,
        "is_syn_only": True,
        "Is_SYN_ACK": True,
        "is_pure_ack": True,
        "is_with_payload": True,
        "FIN or RST": True,
        "Bad_checksum": False,
        "is_SYN_with_RST": False,
        "Scr_pkts": 105,
        "Scr_ip_bytes": 18905,
        "Des_pkts": 105,
        "Des_ip_bytes": 40392,
        "anomaly_alert": True,
        "total_bytes": 107658,
        "total_packet": 210,
        "paket_rate": 311.7160712,
        "byte_rate": 159803.4704,
        "Scr_packts_ratio": 0.5,
        "Des_pkts_ratio": 0.5,
        "Scr_bytes_ratio": 0.300414275,
        "Des_bytes_ratio": 0.699585725,
        "Avg_user_time": 9207,
        "Std_user_time": 5.55584206,
        "Avg_nice_time": 10994,
        "Std_nice_time": 1.356305275,
        "Avg_system_time": 4864,
        "Std_system_time": 1.873004004,
        "Avg_iowait_time": 311,
        "Std_iowait_time": 0.224653066,
        "Avg_ideal_time": 74624,
        "Std_ideal_time": 8.245611196,
        "Avg_tps": 12297,
        "Std_tps": 10.38585004,
        "Avg_rtps": 8,
        "Std_rtps": 10.50714043,
        "Avg_wtps": 4297,
        "Std_wtps": 2.723578712,
        "Avg_ldavg_1": 2146,
        "Std_ldavg_1": 0.102781321,
        "Avg_kbmemused": 915852.8,
        "Std_kbmemused": 2507.97563,
        "Avg_num_Proc/s": 5.1,
        "Std_num_proc/s": 3.238826948,
        "Avg_num_cswch/s": 2806.2,
        "std_num_cswch/s": 158.7493622,
        "OSSEC_alert": 1,
        "OSSEC_alert_level": 5,
        "Login_attempt": 0,
        "Succesful_login": 0,
        "File_activity": 0,
        "Process_activity": 0,
        "read_write_physical.process": 0,
        "is_privileged": 0,
        "class1": "Scanning_vulnerability",
        "class2": "Reconnaissance",
        "class3": "Attack"
    },
    {
        "Date": "12/12/2019",
        "Timestamp": 1576100087,
        "Scr_IP": "172.24.1.80",
        "Scr_port": 39649,
        "Des_IP": "172.24.1.1",
        "Des_port": 53,
        "Protocol": "udp",
        "Service": "dns",
        "Duration": 0.000119,
        "Scr_bytes": 37,
        "Des_bytes": 37,
        "Conn_state": 1,
        "missed_bytes": 0,
        "is_syn_only": False,
        "Is_SYN_ACK": False,
        "is_pure_ack": False,
        "is_with_payload": True,
        "FIN or RST": False,
        "Bad_checksum": False,
        "is_SYN_with_RST": False,
        "Scr_pkts": 1,
        "Scr_ip_bytes": 65,
        "Des_pkts": 1,
        "Des_ip_bytes": 65,
        "anomaly_alert": False,
        "total_bytes": 204,
        "total_packet": 2,
        "paket_rate": 16806.72269,
        "byte_rate": 1714285714,
        "Scr_packts_ratio": 0.5,
        "Des_pkts_ratio": 0.5,
        "Scr_bytes_ratio": 0.5,
        "Des_bytes_ratio": 0.5,
        "Avg_user_time": 2.47,
        "Std_user_time": 4.589873637,
        "Avg_nice_time": 2292,
        "Std_nice_time": 1.712418173,
        "Avg_system_time": 2552,
        "Std_system_time": 0.949187021,
        "Avg_iowait_time": 504,
        "Std_iowait_time": 1.098446175,
        "Avg_ideal_time": 92183,
        "Std_ideal_time": 5.98835712,
        "Avg_tps": 10.3,
        "Std_tps": 7.523961722,
        "Avg_rtps": 5.1,
        "Std_rtps": 6.425729531,
        "Avg_wtps": 5.2,
        "Std_wtps": 5.035871325,
        "Avg_ldavg_1": 0.44,
        "Std_ldavg_1": 0.02,
        "Avg_kbmemused": 907108,
        "Std_kbmemused": 2995.482465,
        "Avg_num_Proc/s": 1,
        "Std_num_proc/s": 0,
        "Avg_num_cswch/s": 1185.5,
        "std_num_cswch/s": 188.0756497,
        "OSSEC_alert": 0,
        "OSSEC_alert_level": 0,
        "Login_attempt": 0,
        "Succesful_login": 0,
        "File_activity": 0,
        "Process_activity": 0,
        "read_write_physical.process": 0,
        "is_privileged": 0,
        "class1": "Normal",
        "class2": "Normal",
        "class3": "Normal"
    }
    ]

    parser = argparse.ArgumentParser(description="Inferencia con el modelo de detección de anomalías")
    parser.add_argument(
        "--modelo_path",
        type = str,
        default = "modelos/RF_model.joblib",
        help = "Ruta del archivo del modelo entrenado"
    )
    parser.add_argument(
        "--input_data",
        type = str,
        default = dato_prueba,
        help = "Ruta del archivo de datos de entrada"
    )
    parser.add_argument(
        "--artefactos_path",
        type = str,
        default = "datos/preprocesados/artefactos_inferencia.joblib",
        help = "Ruta del archivo de artefactos para el preprocesamiento"
    )
    args = parser.parse_args()

    # Crear instancia del modelo de detección de anomalías
    modelo = modelo_deteccion_anomalias(args.modelo_path)
    # Cargar datos de entrada
    X_input = pd.DataFrame(args.input_data)
    X_input = preprocesamiento_inferencia(X_input, args.artefactos_path)
    # Realizar predicciones
    predicciones = modelo.predecir(X_input)
    print("Predicciones realizadas exitosamente.")
    print(predicciones)

    # Guardar predicciones en un archivo CSV
    output_pred_path = "predicciones/predicciones_test.csv"
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    pd.DataFrame(predicciones, columns=["prediccion"]).to_csv(output_pred_path, index=False)
    print(f"Predicciones guardadas en: {output_pred_path}")
