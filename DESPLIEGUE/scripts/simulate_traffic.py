import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = "http://localhost:8000/predict"
DEFAULT_PAYLOAD = Path(__file__).resolve().parents[1] / "data_output" / "predict_example.json"
DEFAULT_TEST_DATA = Path(__file__).resolve().parents[1] / "data_output" / "X_test_api.csv"

MODE_CONFIG = {
    "normal": {"requests": 1000, "delay": 1.0, "jitter": 0.2},
    "burst": {"requests": 1000, "delay": 0.1, "jitter": 0.05},
    "slow": {"requests": 1000, "delay": 3.0, "jitter": 0.5},
    "anomalous": {"requests": 150, "delay": 0.2, "jitter": 0.05},
}

STRING_FIELDS = {"Date", "Scr_IP", "Des_IP", "Protocol", "Service"}
BOOL_FIELDS = {
    "is_syn_only",
    "Is_SYN_ACK",
    "is_pure_ack",
    "is_with_payload",
    "FIN_or_RST",
    "Bad_checksum",
    "is_SYN_with_RST",
    "anomaly_alert",
}
INT_FIELDS = {
    "Timestamp",
    "Scr_port",
    "Des_port",
    "Scr_bytes",
    "Des_bytes",
    "Conn_state",
    "missed_bytes",
    "Scr_pkts",
    "Scr_ip_bytes",
    "Des_pkts",
    "Des_ip_bytes",
    "total_bytes",
    "total_packet",
    "OSSEC_alert",
    "OSSEC_alert_level",
    "Login_attempt",
    "Succesful_login",
    "File_activity",
    "Process_activity",
    "read_write_physical_process",
    "is_privileged",
}

ANOMALOUS_NUMERIC_FIELDS = {
    "Duration",
    "missed_bytes",
    "total_bytes",
    "total_packet",
    "paket_rate",
    "byte_rate",
    "Avg_user_time",
    "Std_user_time",
    "Avg_nice_time",
    "Std_nice_time",
    "Avg_system_time",
    "Std_system_time",
    "Avg_iowait_time",
    "Std_iowait_time",
    "Avg_tps",
    "Std_tps",
    "Avg_rtps",
    "Std_rtps",
    "Avg_wtps",
    "Std_wtps",
    "Avg_ldavg_1",
    "Std_ldavg_1",
    "Avg_kbmemused",
    "Std_kbmemused",
    "Avg_num_Proc_s",
    "Avg_num_cswch_s",
    "std_num_cswch_s",
    "OSSEC_alert_level",
}


def load_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("El payload debe ser una lista de registros para POST /predict")

    return payload


def parse_csv_value(value):
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False

    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return value


def is_valid_value(field_name, value):
    if value is None:
        return False

    if field_name in BOOL_FIELDS:
        return isinstance(value, bool)

    if field_name in INT_FIELDS:
        return isinstance(value, int) and not isinstance(value, bool)

    if field_name in STRING_FIELDS:
        return isinstance(value, str) and value != ""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_valid_row(row, reference_row):
    for key in reference_row.keys():
        if key not in row or not is_valid_value(key, row[key]):
            return False
    return True


def load_test_rows(path, reference_payload):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero el preprocesamiento para generar X_test_api.csv "
            "o usa --source example."
        )

    reference_row = reference_payload[0]
    valid_rows = []
    skipped_rows = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {key: parse_csv_value(value) for key, value in row.items()}

            if is_valid_row(parsed_row, reference_row):
                valid_rows.append({key: parsed_row[key] for key in reference_row.keys()})
            else:
                skipped_rows += 1

    if not valid_rows:
        raise ValueError(f"No hay filas disponibles en {path}")

    if skipped_rows:
        print(f"Filas de test descartadas por no cumplir el contrato de la API: {skipped_rows}")

    return valid_rows


def perturb_anomalous_row(row, factor):
    anomalous_row = dict(row)

    for field in ANOMALOUS_NUMERIC_FIELDS:
        value = anomalous_row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            new_value = value * factor
            anomalous_row[field] = int(round(new_value)) if field in INT_FIELDS else float(new_value)

    anomalous_row["anomaly_alert"] = True
    anomalous_row["OSSEC_alert"] = 1
    anomalous_row["OSSEC_alert_level"] = max(int(anomalous_row.get("OSSEC_alert_level") or 0), 8)
    return anomalous_row


def build_payload(source, example_payload, test_rows, anomaly_factor):
    if source == "example":
        return example_payload
    if source == "anomalous":
        return [perturb_anomalous_row(random.choice(test_rows), anomaly_factor)]

    return [random.choice(test_rows)]


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        latency = time.perf_counter() - start
        return response.status, body, latency


def sleep_between_requests(base_delay, jitter):
    if base_delay <= 0:
        return

    random_jitter = random.uniform(0, jitter) if jitter > 0 else 0
    time.sleep(base_delay + random_jitter)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulador de trafico para alimentar las metricas de la API DISIA."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"URL del endpoint /predict. Por defecto: {DEFAULT_URL}")
    parser.add_argument("--payload", default=str(DEFAULT_PAYLOAD), help="Ruta al JSON usado como cuerpo de la peticion")
    parser.add_argument("--source", choices=["test", "example", "anomalous"], default="test", help="Origen de los datos enviados")
    parser.add_argument("--test-data", default=str(DEFAULT_TEST_DATA), help="CSV de test crudo compatible con la API")
    parser.add_argument("--mode", choices=MODE_CONFIG.keys(), default="normal", help="Perfil de trafico preconfigurado")
    parser.add_argument("--requests", type=int, default=None, help="Numero de peticiones. Sobrescribe el modo")
    parser.add_argument("--delay", type=float, default=None, help="Espera base entre peticiones en segundos. Sobrescribe el modo")
    parser.add_argument("--jitter", type=float, default=None, help="Variacion aleatoria adicional en segundos")
    parser.add_argument("--anomaly-factor", type=float, default=8.0, help="Factor usado para inflar campos numericos en --source anomalous")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout por peticion en segundos")
    parser.add_argument("--show-response", action="store_true", help="Muestra el cuerpo de cada respuesta")
    return parser.parse_args()


def main():
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    total_requests = args.requests if args.requests is not None else config["requests"]
    delay = args.delay if args.delay is not None else config["delay"]
    jitter = args.jitter if args.jitter is not None else config["jitter"]
    example_payload = load_payload(args.payload)
    test_rows = load_test_rows(args.test_data, example_payload) if args.source in {"test", "anomalous"} else []

    successes = 0
    failures = 0
    latencies = []

    print(f"Simulando trafico contra {args.url}")
    print(f"Modo={args.mode} origen={args.source} peticiones={total_requests} delay={delay}s jitter={jitter}s")

    for index in range(1, total_requests + 1):
        try:
            payload = build_payload(args.source, example_payload, test_rows, args.anomaly_factor)
            status, body, latency = post_json(args.url, payload, args.timeout)
            successes += 1
            latencies.append(latency)
            print(f"[{index}/{total_requests}] HTTP {status} latencia={latency:.3f}s")

            if args.show_response:
                print(body)
        except urllib.error.HTTPError as exc:
            failures += 1
            error_body = exc.read().decode("utf-8", errors="replace")
            print(f"[{index}/{total_requests}] ERROR HTTP {exc.code}: {error_body}")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failures += 1
            print(f"[{index}/{total_requests}] ERROR {exc}")

        if index < total_requests:
            sleep_between_requests(delay, jitter)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    print("\nResumen")
    print(f"OK: {successes}")
    print(f"Errores: {failures}")
    print(f"Latencia media: {avg_latency:.3f}s")
    print(f"Latencia maxima: {max_latency:.3f}s")


if __name__ == "__main__":
    main()
