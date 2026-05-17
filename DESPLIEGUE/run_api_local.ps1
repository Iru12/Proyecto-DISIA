$ErrorActionPreference = "Stop"

$DeployRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $DeployRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "No se encontro el interprete del entorno virtual: $Python"
}

$env:MODELO_PATH = Join-Path $DeployRoot "models_output\RF_model.joblib"
$env:ARTEFACTOS_PATH = Join-Path $DeployRoot "data_output\artefactos_inferencia.joblib"
$env:DRIFT_REFERENCE_PATH = Join-Path $DeployRoot "data_output\drift_reference.json"
$env:MODEL_METADATA_PATH = Join-Path $DeployRoot "models_output\current_model_metadata.json"
$env:MODEL_SELECTION_HISTORY_PATH = Join-Path $DeployRoot "models_output\model_selection_history.jsonl"
$env:ALERT_HISTORY_PATH = Join-Path $DeployRoot "models_output\alerts_history.jsonl"

if (-not $env:ALERT_OPERATIONAL_WINDOW_SIZE) { $env:ALERT_OPERATIONAL_WINDOW_SIZE = "50" }
if (-not $env:ALERT_OPERATIONAL_MIN_WINDOW_SIZE) { $env:ALERT_OPERATIONAL_MIN_WINDOW_SIZE = "10" }
if (-not $env:ALERT_ERROR_RATE_THRESHOLD) { $env:ALERT_ERROR_RATE_THRESHOLD = "0.30" }
if (-not $env:ALERT_LATENCY_P95_THRESHOLD) { $env:ALERT_LATENCY_P95_THRESHOLD = "3.0" }
if (-not $env:ALERT_MODEL_F1_MACRO_MIN) { $env:ALERT_MODEL_F1_MACRO_MIN = "0.90" }

Set-Location (Join-Path $DeployRoot "src")
& $Python -m uvicorn api:app --host 127.0.0.1 --port 8000
