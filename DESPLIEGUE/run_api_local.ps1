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

Set-Location (Join-Path $DeployRoot "src")
& $Python -m uvicorn api:app --host 127.0.0.1 --port 8000
