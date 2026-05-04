# Despliegue del modelo IIoT

Este directorio simula un flujo sencillo de produccion para un modelo de deteccion de anomalias:

1. descarga de datos,
2. preprocesamiento,
3. entrenamiento,
4. versionado de artefactos,
5. exposicion mediante API REST.

## Requisitos

- Docker Desktop funcionando.
- Internet en la primera ejecucion para descargar la imagen base, dependencias y dataset.

No hace falta instalar `requirements.txt` en Windows si se usa Docker.

## Construir imagenes

```bash
docker compose build
```

## Entrenar modelo como job offline

Entrenamiento por defecto:

```bash
docker compose run --rm train
```

Entrenamiento versionado con parametros:

```bash
MODEL_VERSION=RF_v2 N_ESTIMATORS=250 MIN_SAMPLES_LEAF=3 docker compose run --rm train
```

En PowerShell, las variables se pasan asi:

```powershell
$env:MODEL_VERSION="RF_v2"
$env:N_ESTIMATORS="250"
$env:MIN_SAMPLES_LEAF="3"
docker compose run --rm train
```

Este servicio se ejecuta como un job offline: empieza, descarga/preprocesa datos, entrena, guarda artefactos y termina. La API de inferencia queda separada del entrenamiento para no mezclar cargas pesadas con el servicio online.

El entrenamiento genera archivos en:

```text
models_output/
data_output/
```

Archivos principales:

```text
models_output/RF_model.joblib
models_output/RF_v1.joblib
models_output/RF_v1_metadata.json
models_output/RF_v1_metrics.json
models_output/current_model_metadata.json
data_output/artefactos_inferencia.joblib
```

`RF_model.joblib` es el alias estable que usa la API por defecto. Los archivos `RF_v*.joblib` permiten conservar versiones historicas.

## Levantar API

```bash
docker compose up inferencia
```

Swagger:

```text
http://localhost:8000/docs
```

Salud:

```text
http://localhost:8000/health
```

Informacion del modelo:

```text
http://localhost:8000/model-info
```

Modelos disponibles:

```text
http://localhost:8000/models
```

Comparacion de modelos:

```text
http://localhost:8000/models/compare
```

Historico de cambios de modelo:

```text
http://localhost:8000/admin/models/history
```

## Probar prediccion

Desde Swagger, copiar el contenido de:

```text
examples/predict_example.json
```

en el endpoint:

```text
POST /predict
```

## Cambiar modelo activo reiniciando API

La API usa esta variable:

```yaml
MODELO_PATH=/app/modelos/RF_model.joblib
```

Para usar una version concreta, cambiarla en `docker-compose.yml`, por ejemplo:

```yaml
MODELO_PATH=/app/modelos/RF_v2.joblib
MODEL_METADATA_PATH=/app/modelos/RF_v2_metadata.json
```

Despues reiniciar:

```bash
docker compose restart inferencia
```

## Cambiar modelo activo desde la API

Para simular promocion o rollback de modelos sin reiniciar el contenedor:

```text
POST /admin/models/select
```

Body:

```json
{
  "model_version": "RF_v1"
}
```

La API valida que exista:

```text
models_output/RF_v1.joblib
```

y carga el modelo en memoria antes de activarlo.

Cada intento de cambio queda auditado en:

```text
models_output/model_selection_history.jsonl
```

El historico puede consultarse con:

```text
GET /admin/models/history
```

En una produccion real, este endpoint no deberia estar abierto publicamente. Debe protegerse con autenticacion, autorizacion, auditoria y, en sistemas mas maduros, puede sustituirse por un pipeline CI/CD que despliegue una nueva version del servicio.

## Parar servicios

```bash
docker compose down
```
