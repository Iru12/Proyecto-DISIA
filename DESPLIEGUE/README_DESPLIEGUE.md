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
data_output/drift_reference.json
data_output/X_test_api.csv
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

Metricas del modelo activo:

```text
http://localhost:8000/model/metrics
```

Metricas operativas en formato Prometheus:

```text
http://localhost:8000/metrics
```

Este endpoint expone, entre otras, estas metricas:

```text
api_http_requests_total
api_http_request_duration_seconds
api_predictions_total
api_predictions_by_class_total
api_active_model_info
api_model_metric
api_drift_score
api_drift_alert_active
api_drift_alerts_total
api_alerts_total
api_alert_active
```

Las metricas HTTP de la API excluyen endpoints internos como `/metrics` y `/health` para que Prometheus y el healthcheck no inflen el trafico de inferencia.

## Deteccion de deriva de datos

La explicacion completa de la estrategia, sus limitaciones y la demo esta en:

```text
README_DERIVA.md
```

Durante el preprocesamiento se genera:

```text
data_output/drift_reference.json
```

Esta referencia aprende un perfil no supervisado del trafico normal de validacion (`class3 = normal`) sobre las 34 variables finales del modelo. En inferencia, cada peticion se compara contra ese perfil y la API mantiene una ventana deslizante para detectar si las muestras recientes se salen de la normalidad.

Estado de deriva:

```text
http://localhost:8000/drift/status
```

Reiniciar la ventana de deriva para una demo:

```text
POST http://localhost:8000/admin/drift/reset
```

La respuesta de `/predict` incluye un bloque `drift` con `alert_active`, `last_sample_score` y `rolling_drift_score`.

## Sistema de alertas

La explicacion completa esta en:

```text
README_ALERTAS.md
```

Estado de alertas:

```text
http://localhost:8000/alerts/status
```

La API registra alertas en:

```text
models_output/alerts_history.jsonl
```

Hay alertas operativas para tasa de errores/latencia y alertas de modelo para `f1_macro` bajo y deriva activa.

El canal externo elegido para las notificaciones es Telegram. Las credenciales se configuran con:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

La plantilla esta en:

```text
.env.example
```

El archivo real `.env` no se sube a Git.

## Levantar monitorizacion con Prometheus

Prometheus recoge automaticamente las metricas de la API desde `/metrics`.

```bash
docker compose up inferencia prometheus
```

Interfaz de Prometheus:

```text
http://localhost:9090
```

Estado del target de la API:

```text
http://localhost:9090/targets
```

El target `api-inferencia` debe aparecer como `UP`.

Consultas utiles para la demo:

```text
api_http_requests_total
api_predictions_total
api_predictions_by_class_total
api_drift_score
api_drift_alert_active
api_alert_active
api_alerts_total
process_resident_memory_bytes
process_cpu_seconds_total
rate(api_http_requests_total[1m])
rate(api_predictions_total[1m])
```

## Levantar dashboard con Grafana

Grafana queda conectado automaticamente a Prometheus y carga un dashboard inicial de observabilidad.

```bash
docker compose up inferencia prometheus grafana
```

Interfaz de Grafana:

```text
http://localhost:3000
```

Credenciales locales:

```text
usuario: admin
password: admin
```

Dashboard:

```text
Dashboards > DISIA > DISIA - Observabilidad API
```

El dashboard incluye:

```text
Resumen ejecutivo: estado de la API, modelo activo y predicciones
Deriva y alertas: score de deriva, alerta activa y estado por tipo de alerta
Calidad del modelo desplegado
Trafico de inferencia
Inferencia y errores
Recursos del proceso
```

La seccion `Deriva y alertas` muestra:

```text
Deriva activa: 0/1 convertido a OK/DERIVA
Score de deriva: rolling_drift_score actual
Alertas activas: suma de alertas actualmente encendidas
Estado por alerta: model_data_drift, operational_http_error_rate, etc.
Evolucion de deriva: last_sample y rolling_window
Alertas disparadas: alertas generadas en los ultimos minutos
```

Las secciones detalladas aparecen como desplegables para que la primera vista sea limpia.
La seccion de trafico se centra en `POST /predict`, por lo que se activa al usar Swagger o el simulador de trafico.

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

Para probar con datos reales del split de test, usar el simulador con:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source test --requests 1 --delay 0 --show-response
```

El CSV usado como fuente es:

```text
data_output/X_test_api.csv
```

El archivo `examples/predict_example.json` solo se usa como plantilla del contrato JSON esperado por la API.

Respuesta esperada:

```json
{
  "prediccion": ["normal"],
  "prediccion_codificada": [14]
}
```

`prediccion` contiene la etiqueta legible del modelo. `prediccion_codificada` conserva el indice numerico interno para trazabilidad.

## Simular trafico para observabilidad

El simulador envia peticiones al endpoint `POST /predict` usando muestras reales del split de test.
Sirve para alimentar las metricas de Prometheus y ver actividad en Grafana.

El archivo usado por defecto es:

```text
data_output/X_test_api.csv
```

Este CSV se genera durante el preprocesamiento a partir del 15% reservado para test, manteniendo el formato crudo que espera la API.
Si no existe, reconstruir la imagen de entrenamiento y regenerar artefactos:

```powershell
docker compose build train
docker compose run --rm train
```

Con la API levantada:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source test --mode normal
```

Los parametros mas importantes son:

```text
--source test       usa filas reales del split de test
--source example    repite un ejemplo fijo, util para trafico sano y estable
--source anomalous  altera filas validas para simular datos fuera del perfil normal

--mode normal       ritmo tranquilo de peticiones
--mode burst        ritmo rapido
--mode slow         ritmo lento
--mode anomalous    perfil de demo para trafico anomalo
```

Importante: `--mode normal` describe el ritmo de envio, no la clase real del dataset.
Para una demo limpia de recuperacion o ausencia de deriva, usar `--source example`.

Modos disponibles:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --mode normal
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --mode burst
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --mode slow
```

Tambien se puede ajustar manualmente:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --requests 100 --delay 0.5
```

Si la API esta en otra URL:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --url http://localhost:8000/predict --requests 50 --delay 1
```

Para enviar una unica muestra real del split de test y ver la respuesta:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source test --requests 1 --delay 0 --show-response
```

Para simular un entorno anomalo y activar la deteccion de deriva:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --mode anomalous --source anomalous --requests 150 --delay 0.2
```

El modo anomalous conserva el contrato JSON de la API, pero infla varias metricas numericas para generar muestras fuera del perfil normal aprendido.

## Demo completa de deriva y alertas

La siguiente secuencia permite probar todo lo nuevo del Hito 5 con la pila Docker levantada:

```powershell
cd C:\Users\usuario\Desktop\MARCELO\MASTER\dsia-hito5\Proyecto-DISIA\DESPLIEGUE
```

URLs que se deben tener abiertas:

```text
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
Dashboard:  Dashboards > DISIA > DISIA - Observabilidad API
```

### 1. Empezar con estado limpio

```powershell
Invoke-RestMethod http://localhost:8000/admin/drift/reset -Method Post
Invoke-RestMethod http://localhost:8000/admin/alerts/reset -Method Post
```

### 2. Trafico normal y metricas sin alerta

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source example --requests 80 --delay 0.03
```

Comprobar:

```powershell
Invoke-RestMethod http://localhost:8000/drift/status
Invoke-RestMethod http://localhost:8000/alerts/status
```

Resultado esperado:

```text
rolling_drift_score por debajo de 0.15
api_drift_alert_active = 0
active_alerts vacio
```

En Grafana, la seccion `Deriva y alertas` debe mostrar `Deriva activa = OK`.

### 3. Trafico anomalo y alerta de deriva

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source anomalous --mode anomalous --requests 80 --delay 0.05
```

Comprobar:

```powershell
Invoke-RestMethod http://localhost:8000/drift/status
Invoke-RestMethod http://localhost:8000/alerts/status
```

Resultado esperado:

```text
rolling_drift_score por encima de 0.15
api_drift_alert_active = 1
model_data_drift activa
```

En Grafana, `Deriva activa` debe pasar a `DERIVA`.

### 4. Forzar alerta operativa HTTP

Sin resetear la deriva, forzar errores 404:

```powershell
1..25 | ForEach-Object {
  try {
    Invoke-WebRequest -UseBasicParsing "http://localhost:8000/ruta-inexistente" | Out-Null
  } catch {}
}
```

Comprobar:

```powershell
Invoke-RestMethod http://localhost:8000/alerts/status
```

Resultado esperado:

```text
model_data_drift activa
operational_http_error_rate activa
Alertas activas = 2 en Grafana
```

### 5. Recuperacion automatica con trafico sano

Enviar suficientes peticiones sanas para que las ventanas deslizantes se limpien:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source example --requests 160 --delay 0.02
```

Resultado esperado:

```text
Deriva activa vuelve a OK
model_data_drift pasa a off
operational_http_error_rate pasa a off
```

Esto ocurre porque tanto la deriva como la tasa de errores HTTP se calculan sobre ventanas recientes. Las alertas se activan cuando la ventana reciente es problematica y se resuelven cuando vuelve a entrar trafico sano.

### 6. Consultas utiles en Prometheus

```text
api_drift_score
api_drift_alert_active
api_alert_active
api_alerts_total
api_http_requests_total
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
