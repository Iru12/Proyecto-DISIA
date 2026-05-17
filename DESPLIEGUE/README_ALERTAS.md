# Sistema de alertas

Este documento describe las alertas implementadas para el Hito 5.

## Objetivo

El sistema de alertas convierte las metricas de observabilidad en avisos accionables. La idea es que no sea necesario mirar Grafana continuamente: si aparece una condicion anomala, la API registra una alerta y, opcionalmente, la envia a un canal externo.

La implementacion cubre dos requisitos del Hito 5:

```text
1. Alerta sobre metrica operativa.
2. Alerta sobre metrica de modelo/calidad.
```

## Canales de alerta

Siempre queda activo un historial local en formato JSONL:

```text
DESPLIEGUE/models_output/alerts_history.jsonl
```

Ademas, se pueden activar canales externos mediante variables de entorno:

```text
ALERT_WEBHOOK_URL
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Si no se configuran, la alerta no se envia fuera, pero sigue quedando registrada localmente y expuesta por la API.

## Estado activo e historico

El endpoint:

```text
GET /alerts/status
```

devuelve dos bloques importantes:

```text
active_alerts
recent_alerts
```

`active_alerts` representa las alertas encendidas ahora mismo. Es el bloque que se debe mirar para saber el estado actual del sistema.

`recent_alerts` es un historico reciente leido desde `alerts_history.jsonl`. Puede mostrar alertas antiguas aunque ya no esten activas. Esto es util para auditoria, pero no significa necesariamente que la alerta siga encendida.

Las alertas pueden resolverse automaticamente. Por ejemplo:

```text
operational_http_error_rate
```

se apaga cuando entran suficientes peticiones correctas y la tasa de errores de la ventana reciente cae por debajo del umbral.

```text
model_data_drift
```

se apaga cuando entran suficientes peticiones sanas y el score de deriva de la ventana reciente vuelve por debajo del umbral.

El campo `cooldown_active` indica que la alerta sigue activa pero el sistema no vuelve a escribir/enviar el mismo aviso continuamente durante el periodo de enfriamiento.

## Alertas implementadas

### 1. Tasa elevada de errores HTTP

Clave:

```text
operational_http_error_rate
```

Categoria:

```text
operational
```

Se activa cuando el porcentaje de respuestas HTTP con estado `>= 400` supera el umbral configurado en la ventana reciente de peticiones.

Variables:

```text
ALERT_OPERATIONAL_WINDOW_SIZE=50
ALERT_OPERATIONAL_MIN_WINDOW_SIZE=10
ALERT_ERROR_RATE_THRESHOLD=0.30
```

Con la configuracion por defecto, la alerta se evalua cuando existen al menos 10 peticiones y se activa si el 30% o mas son errores HTTP.

Tambien se resuelve automaticamente cuando las ultimas peticiones vuelven a ser correctas y la tasa de error baja del umbral.

### 2. Latencia elevada

Clave:

```text
operational_high_latency
```

Categoria:

```text
operational
```

Se activa cuando la latencia p95 de la ventana reciente supera el umbral configurado.

Variable:

```text
ALERT_LATENCY_P95_THRESHOLD=3.0
```

### 3. F1 macro bajo del modelo

Clave:

```text
model_f1_macro_low
```

Categoria:

```text
model
```

Se activa si el `f1_macro` del modelo activo cae por debajo del umbral configurado. Esta alerta se evalua al arrancar la API y al cambiar de modelo mediante `/admin/models/select`.

Variable:

```text
ALERT_MODEL_F1_MACRO_MIN=0.90
```

Para una demo forzada, se puede arrancar la API con un umbral mas alto que el rendimiento real del modelo, por ejemplo:

```powershell
$env:ALERT_MODEL_F1_MACRO_MIN="0.99"
.\run_api_local.ps1
```

### 4. Deriva de datos activa

Clave:

```text
model_data_drift
```

Categoria:

```text
model
```

Se activa cuando el detector de deriva indica que la ventana reciente de peticiones se sale del perfil normal aprendido. Usa la salida de `/drift/status` y las metricas:

```text
api_drift_score
api_drift_alert_active
```

Tambien se resuelve automaticamente cuando las peticiones recientes vuelven a parecerse al perfil normal.

## Endpoints

Consultar estado de alertas:

```text
GET /alerts/status
```

Resetear estado en memoria, util para demos:

```text
POST /admin/alerts/reset
```

Enviar una alerta manual de prueba:

```text
POST /admin/alerts/test
```

Body:

```json
{
  "category": "operational",
  "severity": "warning",
  "title": "Alerta de prueba",
  "detail": "Validacion manual del canal de alertas"
}
```

## Metricas Prometheus

La API expone:

```text
api_alerts_total
api_alert_active
```

Ejemplo de consulta:

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics).Content | Select-String "api_alert"
```

En Prometheus tambien se pueden consultar directamente:

```text
api_alert_active
api_alert_active{key="model_data_drift"}
api_alert_active{key="operational_http_error_rate"}
api_alerts_total
```

`api_alert_active` vale `1` si la alerta esta activa y `0` si esta apagada. `api_alerts_total` es acumulativa: cuenta cuantas veces se ha disparado una alerta desde que arranco la API.

## Visualizacion en Grafana

Grafana carga un dashboard desde:

```text
DESPLIEGUE/monitoring/grafana/dashboards/disia-api-observability.json
```

La seccion:

```text
Deriva y alertas
```

muestra:

```text
Deriva activa
Score de deriva
Alertas activas
Estado por alerta
Evolucion de deriva
Alertas disparadas
```

La interfaz de Grafana esta en:

```text
http://localhost:3000
```

Dashboard:

```text
Dashboards > DISIA > DISIA - Observabilidad API
```

## Demo local

Levantar la API:

```powershell
cd DESPLIEGUE
.\run_api_local.ps1
```

En otra terminal, resetear alertas:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/admin/alerts/reset -Method Post
```

### Demo de alerta operativa

Generar errores 404 llamando a una ruta inexistente:

```powershell
1..25 | ForEach-Object {
  try {
    Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ruta-inexistente
  } catch {}
}
```

Consultar alertas:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/alerts/status
```

Debe aparecer activa:

```text
operational_http_error_rate
```

### Demo de alerta de deriva

Resetear deriva y alertas:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/admin/drift/reset -Method Post
Invoke-RestMethod http://127.0.0.1:8000/admin/alerts/reset -Method Post
```

Enviar trafico anomalo:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --mode anomalous --source anomalous --requests 80 --delay 0.05
```

Consultar alertas:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/alerts/status
```

Debe aparecer activa:

```text
model_data_drift
```

### Demo de alertas apiladas

Para ver dos alertas activas a la vez, no se deben resetear las alertas entre una prueba y otra.

Desde `DESPLIEGUE`:

```powershell
Invoke-RestMethod http://localhost:8000/admin/drift/reset -Method Post
Invoke-RestMethod http://localhost:8000/admin/alerts/reset -Method Post
```

Activar deriva:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source anomalous --mode anomalous --requests 80 --delay 0.05
```

Sin resetear, activar la alerta HTTP:

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
active_alerts contiene model_data_drift
active_alerts contiene operational_http_error_rate
Grafana muestra Alertas activas = 2
```

### Demo de recuperacion automatica

Enviar trafico sano para limpiar las ventanas recientes:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source example --requests 160 --delay 0.02
```

Comprobar:

```powershell
Invoke-RestMethod http://localhost:8000/drift/status
Invoke-RestMethod http://localhost:8000/alerts/status
```

Resultado esperado:

```text
model_data_drift pasa a off
operational_http_error_rate pasa a off
active_alerts queda vacio o sin esas claves
```

Esto ocurre porque las alertas se calculan sobre ventanas recientes, no sobre todo el historico desde que arranco la API.

### Demo de alerta manual

Sirve para comprobar el canal local o un webhook externo sin provocar deriva ni errores reales:

```powershell
$body = @{
  category = "operational"
  severity = "warning"
  title = "Alerta de prueba"
  detail = "Validacion manual del sistema de alertas"
} | ConvertTo-Json

Invoke-RestMethod -Method Post http://localhost:8000/admin/alerts/test -ContentType "application/json" -Body $body
```

## Como defenderlo en la memoria

La parte de alertas puede explicarse asi:

```text
El sistema transforma metricas de observabilidad en eventos accionables. Se implementan alertas operativas sobre tasa de errores y latencia de la API, y alertas de calidad del modelo sobre f1_macro y deriva de datos. Todas las alertas quedan auditadas en un historico JSONL y pueden enviarse a canales externos mediante webhook o Telegram.
```
