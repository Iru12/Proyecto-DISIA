# Prueba completa de despliegue

Guia corta para probar API, Prometheus, Grafana, deriva, alertas, validacion y reentrenamiento.

## 1. Preparar entorno

En `.env`, para demo local:

```env
API_PORT=18080
RETRAINING_MIN_VALIDATED_SAMPLES=5
```

Levantar desde `DESPLIEGUE`:

```bash
docker compose build
docker compose run --rm train
docker compose up inferencia prometheus grafana
```

Comprobar:

```bash
curl http://localhost:18080/health
```

Abrir:

```text
API:        http://localhost:18080/docs
Prometheus: http://localhost:9090/targets
Grafana:    http://localhost:3000
```

Grafana:

```text
usuario: admin
password: admin
```

Dashboard:

```text
Dashboards > DISIA > DISIA - Observabilidad API
```

## 2. Estado limpio

```bash
curl -X POST http://localhost:18080/admin/drift/reset
curl -X POST http://localhost:18080/admin/alerts/reset
```

Opcional, limpiar colas de demo:

```bash
> data_output/retraining_review_queue.jsonl
> data_output/retraining_validated.jsonl
```

## 3. Trafico base

Enviar volumen grande de trafico estable:

```bash
python scripts/simulate_traffic.py --url http://localhost:18080/predict --source example --requests 5000 --delay 0
```

Comprobar en Grafana:

```text
Predicciones
Trafico de inferencia
Latencia
Distribucion de predicciones
```

## 4. Trafico anomalo

Enviar trafico fuera de perfil para activar deriva:

```bash
python scripts/simulate_traffic.py --url http://localhost:18080/predict --source anomalous --mode anomalous --requests 500 --delay 0.01
```

Comprobar:

```bash
curl http://localhost:18080/drift/status
curl http://localhost:18080/alerts/status
curl http://localhost:18080/admin/retraining/review-queue
```

Esperado:

```text
Deriva activa
Alerta model_data_drift
Muestras pendientes de revision
```

## 5. Validacion simulada

Validar las ultimas 10 muestras pendientes como demo:

```bash
python scripts/validate_review_queue.py --base-url http://localhost:18080 --limit 10 --label generic_scanning
```

Comprobar en Grafana:

```text
Reentrenamiento y validacion
Pendientes de revision baja
Muestras validadas sube
Progreso validacion >= 100%
```

## 6. Reentrenamiento

Provocar nuevo intento con unas muestras anomalas:

```bash
python scripts/simulate_traffic.py --url http://localhost:18080/predict --source anomalous --mode anomalous --requests 5 --delay 0.05
```

Comprobar logs:

```bash
docker logs api --tail 200
```

Buscar:

```text
Iniciando proceso de re-entrenamiento
Entrenamiento completado exitosamente
Modelo reentrenado cargado en memoria
```

Comprobar modelo nuevo:

```bash
ls -lh models_output
curl http://localhost:18080/model-info
curl http://localhost:18080/admin/models/history
```

Esperado:

```text
RF_v2.joblib o version superior
Modelo activo actualizado
Ultimo reentrenamiento OK en Grafana
```

## 7. Parar

```bash
docker compose down
```
