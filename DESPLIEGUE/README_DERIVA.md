# Deteccion de deriva de datos

Este documento explica la parte de deriva de datos implementada para el Hito 5.

## Contexto

El sistema ya dispone de un modelo supervisado capaz de clasificar eventos IIoT en una de las clases del dataset X-IIoTID. Ese modelo responde a la pregunta:

```text
Que tipo de trafico o ataque es esta instancia?
```

La deriva de datos responde a una pregunta distinta:

```text
Los datos que llegan ahora se siguen pareciendo a los datos normales que conocia el sistema?
```

Por tanto, la deteccion de deriva no sustituye al modelo `RandomForest`. Es una capa de monitorizacion que vigila si el perfil estadistico de las entradas de produccion empieza a alejarse del perfil esperado.

## Problema del dataset

El dataset X-IIoTID no esta planteado como una serie temporal productiva estricta. Sus filas sirven para entrenar y evaluar clasificadores, pero no representan necesariamente una secuencia real del tipo:

```text
evento 1 -> evento 2 -> evento 3 -> cambio progresivo del sistema
```

Por eso, implementar una deriva temporal real seria forzar una suposicion que el dataset no garantiza. Para el Hito 5 se ha adoptado una aproximacion mas defendible: simular monitorizacion online usando ventanas de peticiones y comparar esas peticiones contra un perfil de normalidad aprendido previamente.

## Idea propuesta

La recomendacion comentada por Rafa consistia en usar el conjunto de validacion para aprender que aspecto tiene el trafico normal. En vez de detectar el tipo exacto de ataque, se aprende un perfil de normalidad:

```text
1. Coger instancias normales.
2. Aprender rangos habituales de sus variables.
3. Comparar las nuevas peticiones contra esos rangos.
4. Activar una alerta si las peticiones recientes se salen demasiado del perfil normal.
```

Esta aproximacion es no supervisada en la fase de monitorizacion: durante la inferencia no necesitamos saber la etiqueta real de la nueva muestra. Solo medimos si su distribucion se parece o no al perfil normal aprendido.

## Como se ha implementado

Durante el preprocesamiento de despliegue, el script:

```text
DESPLIEGUE/src/preprocesamiento.py
```

filtra las muestras normales del conjunto de validacion:

```text
class3 = normal
```

Despues genera una referencia de deriva en:

```text
DESPLIEGUE/data_output/drift_reference.json
```

Esa referencia contiene, para cada una de las 34 variables finales del modelo, un rango esperado calculado con cuantiles:

```text
q_low  = percentil 1
q_high = percentil 99
```

La implementacion principal esta en:

```text
DESPLIEGUE/src/drift.py
```

La API carga esta referencia al arrancar y mantiene una ventana deslizante de peticiones recientes. Cada vez que llega una peticion a `/predict`, ocurre lo siguiente:

```text
1. La API recibe el JSON de entrada.
2. Se aplica el mismo preprocesamiento que en entrenamiento.
3. El modelo genera la prediccion.
4. El monitor de deriva calcula cuantas variables quedan fuera del perfil normal.
5. Se actualiza la ventana de observaciones recientes.
6. Si la ultima muestra o la ventana reciente superan el umbral de deriva, se activa la alerta `model_data_drift`.
```

El modelo no se reentrena durante estas peticiones. El flujo de despliegue queda separado:

```text
docker compose run --rm train
```

entrena y guarda artefactos, mientras que:

```text
docker compose up inferencia prometheus grafana
```

solo levanta la API, carga el modelo entrenado y observa las peticiones.

## Scores de deriva

El monitor calcula dos valores principales:

```text
last_sample_score
```

Proporcion de variables de la ultima muestra que caen fuera del rango normal.

```text
rolling_drift_score
```

Media del score en la ventana deslizante reciente.

Ejemplo:

```text
rolling_drift_score = 0.37
```

significa que, de media, el 37% de las variables monitorizadas se estan saliendo del perfil normal en la ventana reciente.

Los umbrales actuales son:

```text
sample_alert_threshold = 0.15
window_alert_threshold = 0.15
min_window_size = 30
window_size = 100
```

La alerta de ventana no se activa hasta tener al menos 30 observaciones. Esto evita disparos demasiado tempranos por acumulacion con pocas muestras. La alerta por muestra individual si puede activarse antes si una peticion aislada queda muy fuera del perfil normal.

## Endpoints nuevos

Consultar estado de deriva:

```text
GET /drift/status
```

Reiniciar la ventana de deriva, util para demos:

```text
POST /admin/drift/reset
```

El endpoint de prediccion tambien devuelve informacion resumida de deriva:

```text
POST /predict
```

Respuesta abreviada:

```json
{
  "prediccion": ["normal"],
  "prediccion_codificada": [14],
  "drift": {
    "alert_active": false,
    "last_sample_score": 0.147,
    "rolling_drift_score": 0.147,
    "alert_reason": null
  }
}
```

## Metricas Prometheus

La API expone estas metricas nuevas en:

```text
GET /metrics
```

Metricas:

```text
api_drift_score{score_type="last_sample"}
api_drift_score{score_type="rolling_window"}
api_drift_alert_active
api_drift_alerts_total
api_drift_window_observations
```

Estas metricas permiten conectar la deriva con Prometheus, Grafana y, mas adelante, reglas de alerta.

## Simulador de trafico

El script:

```text
DESPLIEGUE/scripts/simulate_traffic.py
```

manda peticiones a `POST /predict` para simular trafico online. No entrena el modelo ni modifica los artefactos: solo alimenta inferencia, metricas, deriva y alertas.

Los parametros importantes son:

```text
--source test       usa filas reales del split de test
--source example    repite un ejemplo fijo compatible con la API
--source anomalous  altera filas validas para simular datos fuera del perfil normal

--mode normal       ritmo tranquilo de peticiones
--mode burst        ritmo rapido
--mode slow         ritmo lento
--mode anomalous    perfil de demo para trafico anomalo
```

Importante: `--mode normal` no significa necesariamente "clase normal del dataset"; significa ritmo normal de envio. Para probar recuperacion limpia de deriva conviene usar `--source example`, porque repite un ejemplo estable y sano.

## Demo Docker

Levantar la API:

```powershell
cd DESPLIEGUE
docker compose up inferencia
```

Consultar estado inicial:

```powershell
Invoke-RestMethod http://localhost:18080/drift/status
```

Resetear la ventana de deriva:

```powershell
Invoke-RestMethod http://localhost:18080/admin/drift/reset -Method Post
```

Simular trafico normal:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source example --requests 80 --delay 0.03
```

Simular trafico anomalo:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --mode anomalous --source anomalous --requests 80 --delay 0.05
```

El modo `anomalous` conserva el contrato JSON de la API, pero infla varias metricas numericas para simular un entorno fuera del perfil normal.

### Secuencia recomendada para la demo

Desde `DESPLIEGUE`:

```powershell
Invoke-RestMethod http://localhost:18080/admin/drift/reset -Method Post
Invoke-RestMethod http://localhost:18080/admin/alerts/reset -Method Post
```

1. Enviar trafico sano:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source example --requests 80 --delay 0.03
```

Comprobar:

```powershell
Invoke-RestMethod http://localhost:18080/drift/status
```

Esperado:

```text
alert_active = false
rolling_drift_score < 0.15
```

2. Enviar trafico anomalo:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source anomalous --mode anomalous --requests 80 --delay 0.05
```

Comprobar:

```powershell
Invoke-RestMethod http://localhost:18080/drift/status
Invoke-RestMethod http://localhost:18080/alerts/status
```

Esperado:

```text
alert_active = true
rolling_drift_score > 0.15
model_data_drift activa
```

3. Recuperar con trafico sano:

```powershell
..\.venv\Scripts\python.exe .\scripts\simulate_traffic.py --source example --requests 160 --delay 0.02
```

Esperado:

```text
alert_active vuelve a false
model_data_drift pasa a off
```

Esto pasa porque la deriva usa una ventana deslizante. Cuando entran suficientes peticiones sanas, las muestras anomalas salen de la ventana y el `rolling_drift_score` cae por debajo del umbral.

## Visualizacion en Grafana

Con la pila completa:

```powershell
docker compose up inferencia prometheus grafana
```

Grafana queda disponible en:

```text
http://localhost:3000
```

Dashboard:

```text
Dashboards > DISIA > DISIA - Observabilidad API
```

La seccion `Deriva y alertas` muestra:

```text
Deriva activa
Score de deriva
Alertas activas
Estado por alerta
Evolucion de deriva
Alertas disparadas
```

Los endpoints `/drift/status` y `/alerts/status` siguen siendo utiles para ver el detalle tecnico completo, como `top_features`, `alert_reason` e historico.

## Validacion realizada

Se probo el siguiente escenario:

```text
35 predicciones del ejemplo normal -> alert_active = false
35 peticiones anomalas simuladas   -> alert_active = true
```

Resultado observado con trafico anomalo:

```text
rolling_drift_score = 0.3764
alert_active = true
alert_reason = ventana_deslizante_fuera_de_perfil
```

Esto confirma que el monitor no se limita a guardar metricas, sino que detecta un cambio de distribucion cuando llegan muestras artificialmente alejadas del perfil normal.

## Limitaciones

Esta deriva es una simulacion razonable para el Hito 5, no una prueba de deriva temporal real en produccion.

Limitaciones principales:

- El dataset no garantiza orden temporal real.
- El perfil normal se aprende desde validacion, no desde trafico real de una planta industrial.
- Los umbrales son heuristicos y se han fijado para una demo comprensible.
- El envio externo de alertas requiere configurar Telegram mediante `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`; si no se configura, quedan registradas localmente y visibles en API, Prometheus y Grafana.

## Como defenderlo

Una forma clara de explicarlo en la memoria seria:

```text
Dado que X-IIoTID no representa una secuencia temporal productiva estricta, la deteccion de deriva se implementa como una monitorizacion no supervisada de cambio de distribucion respecto a un perfil de trafico normal aprendido a partir del conjunto de validacion. Este enfoque permite simular el comportamiento esperado en produccion sin asumir una temporalidad que el dataset no garantiza.
```

## Relacion con el Hito 5

Esta implementacion cubre la parte de:

```text
Deteccion de deriva de datos
```

Tambien deja preparada la base para:

```text
Alertas sobre metricas de deriva
Monitorizacion en Prometheus/Grafana
Feedback loop y reentrenamiento
```

Lo que queda pendiente como mejora futura es formalizar el feedback loop/reentrenamiento automatico. Las alertas ya quedan auditadas localmente y pueden enviarse a canales externos si se configuran las variables correspondientes.
