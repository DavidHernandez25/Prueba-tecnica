# Parte 2: Modelado Predictivo - Clasificación de la Dirección del Precio

Este documento responde a las preguntas planteadas en la Parte 2 de la prueba técnica, abordando la construcción de la variable objetivo, la ingeniería de características, el modelo elegido, el manejo del desbalance de clases, y la justificación de cada decisión tomada.

---

## Escenario Elegido

**Escenario 1:** _Clasificar si el precio subirá o bajará en la próxima hora._

Tambien he decidido mantener el mismo activo que en la Parte 1 (Coca Cola - KO), dado que su comportamiento presenta una distribución más balanceada entre movimientos alcistas y bajistas, lo cual resulta útil para entrenar un modelo con menos sesgo de clase.

---

## 1. ¿Cómo construiste la variable objetivo?

La variable objetivo se basa en la diferencia entre el **cierre actual** (`Close_t`) y el **cierre futuro** (`Close_t+1`). Sin embargo, en muchos casos este cambio es mínimo o irrelevante, por lo que se introdujo una **clasificación con umbral dinámico** para distinguir entre movimientos significativos y ruido.

Se definieron tres clases:

- `1` → Movimiento **alcista significativo**:  
  `Close_t+1 - Close_t > μ`

- `0` → Movimiento **bajista significativo**:  
  `Close_t+1 - Close_t < -μ`

- `2` → Movimiento **insignificante** (lateral o ruido):  
  `|Close_t+1 - Close_t| ≤ μ`

Donde `μ` es un umbral adaptativo definido como el 30% del **promedio del rango de las últimas 5 velas** (`mean(high - low)`), lo cual ajusta la sensibilidad del modelo a la volatilidad reciente y excluye las señales que solo podrian inducir ruido.

Este enfoque tiene varias ventajas:

1. Es más interpretable en un flujo de trading real, ya que permite ejecutar decisiones binarias de compra o venta.
2. Evita la regresión directa del precio, la cual suele tener ruido y ser difícil de evaluar en términos prácticos.
3. Es más fácil de integrar con métricas de rendimiento como la precisión, F1-score, y permite, eventualmente, backtests simples.


---

## 2. ¿Qué ventana de datos pasados consideraste?

No se fijó una longitud de ventana fija como input directo (por ejemplo, usar las últimas _n_ velas), sino que se extrajeron características agregadas de múltiples temporalidades:

- **H1**: Indicadores de volatilidad y retornos (retornos log, desviación estándar, skewness).
- **D1**: Indicadores de fuerza de tendencia y dirección (ADX, pendiente de la EMA20).

La lógica detrás de esto fue permitir que el modelo tenga **visibilidad contextual** sin forzar una estructura secuencial como un modelo recurrente.  
Además, se utilizaron todos los datos históricos disponibles en `yfinance` (~2 años) para asegurar diversidad de escenarios económicos. Esto se hizo con el obejtivo de utilizar la mayor cantidad de datos históricos disponible vía yfinance (aprox. 2 años en intervalos de 1 hora).
Idealmente, se deberían usar datos desde 2019 para incluir:

- Etapa pre-pandemia
- Pandemia COVID-19
- Reapertura económica
- Contexto inflacionario actual y políticas de la FED

Un mayor historial permite capturar más contextos de mercado y patrones diversos, favoreciendo la generalización del modelo.

---

## 3. Ingenieria de features

Se diseñó un conjunto robusto de variables predictoras que capturan **momentum, tendencia, fuerza, volatilidad, asimetría y posición relativa del precio** basados en diferentes temporalidades, los indicadores clave son:

### Indicadores en H1
1. **Retornos Logarítmicos** y **Sesgo de Precio `(Log_Ret, Skewness)`**:

Estas transformaciones del precio de cierre permiten al modelo entender la distribución y la dirección del movimiento del precio de una manera robusta. Los retornos logarítmicos manejan bien los cambios extremos del precio, mientras que el sesgo (`skewness`) indica si el precio tiende a inclinarse hacia valores mayores o menores, ofreciendo una señal de la presión compradora o vendedora.

2. **Pendiente de una regresión lineal `(Reg_Slope)`**: La pendiente de una regresion lineal, es una excelente herramienta para contextualizar la tendencia actual. Si la pendiente es positiva, el mercado está en una tendencia alcista; si es negativa, está en una tendencia bajista. Esta característica es crucial para el modelo, ya que la probabilidad de una subida o bajada en la siguiente hora a menudo depende de la dirección de la tendencia predominante.

3. **Indicadores de Momentum `(RSI, MACD)`** : Estos indicadores proporcionan información sobre la velocidad y la fuerza de los movimientos del precio, lo cual es fundamental para una predicción a corto plazo.

Estos indicadores complementan las transformaciones de precio más simples, proporcionando al modelo un conjunto de datos rico y multi-dimensional para la predicción.

---

## 4. ¿Cómo manejaste el desbalance de clases?

El desequilibrio de clases es un problema crítico en los datos financieros, ya que las grandes subidas o bajadas son mucho menos frecuentes que los períodos de "no cambio". Si el modelo no se aborda, el modelo puede sobreajustarse a la clase mayoritaria y fallar en predecir las clases minoritarias.

La estrategia elegida para este problema es el **oversampling con `RandomOverSampler`** (sobremuestreo). Este método duplica aleatoriamente los ejemplos de las clases minoritarias (subida y bajada) en el conjunto de entrenamiento hasta que la distribución sea más equilibrada.

Se eligió esta técnica por su sencillez y eficacia. Si bien existen métodos más complejos como `SMOTE` o `ADASYN`, estos pueden introducir ruido artificial en los datos al crear ejemplos sintéticos. En el contexto de series de tiempo financieras, donde el ruido es una parte inherente de los datos, un método más conservador como `RandomOverSampler` evita la creación de patrones falsos que podrían confundir al modelo y empeorar su desempeño en datos reales.

---

## 5. ¿Qué modelo elegiste y por qué?

Para este problema de clasificación multi-clase, la familia de algoritmos de Árboles de Decisión es la más idónea, especialmente los modelos de Gradient Boosting como `XGBoost`.

- Rendimiento en datos tabulares: Los modelos basados en árboles sobresalen en problemas con datos tabulares y relaciones no lineales, que son comunes en los mercados financieros.
- Eficiencia: Requieren menos recursos computacionales que modelos más complejos como las redes neuronales, lo cual es ventajoso dada la disponibilidad limitada de datos históricos.
- Robustez: Manejan bien los outliers y no requieren una normalización de los datos.

La complejidad del modelo se evalúa a través del rendimiento en los conjuntos de entrenamiento y prueba. Sin embargo modelos más complejos como redes neuronales no fueron utilizados porque:

- El tamaño del dataset no lo justifica.
- La complejidad del problema no requiere arquitectura secuencial ni embeddings.
- El enfoque está orientado a interpretabilidad y robustez.


---

##  6. ¿Qué métricas usaste para evaluar el modelo?


Dada la naturaleza del problema de desequilibrio de clases que es intrínseca en las series de tiempo financieras, la métrica principal no puede ser la precisión (`accuracy`), ya que podría dar una falsa sensación de buen rendimiento si el modelo solo predice la clase mayoritaria.

En su lugar, se emplearían las siguientes métricas para obtener una comprensión completa del rendimiento del modelo:

- Informe de Clasificación (`classification_report`): Esta métrica es crucial, ya que proporciona un desglose del `precision`, `recall` y `f1-score` para cada una de las tres clases.

    - `Precision`: Mide la exactitud de las predicciones positivas de cada clase. Es importante para entender cuántas de las "subidas" o "bajadas" predichas fueron correctas.
    - `Recall`: Mide la capacidad del modelo para encontrar todas las instancias de cada clase. Es vital para saber cuántas de las subidas o bajadas reales el modelo fue capaz de identificar.
    - `F1-score`: Es un promedio ponderado de la precision y el recall. Es una excelente métrica para obtener una visión balanceada del rendimiento de cada clase.

- Matriz de Confusión: Visualiza las predicciones correctas e incorrectas de cada clase, proporcionando una visión detallada de los errores del modelo.
- `Permutation feauture importance`: No solo permite la fácil comparabilidad entre modelos sino que también mide que tanto poder explicativo gana el modelo cuando se tiene en cuenta ese feature lo cual lo hace fácil de leer e interpretar, además de que es independiente a los modelos por lo que se puede comprar el performance de una  red neuronal y de un árbol de decisión.

Una vez que el modelo esté optimizado con estas métricas, se podrían utilizar herramientas de interpretabilidad como SHAP (SHapley Additive exPlanations) para entender qué características específicas están influyendo más en las predicciones, lo cual es fundamental para validar la lógica del modelo y generar confianza en sus resultados.

---

## Conclusiones Finales

- Se construyó una solución robusta, usando ingeniería de features multitemporal y un modelo interpretable.
- El etiquetado adaptativo fue clave para evitar errores por ruido de mercado.
- Se priorizó la simplicidad y generalización, sin caer en overfitting.
- El código está modularizado y documentado, y puede extenderse fácilmente a otros activos.

---

