# Parte 3: Diagrama del Modelo e Implementación

Este documento responde a las preguntas planteadas en la Parte 3 de la prueba técnica, tocando temas acerca de la implementación y posibles asuntos a tener en cuenta para su uso en vivo.

---

## Definición de la ventaja competitiva

La "ventaja" o **"edge"** que se busca en esta estrategia es la filtración de ruido y la identificación de señales direccionales de alta convicción.

A diferencia de un modelo binario que solo clasifica entre "sube" o "baja," este modelo de tres clases añade una capa de inteligencia crucial. La clase de "cambio insignificante" (Clase 2) actúa como un filtro de ruido, permitiendo al sistema identificar los períodos en los que el mercado no ofrece una dirección clara mejorando así el _noise-to-signal ratio_ (el porcentaje de señales claras). Esto es absolutamente relevante en esye contexto, ya que evita que el sistema entre en posiciones en momentos de indecisión, lo cual típicamente aumenta los costos por _slippage_ y comisiones, y expone al capital a movimientos erráticos.

Al enfocarse solo en las señales de las clases "sube" y "baja", el modelo busca mejorar el _hit-ratio_ (el porcentaje de trades ganadores) y optimizar las entradas y salidas, lo que aporta un valor tangible a la estrategia. Este modelo fue hecho pensando en que se pueden ejecutar operaciones de compra y de venta como se podría hacer en instrumentos financieros como los CFDs que simulan una accion, Coca-cola (KO) en este caso.

---

## Arquitectura de solución

La arquitectura propuesta, reflejada en el diagrama de flujo, sigue un pipeline de extremo a extremo:

- Ingesta de Datos: El proceso comienza con la extracción de datos de precios (OHLCV) de una fuente confiable como `yfinance`. En un entorno de producción, esto sería una conexión a un `websocket` o una API de un bróker.

- Procesamiento y Features: Esta es la fase de transformación de los datos crudos. Se calculan los indicadores técnicos (RSI, MACD, etc.) y la variable objetivo, y se aplican técnicas para manejar el desbalance de clases.

- Almacén de Features: Aunque no se implementó en este prototipo, en una arquitectura de producción, un `feature store` almacenaría y serviría de manera consistente las características calculadas, asegurando que los datos usados para el entrenamiento y para la predicción en tiempo real sean idénticos.

- Entrenamiento y Validación: El modelo `XGBoost` se entrena sobre los datos históricos procesados. La validación se realiza mediante un `backtest` realista, donde se evalúa el rendimiento del modelo en un conjunto de datos de prueba nunca antes visto. Si el rendimiento es insatisfactorio, el pipeline entra en un ciclo de re-entrenamiento y ajuste.

- Scoring y Ejecución: Una vez que el modelo es validado, se utiliza para generar una señal en tiempo real. Esta señal, que incluye la clase de predicción y su confianza, es el `input` principal para un módulo de gestión de riesgo que, si las condiciones se cumplen, ejecuta una orden de trading. El modelo se ejecutaría apenas se cierre la última vela y se inicie una nueva, usando el último dato completo posible.

---

## Gestión de riesgo y slippage

Considero que no es prudente usar un único modelo como guía o recomendación de inversión, despúes de todo la diversificación mitiga el riesgo, además al no tener presente el propósito específico del modelo en un entorno de inversión (confirmación, apoyo, indicador de otro modelo, etc) y sin tener en cuenta el perfil de riesgo y la operativa del tomador de desciciones es complejo crear una estrategia de riesgo realista, sin embargo en aras del ejercicio propuesto mi sugerencia como modelo de gestión de riesgo es la siguiente:
    - Stop-Loss: Un umbral de pérdida predefinido (ej. 1% del capital) que cierra automáticamente la posición y una ventana de vencimiento de terminos en caso de que no se alcance ni el _take profit_ ni el _stop loss_ evitando trades nuevos mientras ya se este ejecutando uno.
    - Un umbral de ganancia predefinido (ej. 2% del capital) que cierra la posición para asegurar el beneficio. 
        - Criterio de Cierre por Reversión de Señal: Esto significa que una posición se mantiene abierta mientras el modelo continúe prediciendo la misma dirección. Solo se cerrará la posición cuando la predicción cambie a una dirección opuesta permitiendo que la estrategia "siga la tendencia" y capture movimientos de precios más largos, maximizando las ganancias de los trades ganadores.
        - Balanceando el riesgo: Es crucial mencionar el riesgo asociado. Si el mercado tiene un cambio rápido de dirección, este criterio puede llevar a pérdidas mayores. La implementación ideal sería combinarlo con un _stop-loss_ dinámico (_trailing stop-loss_) para asegurar el beneficio o limitar las pérdidas en caso de una reversión inesperada.
    - Límite de Volumen: Un límite en el tamaño de la posición basado en el capital disponible y la volatilidad del activo, para controlar la exposición al riesgo.
- Relación con la Señal: El modelo podría generar una señal de "subida," pero si la confianza de la predicción es baja (ej. 35%), el tomador de decisciones podría decidir no operar. Del mismo modo, si la señal es de "subida" pero el precio ya ha subido demasiado en poco tiempo, el gestor de riesgo podría abortar la operación para evitar el `slippage`.

---

## Backtest Realista
Para que la ventaja persista en un entorno real, es fundamental que el _backtest_  sea lo más realista posible. Debe incluir los siguientes factores:

- Costos de Transacción: Se simularían las comisiones del bróker, los _spreads_ y otras tarifas para obtener una rentabilidad neta más precisa.
- Latencia: Se modelaría un retraso realista entre la generación de la señal y la ejecución de la orden. En un entorno de trading de alta frecuencia, una latencia de milisegundos puede anular la ventaja de la estrategia.
- Slippage: Se incorporaría un modelo de _slippage_ que simule la diferencia entre el precio de la señal y el precio de ejecución real, especialmente en mercados volátiles o de baja liquidez.
- Gaps: La simulación debe ser capaz de manejar los "gaps" de precios que ocurren entre el cierre de un período y la apertura del siguiente, ya que pueden tener un impacto significativo en el _stop-loss_.

---

## Monitorización y feedback
Una vez en producción, el sistema debe ser monitoreado constantemente para asegurar que la ventaja no se degrade. Las métricas clave a seguir son:

- Drawdown Máximo: El indicador más importante de riesgo. Un drawdown creciente podría indicar que el modelo ya no es efectivo.
- Hit-Ratio Adaptativo: Un descenso sostenido en esta métrica señalaría que la señal del modelo está perdiendo su poder predictivo.
- Turnover: La frecuencia con la que el modelo realiza operaciones. Un turnover muy alto con ganancias bajas podría significar que los costos de transacción están erosionando la ventaja.
- Esperanza matemática: Es el promedio de ganancia o pérdida esperada por cada operación. La fórmula es `(probabilidad_ganadora * ganancia_promedio) - (probabilidad_perdedora * pérdida_promedio)`. Un valor positivo constante es el primer indicador de una estrategia potencialmente rentable. Un valor decreciente de esta métrica es una señal de que la ventaja del modelo podría estar desapareciendo.
- Curva de capital (Beneficio neto acumulado):  Un gráfico de esta curva a lo largo del tiempo es la forma más directa de visualizar el rendimiento total y la consistencia del modelo. Una curva con una pendiente positiva y constante es el objetivo.
- Ratio Riesgo Retorno promedio: Siempre buscamos que el valor que estamos dispuestos a perder sea mayor que el que estamos dispuestos a ganar, por lo que este ratio debería ser menor a 1 (por ejemplo, arriesgar 1 dólar para ganar 2, lo que da un ratio de 0.5).
- Sharpe Ratio: Es una medida de la rentabilidad ajustada por riesgo. Te dice cuánta rentabilidad extra obtienes por cada unidad de volatilidad que asumes. Un Sharpe Ratio más alto indica una estrategia más eficiente. Es una métrica superior a la rentabilidad bruta, ya que premia la consistencia y castiga la volatilidad excesiva.

