# Explicación Parte 1

## Justificación del activo

Coca-Cola (KO) fue seleccionada por las siguientes razones:

He seleccionado Coca-Cola (KO) como activo para esta prueba debido a su baja correlación con el mercado general, lo cual reduce el riesgo de que el comportamiento del precio esté condicionado por movimientos amplios del S&P 500 u otros sectores altamente volátiles. Además, KO no presenta tendencias prolongadas o altamente direccionales, lo que la convierte en un activo interesante para probar estrategias que no dependen exclusivamente de acción de precio o seguimiento de tendencia, permitiendo evaluar setups con mayor robustez y menor sesgo direccional.

---

## ⏱️ Temporalidades y Indicadores

La estrategia se compone de 3 niveles de análisis:

### 🕐 Diario (D1) – Contexto y Fuerza de la Tendencia

- **Indicador**: ADX (Average Directional Index)
- **Justificación**: Evalúa la fuerza de la tendencia sin importar su dirección.
- **Lógica**: Se considera válida una tendencia si ADX > 25.

---

### 🕐 Horario (H1) – Dirección del Momentum

- **Indicador**: Pendiente de regresión lineal sobre los últimos 20 cierres.
- **Justificación**: La pendiente indica el sesgo direccional reciente.
  - Positiva = tendencia alcista
  - Negativa = bajista
  - Cercana a 0 = lateralidad

---

### 🕐 15 Minutos (M15) – Confirmación de Entrada

- **Indicador**: Bandas de Bollinger
- **Justificación**: Captura extremos estadísticos en la acción del precio, asumiendo reversión a la media.
- **Lógica**: Se toma entrada cuando el precio rompe las bandas, alineado con dirección y fuerza de las temporalidades superiores.

---

## 🧮 Gestión de Riesgo

- **Stop Loss**: basado en el ATR (Average True Range)
  - Compra: SL = precio - 1 × ATR
  - Venta: SL = precio + 2 × ATR
- **Take Profit**: se puede usar un múltiplo fijo de ATR o cerrar cuando alguna de las condiciones (ADX o pendiente) deja de cumplirse.

---

## ⚠️ Consideraciones

- Las posiciones cortas (ventas) en acciones requieren derivados o instrumentos como CFDs, ya que no se pueden ejecutar directamente en el activo subyacente.
- El modelo aún es conceptual y debe validarse con backtesting antes de aplicarse en entornos reales igual que los posibles thresholds.

---




## 🗂️ Estructura del Repositorio

```bash
├── data/
│   └── historico_KO.csv       # Dataset descargado con yfinance
├── notebooks/
│   └── estrategia_KO.ipynb    # Implementación de la lógica y visualizaciones
├── utils/
│   └── indicadores.py         # Funciones auxiliares (ADX, Bollinger, pendiente, etc.)
├── README.md                  # Este documento
└── requirements.txt           # Dependencias del entorno