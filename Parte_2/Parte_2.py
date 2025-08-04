# Taller práctico (MVP + código):

import yfinance as yf # Extraccion de datos
import pandas as pd # Para manejos de dataframes
import numpy as np # Para operaciones
from xgboost import XGBClassifier, plot_importance # Feature importance
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # Reportes
from imblearn.over_sampling import RandomOverSampler # Desbalance de clases
import matplotlib.pyplot as plt # Gráficos

# He elegido el modelo de clasificacion que consiste en predecir la direccion del precio en la siguiente hora

# Vamos a usar algunas de las funciones que previamente habiamos creado
def download_and_clean_data(ticker, period, interval):
    """
    Descarga datos de Yahoo Finance y normaliza los nombres de las columnas.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            print(f"No se encontraron datos para {ticker} en el intervalo {interval}")
            return None
        
        # Si el DataFrame tiene un MultiIndex (que suele ser el formato de yf), lo aplanamos
        if isinstance(df.columns, pd.MultiIndex):
            # Eliminamos el nivel de 'Ticker'
            df.columns = df.columns.droplevel(1)
        
        # Renombramos las columnas si fuera necesario, aunque yfinance ya usa nombres estándar
        df.columns = [col.capitalize() for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"Error al descargar datos para {ticker}: {e}")
        return None


ticker = "KO"
d1 = download_and_clean_data(ticker, period="2y", interval="1d")
h1 = download_and_clean_data(ticker, period="2y", interval="60m") 

# Para convertir el indice (Datetime) a UTC
def normalize_timezone(df):
    if df is not None and not df.index.tz: # Si el índice no tiene zona horaria
        df.index = df.index.tz_localize('UTC')
    return df

d1 = normalize_timezone(d1)
h1 = normalize_timezone(h1)


# Creacion de los indicadores a usar

# En D1 vamos a usar ADX y la pendente de una EMA de 20 periodos
def calcular_adx(df, period=14):
    """
    Calcula el Average Directional Index (ADX) en un DataFrame de precios.
    """
    # High-Low
    df['high_low'] = df['High'] - df['Low']
    # High-Previous_Close
    df['high_close'] = np.abs(df['High'] - df['Close'].shift(1))
    # Low-Previous_Close
    df['low_close'] = np.abs(df['Low'] - df['Close'].shift(1))
    
    # True Range (TR)
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Directional Movement (DM)
    df['+dm'] = np.where((df['High'] > df['High'].shift(1)) & (df['High'] - df['High'].shift(1) > df['Low'].shift(1) - df['Low']), 
                         df['High'] - df['High'].shift(1), 0)
    df['-dm'] = np.where((df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) - df['Low'] > df['High'] - df['High'].shift(1)), 
                         df['Low'].shift(1) - df['Low'], 0)
    
    # Exponential Moving Averages of DM and TR
    df['+di'] = df['+dm'].ewm(alpha=1/period, min_periods=period).mean() / df['tr'].ewm(alpha=1/period, min_periods=period).mean() * 100
    df['-di'] = df['-dm'].ewm(alpha=1/period, min_periods=period).mean() / df['tr'].ewm(alpha=1/period, min_periods=period).mean() * 100

    df['dx'] = np.abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']) * 100
    df['adx'] = df['dx'].ewm(alpha=1/period, min_periods=period).mean()
    
    return df['adx']

d1['ADX'] = calcular_adx(d1)

d1['EMA20'] = d1['Close'].ewm(span=20).mean()
d1['EMA20_Slope'] = np.gradient(d1['EMA20']) # Pendiente de la EMA

# En H1 vamos a usar transformaciones de precio como:
# - Retornos logaritmicos del precio
# - Volatilidad como desviacion estandar de los retornos logaritmicos 
# - Una proxy de sesgo del precio
# - Tamaño del cuerpo de las velas 
# Y algunos indicadores como la pendiente de la regresion lineal, MACD y RSI
h1['Log_Ret'] = np.log(h1['Close'] / h1['Close'].shift())
h1['Volatility'] = h1['Log_Ret'].rolling(20).std()
h1['Skewness'] = h1['Log_Ret'].rolling(50).skew()
h1['candle_size'] = abs(h1['Close'] - h1['Open']) / (h1['High'] - h1['Low'])

def regression_slope(df, period=20):
    """
    Calcula la pendiente de la regresión lineal para el precio de cierre.
    """
    def get_slope(x):
        y = np.arange(len(x))
        # np.polyfit devuelve los coeficientes de la regresión (pendiente, intercepto)
        slope, _ = np.polyfit(y, x, 1)
        return slope

    # El rolling aplica la función get_slope en una ventana deslizante
    df['lin_reg_slope'] = df['Close'].rolling(window=period).apply(get_slope, raw=True)
    return df['lin_reg_slope']
h1['Reg_Slope'] = regression_slope(h1)


def calculate_rsi(data, window=14):
    """Calcula el Relative Strength Index (RSI)."""
    delta = data.diff()
    up = delta.copy()
    down = delta.copy()
    
    up[up < 0] = 0
    down[down > 0] = 0
    
    # Usamos EWM para calcular la media móvil exponencial,
    # que es el método estándar para el RSI.
    avg_gain = up.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = down.ewm(com=window - 1, min_periods=window).mean().abs()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
h1['RSI'] = calculate_rsi(h1['Close'], window=14)


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calcula el MACD, la línea de Señal y el Histograma."""
    # Calcula la EMA rápida y lenta
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calcula la línea MACD
    macd_line = ema_fast - ema_slow
    
    # Calcula la línea de Señal (EMA de la línea MACD)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calcula el Histograma del MACD
    macd_histogram = macd_line - signal_line
    
    # Devuelve un DataFrame con los tres componentes
    macd_df = pd.DataFrame({
        'MACD': macd_line,
        'MACD_SIG': signal_line,
        'MACD_HIST': macd_histogram
    })
    
    return macd_df
macd_features = calculate_macd(h1['Close'])
h1 = pd.concat([h1, macd_features], axis=1)

# Alienacion de los 2 dataframes
d1_features = d1[['ADX', 'EMA20_Slope']]
d1_features_shifted = d1_features.shift(1) #Desfasamos los valores de la temporalidad diaria para evitar un look-ahead bias

# Unimos los features de D1 al DataFrame de H1 usando merge_asof.
df_final = pd.merge_asof(
    h1,
    d1_features_shifted,
    left_index=True,
    right_index=True,
    direction='backward' 
)

# Limpieza y preparación final del DataFrame
df_final.dropna(inplace=True)

# Verificamos el DataFrame final
print(df_final.head())

# Crear la Variable Objetivo (output)
atr = df_final['High'] - df_final['Low']
umbral = 0.3 * atr.rolling(5).mean()
cambio_de_precio = df_final['Close'].shift(-1) - df_final['Close']

# Paso 2: Define las condiciones y los valores para cada clase
condiciones = [
    (cambio_de_precio > umbral),        # Subida significativa
    (cambio_de_precio < -umbral),       # Bajada significativa
    (abs(cambio_de_precio) <= umbral)   # Cambio insignificante
]

valores = [1, 0, 2] # 1: Subida, 0: Bajada, 2: Sin cambio
df_final['Target'] = np.select(condiciones, valores, default=2).astype(int)
df_final.dropna(inplace=True)

print("\nDistribución de la nueva variable Target:")
print(df_final['Target'].value_counts())
print("-" * 40)

# Preparar los datos para el modelo
features = df_final[['ADX', 'EMA20_Slope', 'Log_Ret', 'Volatility', 'Skewness', 'candle_size', 'Reg_Slope', 'High', 'Low', 'Open', 'Close', 'RSI', 'MACD', 'MACD_SIG' ,'MACD_HIST']]
target = df_final['Target']


# Dividir los datos entre entrenamiento y prueba
split_point = int(len(features) * 0.85)
features_train = features.iloc[:split_point]
target_train = target.iloc[:split_point]
features_test = features.iloc[split_point:]
target_test = target.iloc[split_point:]

print(f"Total de datos: {len(features)}")
print(f"Datos de entrenamiento: {len(features_train)}")
print(f"Datos de prueba: {len(target_test)}")

#Oversampling para balancear las clases
oversampler = RandomOverSampler(random_state=42)
# Aplicamos el oversampling SOLO en el conjunto de entrenamiento
features_train_resampled, target_train_resampled = oversampler.fit_resample(features_train, target_train)

eval_set = [(features_train, target_train)]
if len(features_test) > 0:
    eval_set.append((features_test, target_test))

# Entrenar el modelo XGBoost
num_classes = len(target_train.unique())
print(f"Número de clases detectadas: {num_classes}")

model = XGBClassifier(
    objective='multi:softprob',      
    num_class=num_classes,           
    max_depth=6,
    learning_rate=0.005,
    n_estimators=1000,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=1.0,
    gamma=0.1,
    eval_metric=['mlogloss', 'merror'],
    use_label_encoder=False,  
    early_stopping_rounds=50,
    random_state=42

)

# Compilación del modelo
history = model.fit(
    features_train_resampled,  # <-- Usamos el conjunto balanceado aquí
    target_train_resampled,
    eval_set=eval_set,
    verbose=100
)

# Prediccion
y_pred = model.predict(features_test)

# Gráfico de la funcion de pérdida durante el entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(model.evals_result()['validation_0']['mlogloss'], label='Pérdida de Entrenamiento')
plt.plot(model.evals_result()['validation_1']['mlogloss'], label='Pérdida de Prueba')
plt.title('Loss durante Entrenamiento')
plt.xlabel('Iteraciones')
plt.ylabel('Log Loss')
plt.legend()
plt.grid()
plt.savefig('loss_curve.png')



# Importancia de características (Permutation Feature Importance)
fig, ax = plt.subplots(figsize=(12, 8))
plot_importance(
    model, 
    ax=ax, 
    importance_type='gain',  # Ganancia promedio por división
    max_num_features=15,
    title='Feature Importance por Ganancia Promedio',
    xlabel='Ganancia Promedio'
)
plt.tight_layout()
plt.savefig('feature_importance.png')

# Informe de clasificación y matriz de confusión
print("\n--- Informe de Clasificación para el modelo de 3 clases ---")
print(classification_report(target_test, y_pred, zero_division=0))
cm = confusion_matrix(target_test, y_pred)
print("\nMatriz de Confusión:\n", cm)

# CORRECCIÓN: Actualizamos las etiquetas para incluir la tercera clase
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Baja (0)", "Sube (1)", "Sin Cambio (2)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión (3 Clases)")
plt.savefig('Matriz de Confusión.png')



#    Seleccionamos la última fila del DataFrame de características.
#    Utilizamos .iloc[-1:] para asegurarnos de que sea un DataFrame
#    y mantenga el formato correcto para el modelo.
last_features = features.iloc[-1:].copy()


# El modelo predice la dirección de la siguiente hora
final_prediction = model.predict(last_features)[0]
final_proba = model.predict_proba(last_features)[0]

# 3. Formatear la salida para una interpretación clara
print("\n--- Predicción para el Siguiente Periodo ---")
print(f"Momento de la predicción: {last_features.index[0]}")
print(f"Características usadas:\n{last_features.to_string()}")
print("-" * 40)


if final_prediction == 1:
    print("La predicción del modelo es: SUBIDA significativa")
    print(f"Confianza de la predicción: {final_proba[1]:.2%}")
elif final_prediction == 0:
    print("La predicción del modelo es: BAJADA significativa")
    print(f"Confianza de la predicción: {final_proba[0]:.2%}")
else: # final_prediction == 2
    print("La predicción del modelo es: CAMBIO INSIGNIFICANTE")
    print(f"Confianza de la predicción: {final_proba[2]:.2%}")




