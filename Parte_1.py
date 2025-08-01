# Test de Nivel Inicial en Python – Automatización de Datos Financieros:

# Paso 1: Importar las librerias pertinentes
import yfinance as yf # Extraccion de datos
import pandas as pd # Para manejos de dataframes
import numpy as np # Para operaciones


# Paso 2: Extracción de datos desde yfinance
ticker = "KO"
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


daily = download_and_clean_data(ticker, period="6mo", interval="1d")
hourly = download_and_clean_data(ticker, period="15d", interval="60m")
minutes_15= download_and_clean_data(ticker, period="5d", interval="15m")


# Paso 3: Calcular los indicadores

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




def calculate_bollinger_bands(df, period=20, num_std=2):
    """
    Calcula las Bandas de Bollinger.
    """
    df['bb_middle'] = df['Close'].rolling(window=period).mean()
    df['bb_std'] = df['Close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * num_std)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * num_std)
    return df[['bb_upper', 'bb_middle', 'bb_lower']]




def calculate_atr(df, period=14):
    """
    Calcula el Average True Range (ATR).
    """
    df['high-low'] = df['High'] - df['Low']
    df['high-prev_close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['low-prev_close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['true_range'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=period).mean()
    return df['atr']




# Paso 4: Agrupacion de datos
# Escogí la temporalidad más pequeña, en este caso 15 minutos
def agregar_diario(df_low_freq):
    """
    Agrupa un DataFrame de frecuencia menor en uno diario.
    Utiliza OHLCV estándar, VWAP y la desviación estándar de los retornos.
    """
    if df_low_freq is None or df_low_freq.empty:
        return None

    # Normalizar los nombres de las columnas
    if isinstance(df_low_freq.columns, pd.MultiIndex):
        df_low_freq.columns = df_low_freq.columns.droplevel(1)
    df_low_freq.columns = [col.capitalize() for col in df_low_freq.columns]

    # Calcular los retornos logarítmicos para cada periodo de 15 minutos
    # Usamos el método `pct_change()` para obtener los retornos porcentuales
    # y `np.log1p()` para los retornos logarítmicos
    df_low_freq['Intraday_Volatility'] = np.log1p(df_low_freq['Close'].pct_change())
    
    # Calcular los componentes para el VWAP
    df_low_freq['typical_price'] = (df_low_freq['High'] + df_low_freq['Low'] + df_low_freq['Close']) / 3
    df_low_freq['vwap_numerator'] = df_low_freq['typical_price'] * df_low_freq['Volume']

    # Asegurarse de que el índice es de tipo datetime
    df_low_freq.index = pd.to_datetime(df_low_freq.index)

    # Agrupar por día y aplicar los métodos de agregación
    daily_agg = df_low_freq.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'vwap_numerator': 'sum',
        'Intraday_Volatility': 'std'  
    })

    # Calcular el VWAP diario
    daily_agg['VWAP'] = daily_agg['vwap_numerator'] / daily_agg['Volume']
    
    # Limpiar columnas temporales y días sin operaciones
    daily_agg.drop(columns=['vwap_numerator'], inplace=True)
    daily_agg.dropna(inplace=True)

    return daily_agg


daily_from_minutes_15 = agregar_diario(minutes_15)
print(daily_from_minutes_15)
print(daily)


# Crea la columna en el dataset diario para el indicador adx
daily['ADX'] = calcular_adx(daily)
print(daily)

# Crea la columna en el dataset de datos por hora para el indicador que calcula la pendiente de la regresión
hourly['Slope'] = regression_slope(hourly)
print(hourly)

# Crea la columna en el dataset de datos de 15 minutos para las bollinger bands
minutes_15['BBands_Upper'], minutes_15['BBands_Middle'], minutes_15['BBands_Lower'] = calculate_bollinger_bands(minutes_15).values.T
print(minutes_15)

# Crea la columna en el dataset de datos de 15 minutos para el indicador ATR
minutes_15['ATR'] = calculate_atr(minutes_15)
print(minutes_15)

