import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

TEMPERATURA_MINIMA = -273
TEMPERATURA_MAXIMA = 500
PASO_TEMPERATURA = 2

CELSIUS = np.arange(TEMPERATURA_MINIMA, TEMPERATURA_MAXIMA + PASO_TEMPERATURA, PASO_TEMPERATURA, dtype=float)
FAHRENHEIT = (CELSIUS * 9/5) + 32

CELSIUS_MEAN = np.mean(CELSIUS)
CELSIUS_STD = np.std(CELSIUS)
CELSIUS_NORMALIZADO = (CELSIUS - CELSIUS_MEAN) / CELSIUS_STD

NUM_CAPAS_OCULTAS = 30
UNIDADES_POR_CAPA_OCULTA = 20
UNIDADES_CAPA_SALIDA = 1

TAZA_APRENDIZAJE = 0.01
EPOCAS_MAXIMAS = 10000
PRECISION_MINIMA = 0.90
PRECISION_MAXIMA = 1
PACiencia = 20

VALOR_PREDICCION = 30

capas = [tf.keras.layers.Input(shape=[1])]

for _ in range(NUM_CAPAS_OCULTAS):
    capas.append(tf.keras.layers.Dense(units=UNIDADES_POR_CAPA_OCULTA, activation='linear'))

capas.append(tf.keras.layers.Dense(units=UNIDADES_CAPA_SALIDA))

modelo = tf.keras.Sequential(capas)

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(TAZA_APRENDIZAJE),
    loss='mean_squared_error'
)

class AfinarMaximo(tf.keras.callbacks.Callback):
      
    def __init__(self, datos_entrada, datos_salida, precision_minima, precision_maxima, paciencia):
        super().__init__()
        self.datos_entrada = datos_entrada
        self.datos_salida = datos_salida
        self.precision_minima = precision_minima
        self.precision_maxima = precision_maxima
        self.paciencia = paciencia
        self.mejor_precision = 0.0
        self.epocas_sin_mejora = 0
        self.objetivo_minimo_alcanzado = False
    
    def calcular_precision(self):
      
        predicciones = self.model.predict(self.datos_entrada, verbose=0).flatten()
        
        if np.any(np.isnan(predicciones)) or np.any(np.isinf(predicciones)):
            return 0.0
        
        errores_absolutos = np.abs(predicciones - self.datos_salida)
        
        valores_absolutos = np.abs(self.datos_salida)
        divisor = np.where(valores_absolutos > 1e-6, valores_absolutos, np.max(valores_absolutos) + 1e-6)
        errores_porcentuales = errores_absolutos / divisor
        
        error_medio = np.mean(errores_porcentuales)
        return max(0, 1 - error_medio)
    
    def on_epoch_end(self, epoch, logs=None):
     
        precision_actual = self.calcular_precision()
        
        if not self.objetivo_minimo_alcanzado and precision_actual >= self.precision_minima:
            self.objetivo_minimo_alcanzado = True
            print(f"\n✓ Objetivo mínimo alcanzado: {precision_actual*100:.2f}%")
            print(f"  Continuando para afinar hasta el máximo posible...")
        
        if precision_actual > self.mejor_precision:
            self.mejor_precision = precision_actual
            self.epocas_sin_mejora = 0
            
            if precision_actual >= self.precision_maxima:
                print(f"\n✓ Precisión máxima alcanzada: {precision_actual*100:.4f}%")
                print(f"  Épocas entrenadas: {epoch + 1}")
                self.model.stop_training = True
        else:
            self.epocas_sin_mejora += 1
        
        if self.objetivo_minimo_alcanzado and self.epocas_sin_mejora >= self.paciencia:
            print(f"\n✓ Entrenamiento completado: {self.mejor_precision*100:.4f}% de precisión")
            print(f"  No se detectó mejora en {self.paciencia} épocas")
            print(f"  Épocas entrenadas: {epoch + 1}")
            self.model.stop_training = True

callback = AfinarMaximo(
    CELSIUS_NORMALIZADO, 
    FAHRENHEIT, 
    precision_minima=PRECISION_MINIMA,
    precision_maxima=PRECISION_MAXIMA,
    paciencia=PACiencia
)

print(f"Datos de entrenamiento: {len(CELSIUS)} ejemplos")
print(f"Rango: {TEMPERATURA_MINIMA}°C a {TEMPERATURA_MAXIMA}°C (paso: {PASO_TEMPERATURA}°C)")
print(f"\nArquitectura: {NUM_CAPAS_OCULTAS} capas ocultas con {UNIDADES_POR_CAPA_OCULTA} unidades cada una")
print("\nComenzando entrenamiento...")
print(f"Objetivo: Mínimo {PRECISION_MINIMA*100}%, continuará hasta alcanzar el máximo posible")
historial = modelo.fit(
    CELSIUS_NORMALIZADO, 
    FAHRENHEIT, 
    epochs=EPOCAS_MAXIMAS, 
    verbose=False,
    callbacks=[callback]
)
print("Modelo entrenado!")

def convertir_a_float(valor_str):
   
    valor_str = valor_str.strip().replace(',', '.')
    return float(valor_str)

while True:
    try:
        entrada = input("Ingrese una temperatura en Celsius: ")
        temperatura_celsius = convertir_a_float(entrada)
        temperatura_normalizada = (temperatura_celsius - CELSIUS_MEAN) / CELSIUS_STD
        resultado = modelo.predict(np.array([temperatura_normalizada]), verbose=0)
        valor_prediccion = float(resultado[0][0])
        
        if np.isnan(valor_prediccion) or np.isinf(valor_prediccion):
            print("Error: El modelo produjo un valor inválido. Intente con otro valor.")
        else:
            print(f"El resultado es {valor_prediccion:.2f}°F")
    except ValueError:
        print("Error: Por favor ingrese un valor numérico válido (ejemplo: 25 o 25,5)")
    except Exception as e:
        print(f"Error inesperado: {e}")
