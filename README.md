# Conversor de Temperatura con Red Neuronal

Red neuronal que convierte temperaturas de Celsius a Fahrenheit usando TensorFlow.

## Características

- Red neuronal secuencial (30 capas, 20 neuronas/capa)
- Precisión hasta 100%
- Interfaz CLI interactiva
- Soporte para comas y puntos decimales

## Instalación

```bash
pip install tensorflow numpy
```

## Uso

```bash
python main.py
```

Ejemplo:
```
Ingrese una temperatura en Celsius: 25
El resultado es 77.00°F
```

## Configuración

Parámetros editables en `main.py`:

- `TEMPERATURA_MINIMA`: -273°C (default)
- `TEMPERATURA_MAXIMA`: 500°C (default)
- `NUM_CAPAS_OCULTAS`: 30 (default)
- `UNIDADES_POR_CAPA_OCULTA`: 20 (default)
- `TAZA_APRENDIZAJE`: 0.01 (default)
- `PRECISION_MINIMA`: 0.90 (default)

## Arquitectura

- Entrada: 1 neurona (Celsius)
- Capas ocultas: 30 × 20 neuronas (activación linear)
- Salida: 1 neurona (Fahrenheit)
- Optimizador: Adam
- Loss: Mean Squared Error

## Notas

- El modelo se entrena desde cero en cada ejecución
- Entrenamiento con early stopping automático
- Mensajes de TensorFlow suprimidos
