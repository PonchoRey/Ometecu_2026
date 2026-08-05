import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import random
import time
from ometecu.Ometecu import Ometecu

# --- 1. DATOS DE ENTRENAMIENTO (Conocimiento de un Botánico) ---

# Entradas: [Humedad del suelo (%), Color de hoja (0=Marrón, 1=Verde)]
# Salida:   [Diagnóstico (0.1=Sana, 0.9=Enferma)]

datos_brutos = [
    # Plantas Sanas 🌿
    {'entrada': [80, 0.9], 'salida': [0.1]}, # Mucha humedad, muy verde -> Sana
    {'entrada': [70, 0.8], 'salida': [0.1]}, # Buena humedad, bastante verde -> Sana
    {'entrada': [65, 1.0], 'salida': [0.1]}, # Humedad decente, perfectamente verde -> Sana

    # Plantas Enfermas 🍂
    {'entrada': [10, 0.2], 'salida': [0.9]}, # Suelo seco, hojas amarillas -> Enferma (sed)
    {'entrada': [25, 0.4], 'salida': [0.9]}, # Poca humedad, hojas pálidas -> Enferma
    {'entrada': [95, 0.3], 'salida': [0.9]}, # Demasiada humedad, hojas amarillas -> Enferma (raíz podrida)
]

# --- 2. PREPARAR DATOS PARA LA RED (Normalización) ---

entradas_normalizadas = []
salidas_deseadas = []

for dato in datos_brutos:
    humedad, color = dato['entrada']
    
    # Normalizamos los datos de entrada al rango 0.1 - 0.9
    humedad_norm = (humedad / 100.0) * 0.8 + 0.1  # El rango de humedad es 0-100
    color_norm = color * 0.8 + 0.1             # El rango de color ya es 0-1
    
    entradas_normalizadas.append([humedad_norm, color_norm])
    salidas_deseadas.append(dato['salida'])

# --- 3. CONFIGURACIÓN Y ENTRENAMIENTO ---

cerebro = Ometecu()
cerebro.inicio_synapsis("plantas")
# Arquitectura: 2 entradas, 3 neuronas ocultas, 1 salida (el diagnóstico)
cerebro.set_config_red(capa_inicial=2, capa_intermedia=30, capa_final=1)

epocas = 12500
print(f"🧠 Entrenando al botánico-robot durante {epocas} épocas...")

for i in range(epocas):
    # Elegimos un ejemplo al azar para entrenar
    idx_aleatorio = random.randint(0, len(entradas_normalizadas) - 1)
    
    cerebro.set_entradas(entradas_normalizadas[idx_aleatorio])
    cerebro.set_valor_aprender(salidas_deseadas[idx_aleatorio])
    cerebro.entrenamiento()
    #cerebro.estadisticaRed()
    #print("-" * 10)
    #time.sleep(0.5)

print("✅ Entrenamiento finalizado.")
print("-" * 30)

# --- 4. DIAGNÓSTICO DE NUEVAS PLANTAS ---

# Estas son plantas que la red nunca ha visto
plantas_a_diagnosticar = [
    {"nombre": "Planta A (Parece sana)", "datos": [75, 0.85]},
    {"nombre": "Planta B (Parece enferma)", "datos": [15, 0.3]},
    {"nombre": "Planta C (Dudosa)", "datos": [50, 0.6]},
    {"nombre": "Planta D (Dudosa)", "datos": [30, 0.2]},
]

print("🤖 Realizando nuevos diagnósticos...")
for planta in plantas_a_diagnosticar:
    humedad, color = planta["datos"]
    
    # Normalizamos los datos de la nueva planta
    humedad_norm = (humedad / 100.0) * 0.8 + 0.1
    color_norm = color * 0.8 + 0.1
    
    # Realizamos la predicción
    cerebro.set_entradas([humedad_norm, color_norm])
    prediccion = cerebro.prediccion()[0] # Solo hay una salida

    # Interpretamos el resultado binario
    diagnostico = "🍂 ENFERMA" if prediccion > 0.5 else "🌿 SANA"
    
    print(f"-> {planta['nombre']}: Humedad={humedad}%, Color={color} | Predicción={prediccion:.2f} | Diagnóstico: {diagnostico}")