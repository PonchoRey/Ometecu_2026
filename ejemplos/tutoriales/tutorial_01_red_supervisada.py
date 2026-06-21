import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
=============================================================================
TUTORIAL 01: RED NEURONAL SUPERVISADA CON OMETECU
=============================================================================
Este archivo es un ejemplo documentado de cómo usar la red neuronal Ometecu 
para resolver un problema básico usando "Aprendizaje Supervisado". 

El aprendizaje supervisado significa que nosotros conocemos la respuesta 
correcta y entrenamos a la red mostrándole ejemplos y corrigiéndola.

EL PROBLEMA:
Vamos a enseñar a la red a comportarse como una compuerta lógica "OR" (O).
Si alguna de las entradas es Alta (Verdadero), la salida debe ser Alta.
Solo si ambas son Bajas, la salida será Baja.

Nota sobre los datos: 
Las redes neuronales funcionan mejor con números decimales entre 0 y 1. 
Usaremos 0.1 para representar "Falso/Bajo" y 0.9 para "Verdadero/Alto".
"""

import random
from ometecu.Ometecu import Ometecu

def ejecutar_ejemplo_supervisado():
    print("Iniciando Tutorial: Red Neuronal Supervisada (Compuerta OR)\n")

    # -----------------------------------------------------------------------
    # PASO 1: PREPARAR LOS DATOS DE ENTRENAMIENTO
    # -----------------------------------------------------------------------
    # Cada diccionario tiene una 'entrada' (2 valores) y una 'salida_esperada' (1 valor).
    datos_entrenamiento = [
        {'entrada': [0.1, 0.1], 'salida_esperada': [0.1]}, # Falso OR Falso = Falso
        {'entrada': [0.1, 0.9], 'salida_esperada': [0.9]}, # Falso OR Verdadero = Verdadero
        {'entrada': [0.9, 0.1], 'salida_esperada': [0.9]}, # Verdadero OR Falso = Verdadero
        {'entrada': [0.9, 0.9], 'salida_esperada': [0.9]}, # Verdadero OR Verdadero = Verdadero
    ]

    # -----------------------------------------------------------------------
    # PASO 2: INSTANCIAR Y CONFIGURAR LA RED NEURONAL (CEREBRO)
    # -----------------------------------------------------------------------
    cerebro = Ometecu()
    cerebro.inicio_synapsis("tutorial_01")
    # Configuramos la arquitectura de las capas:
    # capa_inicial: 2 neuronas (porque tenemos 2 datos de entrada)
    # capa_intermedia: 5 neuronas (capa oculta para procesar la lógica)
    # capa_final: 1 neurona (porque solo queremos 1 resultado de salida)
    
    cerebro.set_config_red(capa_inicial=2, capa_intermedia=5, capa_final=1)

    # -----------------------------------------------------------------------
    # PASO 3: ENTRENAMIENTO DE LA RED (BACKPROPAGATION)
    # -----------------------------------------------------------------------
    epocas = 5000  # Número de veces que la red estudiará los datos para aprender
    print(f"Entrenando la red durante {epocas} iteraciones. Por favor, espera...")

    for i in range(epocas):
        # Elegimos un ejemplo al azar de nuestros datos de entrenamiento
        dato = random.choice(datos_entrenamiento)
        
        # Le mostramos la entrada a la red
        cerebro.set_entradas(dato['entrada'])
        # Le decimos cuál era la respuesta correcta para esa entrada
        cerebro.set_valor_aprender(dato['salida_esperada'])
        
        # Ejecutamos el ciclo de entrenamiento (hace la predicción y ajusta sus errores)
        cerebro.entrenamiento()

    print("Entrenamiento finalizado.\n")

    # -----------------------------------------------------------------------
    # PASO 4: PONER A PRUEBA LA RED (INFERENCIA / PREDICCIÓN)
    # -----------------------------------------------------------------------
    print("Resultados de las pruebas:")
    
    for dato in datos_entrenamiento:
        entradas_prueba = dato['entrada']
        
        # Para hacer una predicción real, solo pasamos las entradas (no la respuesta)
        cerebro.set_entradas(entradas_prueba)
        
        # Llamamos a prediccion() y obtenemos el resultado. 
        # Como devuelve una lista de salidas, tomamos el primer elemento [0]
        resultado = cerebro.prediccion()[0] 
        
        # Interpretamos visualmente para el usuario:
        # Si el resultado es mayor a 0.5, se considera "Alto/Verdadero"
        interpretacion = "VERDADERO" if resultado > 0.5 else "FALSO"

        print(f"Entrada: {entradas_prueba} -> Salida de red: {resultado:.4f} ({interpretacion})")

if __name__ == "__main__":
    ejecutar_ejemplo_supervisado()
