import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
"""
=============================================================================
TUTORIAL 02: ALGORITMO GENÉTICO CON OMETECU (Versión Mejorada)
=============================================================================
Este archivo demuestra cómo usar Ometecu en combinación con AlgoritmoGenetico.

El Aprendizaje Evolutivo (Genético) se usa cuando NO tenemos las respuestas 
exactas para entrenar a la red. En su lugar, evaluamos su comportamiento 
en un entorno y premiamos las acciones beneficiosas (Fitness).

EL PROBLEMA: "La Mascota Virtual Autónoma"
Tenemos una población de 16 mascotas (redes neuronales). 
Cada una recibe 2 datos del entorno (Entradas):
 1. Nivel de Hambre (0.1 = Lleno, 0.9 = Hambriento)
 2. Nivel de Sueño  (0.1 = Fresco, 0.9 = Agotado)

La red debe activar 2 posibles acciones (Salidas):
 1. Impulso de Comer
 2. Impulso de Dormir

SISTEMA DE PUNTUACIÓN (Fitness):
No las entrenaremos con datos correctos. Simplemente pondremos a cada 
mascota en escenarios límite. Si toman decisiones lógicas para sobrevivir 
(comer cuando hay hambre, dormir cuando hay sueño) ganarán puntos (Fitness).
Las que sobrevivan, mezclarán sus "genes" (pesos) para la siguiente generación.
"""

import random
from ometecu.Ometecu import Ometecu
from ometecu.baseGenetico import AlgoritmoGenetico

def ejecutar_mascotas_virtuales():
    print("Iniciando Tutorial 02: Supervivencia Genética de Mascotas\n")

    TAMANO_POBLACION = 16
    GENERACIONES = 40
    
    # 1. INSTANCIAR ALGORITMO GENÉTICO
    # Tasa de mutación del 15% para fomentar diversidad en las decisiones
    ag = AlgoritmoGenetico(tam_poblacion=TAMANO_POBLACION, tasa_mutacion=0.15, num_generaciones=1)

    # 2. CREAR POBLACIÓN INICIAL (Generación 0, totalmente ignorantes)
    mascotas = []
    for _ in range(TAMANO_POBLACION):
        cerebro_mascota = Ometecu()
        cerebro_mascota.inicio_synapsis("mascota")
        # 2 entradas (Hambre, Sueño), 4 neuronas ocultas para procesar, 2 salidas (Comer, Dormir)
        cerebro_mascota.set_config_red(capa_inicial=2, capa_intermedia=4, capa_final=2)
        cerebro_mascota.funcionActivacion('r', 'r', 's') # Usamos sigmoide al final para obtener valores entre 0 y 1
        
        # Asignamos ADN (pesos) aleatorio
        genes_aleatorios = [random.uniform(-1.0, 1.0) for _ in cerebro_mascota.get_memoria()]
        cerebro_mascota.set_memoria_genetico(genes_aleatorios)
        
        mascotas.append(cerebro_mascota)

    print(f"🌍 Se ha creado un mundo con {TAMANO_POBLACION} mascotas de inteligencia aleatoria.")

    # Definimos 4 escenarios a los que se enfrentarán todas las mascotas
    escenarios_prueba = [
        {"nombre": "A punto de morir de hambre", "hambre": 0.9, "sueno": 0.1},
        {"nombre": "Agotamiento extremo", "hambre": 0.1, "sueno": 0.9},
        {"nombre": "Hambre y algo de cansancio", "hambre": 0.7, "sueno": 0.4},
        {"nombre": "Totalmente relajado", "hambre": 0.1, "sueno": 0.1}
    ]

    # 3. CICLO DE EVOLUCIÓN
    for generacion in range(GENERACIONES):
        calificaciones_supervivencia = []

        # Ponemos a prueba a cada mascota individualmente
        for mascota in mascotas:
            puntos_fitness = 1.0 # Empezamos con 1 para evitar que sea 0 absoluto

            # Probamos a la mascota en todos los escenarios
            for escenario in escenarios_prueba:
                # La mascota "percibe" el entorno
                mascota.set_entradas([escenario["hambre"], escenario["sueno"]])
                prediccion = mascota.prediccion()
                
                # Obtenemos sus decisiones
                impulso_comer = prediccion[0]
                impulso_dormir = prediccion[1]

                # --- REGLAS DE SUPERVIVENCIA (El entorno juzga a la IA) ---
                
                # REGLA 1: Si hay mucha hambre, debería querer comer más que dormir
                if escenario["hambre"] > escenario["sueno"]:
                    if impulso_comer > impulso_dormir:
                        puntos_fitness += 10.0  # ¡Buena decisión!
                    else:
                        puntos_fitness -= 5.0   # ¡Mala decisión, podría morir!
                        
                # REGLA 2: Si hay mucho sueño, debería querer dormir
                elif escenario["sueno"] > escenario["hambre"]:
                    if impulso_dormir > impulso_comer:
                        puntos_fitness += 10.0
                    else:
                        puntos_fitness -= 5.0
                        
                # REGLA 3: Eficiencia (Castigar impulsos innecesarios)
                # Si no tiene hambre, el impulso de comer debería ser bajo
                if escenario["hambre"] < 0.3:
                    puntos_fitness += (1.0 - impulso_comer) * 5.0

            # Guardamos la calificación final de esta mascota
            # Aseguramos un piso de 1.0 para el algoritmo de ruleta
            calificaciones_supervivencia.append(max(1.0, puntos_fitness))

        mejor_puntuacion = max(calificaciones_supervivencia)
        
        # Mostrar progreso cada 5 generaciones para no saturar la consola
        if generacion == 0 or (generacion + 1) % 5 == 0:
            print(f"Generación {generacion + 1:02d} | Máxima Supervivencia (Fitness): {mejor_puntuacion:.2f} pts")

        # 4. APLICAR SELECCIÓN NATURAL
        # Enviamos las calificaciones; el algoritmo cruza a los mejores y mata a los peores
        if generacion < GENERACIONES - 1:
            mascotas = ag.redes_genetico(calificaciones_supervivencia, mascotas)

    # -----------------------------------------------------------------------
    # PASO 5: EXAMEN FINAL DEL MEJOR ESPECIMEN EVOLUCIONADO
    # -----------------------------------------------------------------------
    print("\n=======================================================")
    print("🏆 PRUEBA FINAL: COMPORTAMIENTO DEL MEJOR INDIVIDUO")
    print("=======================================================")
    
    # Tomamos al líder de la última generación (El elitismo lo pone en la posición 0)
    mejor_mascota = mascotas[0]

    for escenario in escenarios_prueba:
        mejor_mascota.set_entradas([escenario["hambre"], escenario["sueno"]])
        resultados = mejor_mascota.prediccion()
        
        comer = resultados[0]
        dormir = resultados[1]
        
        # Convertimos los números fríos en acciones legibles
        accion = "🍔 COMER" if comer > dormir else "💤 DORMIR"
        if comer < 0.3 and dormir < 0.3:
            accion = "🧘 DESCANSAR" # Ambas son bajas

        print(f"\nEntorno: {escenario['nombre'].upper()}")
        print(f"   Sensores: Hambre={escenario['hambre']} | Sueño={escenario['sueno']}")
        print(f"   Decisión interna: Comer ({comer:.2f}) vs Dormir ({dormir:.2f})")
        print(f"   -> ACCIÓN TOMADA: {accion}")

if __name__ == "__main__":
    ejecutar_mascotas_virtuales()
