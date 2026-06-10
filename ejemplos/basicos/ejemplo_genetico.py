import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ometecu.baseGenetico import AlgoritmoGenetico
from ometecu.Ometecu import Ometecu
import random

TAM_POBLACION_NEURONAL = 6
NUM_GENERACIONES = 1
array_poblacion_red = []

for x in range(TAM_POBLACION_NEURONAL):
    red_neuronal = Ometecu()
    red_neuronal.set_config_red(capa_inicial=1, capa_intermedia=3, capa_final=1)
    
    valor_aleatorio = [random.uniform(0.2, 0.8) for _ in red_neuronal.get_memoria()]
    red_neuronal.set_memoria_genetico(valor_aleatorio)
    array_poblacion_red.append(red_neuronal)


ag = AlgoritmoGenetico(
    tam_poblacion=len(array_poblacion_red), 
    tasa_mutacion=0.01, 
    num_generaciones=NUM_GENERACIONES
)

array_poblacion_red[0].set_memoria_genetico(ag.obtener_memoria())

print("Se genera la poblacion y se asigna el fitness por primera vez")
fitness_inicial = [1, 2, 9, 4, 5, 1]

array_poblacion_red = ag.redes_genetico(fitness_inicial, array_poblacion_red)
ag.estadisticas_poblacion()


ag.guardar_memoria()
