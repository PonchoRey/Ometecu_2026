from baseGenetico import AlgoritmoGenetico
from Ometecu import Ometecu
import random


TAM_POBLACION_NEURONAL = 6
NUM_GENERACIONES = 20000
array_poblacion_red = []
poblacion_con_fitness = []


for x in range(TAM_POBLACION_NEURONAL):
    red_neuronal = Ometecu()
    red_neuronal.set_config_red(capa_inicial=6, capa_intermedia=20, capa_final=5)
    valor_aleatoreo = [round(random.uniform(0.2, 0.8), 10) for elem in red_neuronal.get_memoria()]
    red_neuronal.set_memoria_genetico(valor_aleatoreo)
    array_poblacion_red.append(red_neuronal)

ag = AlgoritmoGenetico(tam_poblacion=len(array_poblacion_red), 
                       longitud_genoma=len(array_poblacion_red[0].get_memoria()) + 1, 
                       tasa_mutacion=0.01, num_generaciones=NUM_GENERACIONES)



fitnees = [0,0,0,0,0,10]
ag.set_poblacion_fitnees(fitnees, array_poblacion_red)
print(len(array_poblacion_red[3].get_memoria()), len(ag.get_poblacion()[3][:-1]))
# print("----------------")
# for x in ag.get_poblacion():
#     print(x)
# print("----------")
# ag.ejecutar() # EJECUTA EL PROCESO DONDE REALIZA LA MUTACION, EL CRECE Y AGREGA LOS HIJOS NUEVOS
# for x in ag.get_poblacion():
#     print(x)


