from baseGenetico import AlgoritmoGenetico
from Ometecu import Ometecu
import random


TAM_POBLACION_NEURONAL = 6
NUM_GENERACIONES = 1
array_poblacion_red = []
poblacion_con_fitness = []


for x in range(TAM_POBLACION_NEURONAL):
    red_neuronal = Ometecu()
    red_neuronal.set_config_red(capa_inicial=1, capa_intermedia=4, capa_final=1)
    valor_aleatoreo = [round(random.uniform(0.2, 0.8), 10) for elem in red_neuronal.get_memoria()]
    red_neuronal.set_memoria_genetico(valor_aleatoreo)
    array_poblacion_red.append(red_neuronal)

ag = AlgoritmoGenetico(tam_poblacion=len(array_poblacion_red), 
                       longitud_genoma=len(array_poblacion_red[0].get_memoria()) + 1, 
                       tasa_mutacion=0.01, num_generaciones=NUM_GENERACIONES)


print("se genra la poblacion y se signa el fitnees por primera vez")
fitnees = [1,2,9,4,5,1]
array_poblacion_red = ag.redes_genetico(fitnees, array_poblacion_red)
ag.estadisticas_poblacion()
print("--------------------------------------------------------------")
print('\n')
#se realiza primera ejecucion 
ag.ejecutar()

print("se obtiene la poblacion despues de realizar la primera ejecucion")
nuevo_fitnees = []
print('\n')
for pobla in ag.get_poblacion():
    nuevo_fitnees.append(pobla[-1:][0])
    print(pobla)

print("--------------------------------------------------------------")
print('\n')

print("se genera la nueva poblacion con el fitnees")
print(nuevo_fitnees)
array_poblacion_red = ag.redes_genetico(nuevo_fitnees, array_poblacion_red)
ag.estadisticas_poblacion()

print("--------------------------------------------------------------")
print('\n')


# print('\n')
# for pobla in ag.get_poblacion():
#     print(pobla)


