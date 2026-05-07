from baseGenetico import AlgoritmoGenetico
from Ometecu import Ometecu


# CONFIGURACION INICIAL
TAM_POBLACION_NEURONAL = 10
NUM_GENERACIONES = 300


# CONFIGURACION DE ALGORITMO.
# -------------------------------------------------------------------------------------------
array_poblacion_red = []
poblacion_con_fitness = []
for x in range(TAM_POBLACION_NEURONAL):
    red_neuronal = Ometecu()
    red_neuronal.set_config_red(capa_inicial=2, capa_intermedia=3, capa_final=1)
    array_poblacion_red.append(red_neuronal)

ag = AlgoritmoGenetico(tam_poblacion=len(array_poblacion_red), 
                       longitud_genoma=len(array_poblacion_red[0].get_memoria()) + 1, 
                       tasa_mutacion=0.01, num_generaciones=NUM_GENERACIONES)

# SE AGREGA EL VALOR DE FITNESS INCIAL "0" AL CONJUNTO DE GENES DE CADA NEURONA 
# PARA TENER LA REFERENCIA Y MODEIFIAR EL ARRAY SI EL FITNESS CAMBIA
# LA LIBRERIA DE "AlgoritmoGenetico" YA ESTA PREPARADA PARA IGNORAR EL CAMPO. 
for red in array_poblacion_red:
    genoma = red.get_memoria()
    genoma_con_fitness = list(genoma) + [0] #FITNESS INICIAL
    poblacion_con_fitness.append(genoma_con_fitness)
    ag.set_poblacion(poblacion_con_fitness)
# -------------------------------------------------------------------------------------------

# CON ESTE BLOQUE SE GENRA EL FITNESS PARA LAS NEURONAS, TOMANDO COMO REFENCIA EL PESO,
# MIENTRAS MAS PESO MAS POSIBILIDAD DE REPRODUCCION
aux_fitness = ag.get_poblacion()
aux_fitness[0][-1] = 10
aux_fitness[4][-1] = 12
aux_fitness[-1][-1] = 15
ag.set_poblacion(aux_fitness)


ag.ejecutar() # EJECUTA EL PROCESO DONDE REALIZA LA MUTACION, EL CRECE Y AGREGA LOS HIJOS NUEVOS
for x in ag.get_poblacion():
    print(x)


