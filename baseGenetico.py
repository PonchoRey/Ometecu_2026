import random
import json

class AlgoritmoGenetico:
    
    def __init__(self, tam_poblacion, tasa_mutacion, num_generaciones):
        # Inicialización de los hiperparámetros del algoritmo genético
        self.tam_poblacion = tam_poblacion 
        self.tasa_mutacion = tasa_mutacion
        self.num_generaciones = num_generaciones
        # Lista que contendrá a los individuos de la generación actual
        self.poblacion = []  # Almacenará diccionarios estructurales: {'genes': [...], 'fitness': X}

    def redes_genetico(self, array_fitness, array_redes):
        # Limpia o reinicia la población interna para procesar la nueva iteración evolutiva
        self.poblacion = []
        
        # Corrección: Evitar división por cero o pesos nulos si el fitness es 0
        # Reemplaza cualquier valor de fitness igual a 0 por 1 para asegurar estabilidad en la selección
        array_fitness = [1 if valor == 0 else valor for valor in array_fitness]
        
        # CORRECCIÓN DE BUG: Separamos los genes del fitness usando un diccionario.
        # De esta forma el fitness no se mezcla dentro de las listas de genes en el cruce.
        # Recorre las redes externas, extrae sus pesos/memoria y los estructura junto a su respectivo fitness
        for index, red in enumerate(array_redes):
            self.poblacion.append({
                'genes': list(red.get_memoria()),
                'fitness': array_fitness[index]
            })

        # Ejecuta el ciclo evolutivo (internamente procesa las generaciones)
        self.ejecutar()
        
        # Devolvemos los genes optimizados de vuelta a los objetos de tus redes neuronales
        # Sobrescribe la memoria de los objetos originales con los nuevos genes ya evolucionados
        for index, individuo in enumerate(self.poblacion):
            array_redes[index].set_memoria_genetico(individuo['genes'])
        
        return array_redes

    def estadisticas_poblacion(self):
        """Imprime la población en el formato visual original (genes + [fitness])"""
        print("Poblacion actual -----------------------------------------------")
        # Itera sobre cada individuo y genera una visualización unificada en consola
        for ind in self.poblacion:
            # Concatenamos temporalmente para mantener la compatibilidad con tus prints anteriores
            # Muestra la lista de genes seguida del valor de fitness como el último elemento
            print(ind['genes'] + [ind['fitness']])
        print("Poblacion actual -----------------------------------------------")

    def seleccion(self):
        # Extrae de forma aislada los puntajes de aptitud (fitness) de toda la población
        fitness_scores = [ind['fitness'] for ind in self.poblacion]
        # Aplica selección por ruleta (muestreo estocástico con reemplazo) donde 
        # los individuos con mayor fitness tienen proporcionalmente más probabilidad de ser elegidos
        return random.choices(self.poblacion, weights=fitness_scores, k=self.tam_poblacion)

    def cruce(self, parent1, parent2):
        # Realiza el cruce de un solo punto utilizando únicamente la lista de genes.
        # Define un punto de corte aleatorio dentro de la longitud de la cadena de genes
        punto_cruce = random.randint(1, len(parent1['genes']))
        
        # Creamos listas nuevas combinando los segmentos de los padres
        # El hijo 1 toma el inicio del padre 1 y el final del padre 2
        hijo1_genes = parent1['genes'][:punto_cruce] + parent2['genes'][punto_cruce:]
        # El hijo 2 toma el inicio del padre 2 y el final del padre 1
        hijo2_genes = parent2['genes'][:punto_cruce] + parent1['genes'][punto_cruce:]
        
        # Inicializamos los hijos con un fitness base (será reevaluado en la siguiente iteración de la red)
        # Se retorna una tupla con los dos nuevos diccionarios mutables de los descendientes
        return {'genes': hijo1_genes, 'fitness': 1}, {'genes': hijo2_genes, 'fitness': 1}

    def mutacion(self, individuo):
        # Obtiene la referencia a la lista de genes del individuo a evaluar
        genes = individuo['genes']
        # Itera por cada uno de los genes (pesos) de forma secuencial
        for i in range(len(genes)):
            # Determina de forma aleatoria si el gen actual debe mutar según la tasa configurada
            if random.random() < self.tasa_mutacion:
                # Si se cumple la condición, asigna un nuevo peso aleatorio continuo entre -0.9 y 0.9
                genes[i] = random.uniform(-0.9, 0.9)
        return individuo

    def ejecutar(self):
        
        # Bucle principal que controla la transición entre generaciones
        for _ in range(self.num_generaciones):
            
            # --- OPTIMIZACIÓN Y CORRECCIÓN: ELITISMO REAL ---
            # Ordenamos toda la población por fitness de mayor a menor de forma explícita.
            poblacion_ordenada = sorted(self.poblacion, key=lambda x: x['fitness'], reverse=True)
            
            # Inicializa la lista que contendrá a la nueva generación
            nueva_poblacion = []
            # Extraemos copias profundas de los 2 mejores absolutos (los elites de verdad).
            # Usamos list() para clonar los genes en memoria y que no se muten accidentalmente después.
            # Esto garantiza que los mejores diseños pasen intactos sin sufrir alteraciones por cruce o mutación
            nueva_poblacion.append({'genes': list(poblacion_ordenada[0]['genes']), 'fitness': poblacion_ordenada[0]['fitness']})
            nueva_poblacion.append({'genes': list(poblacion_ordenada[1]['genes']), 'fitness': poblacion_ordenada[1]['fitness']})
            # ------------------------------------------------
            
            # Selección por ruleta para rellenar el resto de la población
            # Obtiene un pool de candidatos basados en su rendimiento actual
            seleccionados = self.seleccion()

            # Ciclo de reproducción para cubrir los espacios restantes (tam_poblacion - 2)
            # Avanza de dos en dos para emparejar a los padres seleccionados
            for i in range(0, self.tam_poblacion - 2, 2):
                # Extrae de la selección la pareja de padres correspondiente
                padre1, padre2 = seleccionados[i], seleccionados[i+1]
                # Ejecuta la recombinación de material genético para procrear dos descendientes
                hijo1, hijo2 = self.cruce(padre1, padre2)
                
                # Mutamos a los hijos y los añadimos a la nueva generación
                # Aplica el operador de mutación probabilística a cada hijo antes de guardarlo
                nueva_poblacion.append(self.mutacion(hijo1))
                nueva_poblacion.append(self.mutacion(hijo2))

            # Reemplazamos la población vieja con la nueva generación evolucionada
            # Actualiza el estado del objeto para que la siguiente generación parta de estos nuevos individuos
            self.poblacion = nueva_poblacion


    def guardar_memoria(self):
        # Abre (o crea) el archivo JSON en modo escritura
        with open("synapsis_genetico.json", "w") as archivo:
            # Serializa y almacena la información del mejor individuo actual (el de la posición 0)
            json.dump(self.poblacion[0], archivo)
        

    def obtener_memoria(self):
        # Abre el archivo JSON en modo lectura para recuperar los datos históricos
        with open("synapsis_genetico.json", "r") as archivo:
            # Transforma el texto JSON de vuelta a un diccionario de Python
            datos_cargados = json.load(archivo)
        
        # Retorna específicamente el arreglo de pesos/genes del individuo respaldado
        return datos_cargados["genes"]