import random

class AlgoritmoGenetico:
    def __init__(self, tam_poblacion, longitud_genoma, tasa_mutacion, num_generaciones):
        self.tam_poblacion = tam_poblacion
        self.longitud_genoma = longitud_genoma
        self.tasa_mutacion = tasa_mutacion
        self.num_generaciones = num_generaciones
        self.poblacion = []

    def inicializar_poblacion(self):
        self.poblacion = [[round(random.uniform(0.1, 0.9), 1) for _ in range(self.longitud_genoma)] for _ in range(self.tam_poblacion)]

    def set_poblacion(self, var):
        self.poblacion = var
    
    def get_poblacion(self):
        return self.poblacion
    

    def calcular_aptitud(self, genoma):
        # Debe ser implementada de acuerdo con el problema específico
        return genoma[-1]  # Ejemplo de función de aptitud

    def seleccion(self):
        fitness_scores = [self.calcular_aptitud(genoma) for genoma in self.poblacion]
        return random.choices(self.poblacion, weights=fitness_scores, k=self.tam_poblacion)

    def cruce(self, parent1, parent2):
        punto_cruce = random.randint(1, self.longitud_genoma - 1)
        hijo1 = parent1[:punto_cruce] + parent2[punto_cruce:]
        hijo2 = parent2[:punto_cruce] + parent1[punto_cruce:]
        return hijo1, hijo2

    def mutacion(self, genoma):
        for i in range(self.longitud_genoma):
            if random.random() < self.tasa_mutacion:
                if not i == self.longitud_genoma:
                    genoma[i] = round(random.uniform(0.1, 0.9), 1)
        return genoma

    def ejecutar(self):
        #self.inicializar_poblacion()

        for _ in range(self.num_generaciones):
            seleccionados = self.seleccion()

            nueva_poblacion = []
            for i in range(0, self.tam_poblacion, 2):
                padre1, padre2 = seleccionados[i], seleccionados[i+1]
                hijo1, hijo2 = self.cruce(padre1, padre2)
                nueva_poblacion.extend([self.mutacion(hijo1), self.mutacion(hijo2)])

            self.poblacion = nueva_poblacion

# # Ejemplo de uso
# ag = AlgoritmoGenetico(tam_poblacion=100, longitud_genoma=11, tasa_mutacion=0.01, num_generaciones=30)
# ag.ejecutar()
# for x in ag.poblacion:
#     print(x)

