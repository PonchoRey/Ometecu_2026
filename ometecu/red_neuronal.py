from .Neurona import Neurona

class Red:
    """
    Clase que representa una Red Neuronal Artificial de tipo Perceptrón Multicapa.
    Contiene dos capas ocultas (Y1, Y2) y una capa de salida.
    """
    neuOcultasY2 = []
    neuOcultasY1 = []
    neuSalidas = []
    valorDeseado = 0
    ciclos = 0
    entredas = []
    funcionE = 's'
    funcionO = 's'
    funcionS = 's'

    def __init__(self):
        """
        Constructor de la clase. 
        Inicializa las listas que contendrán las neuronas de cada capa y restablece los valores base.
        """
        self.neuOcultasY1 = []
        self.neuOcultasY2 = []
        self.neuSalidas = []
        self.valorDeseado = 0
        self.ciclos = 0
        self.entredas = []
        self.memoria = []

    def reset(self):
        """
        Restablece la red neuronal vaciando las capas y reiniciando contadores y valores.
        """
        self.neuOcultasY1 = []
        self.neuOcultasY2 = []
        self.neuSalidas = []
        self.valorDeseado = 0
        self.ciclos = 0
        self.entredas = []

    def funcionActivacion(self, x, y, z):
        """
        Define el tipo de función de activación para cada capa de la red.
        's' = Sigmoide, 'r' = ReLU, 't' = Tangente Hiperbólica.
        
        Args:
            x (str): Función para la primera capa oculta (Y1).
            y (str): Función para la segunda capa oculta (Y2).
            z (str): Función para la capa de salida.
        """
        self.funcionE = x
        self.funcionO = y
        self.funcionS = z

    def setRangosRandom(self, mini, maxi):
        """
        Establece el rango de valores permitidos para la generación aleatoria de pesos 
        en todas las neuronas de la red.
        """
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setRangosRandom(mini, maxi)

        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setRangosRandom(mini, maxi)

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setRangosRandom(mini, maxi)

    def setPesosRandom(self):
        """
        Asigna pesos aleatorios a las conexiones de todas las neuronas 
        en las tres capas de la red.
        """
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setPesosRandom()

        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setPesosRandom()

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setPesosRandom()

    def setAprendizaje(self, var):
        """
        Establece la tasa de aprendizaje (learning rate) para todas las neuronas de la red.
        """
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setAprendizaje(var)

        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setAprendizaje(var)

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setAprendizaje(var)

    def numNeuCapasOcultas(self, num):
        """
        Crea e instancia un número específico de neuronas para la segunda capa oculta (Y2).
        """
        for x in range(num):
            obj = Neurona()
            self.neuOcultasY2.append(obj)

    def numNueCapasSalida(self, num):
        """
        Crea e instancia un número específico de neuronas para la capa de salida.
        """
        for x in range(num):
            obj = Neurona()
            self.neuSalidas.append(obj)

    def numEntras(self, num):
        """
        Configura la arquitectura de las capas: crea la primera capa oculta (Y1) 
        y define cuántas entradas recibirá cada neurona en cada capa subsecuente.
        """
        # Crea las neuronas de la primera capa oculta (Y1)
        for x in range(num):
            obj = Neurona()
            self.neuOcultasY1.append(obj)

        # Configura las entradas de Y1 (1 entrada por neurona)
        for x in range(len(self.neuOcultasY1)):
            # SE DEFINE CON VALOR "1" POR EL DISEÑO DE RED, CADA ENTRA ES UNA NEURONA
            # POR LO CUAL SOLO TIENE UNA RAMA DE PESO.
            self.neuOcultasY1[x].setNumEntras(num)  # NO SOLO SE ASIGNADA LA ENTRA, TAMBIEN EL PESO DE LA MISMA.

        # Configura que Y2 reciba tantas entradas como salidas tenga Y1
        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setNumEntras(len(self.neuOcultasY1))

        # Configura que la capa de salida reciba tantas entradas como salidas tenga Y2
        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setNumEntras(len(self.neuOcultasY2))

    def setValorEntradas(self, var):
        """
        Asigna los valores de los datos de entrada a la primera capa (Y1).
        """
        self.entradas = len(var)
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setValoresEntras(var)

    def valorEntrenamiento(self, var):
        """
        Asigna los valores reales/deseados (targets) a la capa de salida 
        para el proceso de cálculo de error supervisado.
        """
        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].valorReal = var[x]

    def totalCiclos(self):
        """
        Devuelve el número total de ciclos (épocas) transcurridos.
        """
        return self.ciclos

    def getEntradasRed(self):
        """
        Devuelve las entradas registradas en la primera neurona de la capa Y1.
        """
        return self.neuOcultasY1[0].entradas

    def estadisticaRed(self):
        """
        Imprime en consola el estado actual (entradas, pesos y salidas) 
        de todas las neuronas, separadas por capas. Útil para depuración.
        """
        for x in range(len(self.neuOcultasY1)):
            print(" capa 1,", "neurona:", str(x)+"," , "datos_entrada:",self.neuOcultasY1[x].entradas)
            print(" capa 1,", "neurona:", str(x)+"," , "datos_pesos:", self.neuOcultasY1[x].pesos)
            print(" capa 1,", "neurona:", str(x)+"," , "datos_salida:", self.neuOcultasY1[x].salida)

        for x in range(len(self.neuOcultasY2)):
            print(" capa 2,", "neurona:", str(x)+"," , "datos_entrada:",self.neuOcultasY2[x].entradas)
            print(" capa 2,", "neurona:", str(x)+"," , "datos_pesos:", self.neuOcultasY2[x].pesos)
            print(" capa 2,", "neurona:", str(x)+"," , "datos_salida:", self.neuOcultasY2[x].salida)

        for x in range(len(self.neuSalidas)):
            print(" capa 3,", "neurona:", str(x)+"," , "datos_entrada:",self.neuSalidas[x].entradas)
            print(" capa 3,", "neurona:", str(x)+"," , "datos_pesos:", self.neuSalidas[x].pesos)
            print(" capa 3,", "neurona:", str(x)+"," , "datos_salida:", self.neuSalidas[x].salida)


    def entrenar(self):
        """
        Realiza el proceso de Retropropagación (Backpropagation).
        Calcula el error en la capa de salida y lo propaga hacia atrás ajustando los pesos.
        """
        # 1. Calcula error y ajusta pesos en la Capa de Salida
        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].errorReal()
            self.neuSalidas[x].calcularPeso()

        # 2. Propaga el error y ajusta pesos en la Capa Oculta Y2
        for x, neu2 in enumerate(self.neuOcultasY2):
            # Calcula el error acumulado para esta neurona de Y2
            error_acumulado = sum(
                neu_salida.pesos[x] * neu_salida.errorC
                for neu_salida in self.neuSalidas
            )

            # Propaga el error estimado a la neurona y ajusta sus pesos
            neu2.error_estimado_propagado(error_acumulado)
            neu2.calcularPeso()

        # 3. Propaga el error y ajusta pesos en la Capa Oculta Y1
        for x, neu1 in enumerate(self.neuOcultasY1):
            # Calcula el error acumulado de esta neurona en la capa Y1
            error_acumulado = sum(
                neu2.pesos[x] * neu2.errorC
                for neu2 in self.neuOcultasY2
            )

            # Propaga el error estimado a la neurona y ajusta sus pesos
            neu1.error_estimado_propagado(error_acumulado)
            neu1.calcularPeso()

 
    def ejecutar(self):
        """
        Realiza el proceso de propagación hacia adelante (Feedforward).
        Toma las entradas, las pasa por cada capa calculando su activación 
        hasta llegar a la salida.
        """
         # --- Capa oculta Y1 ---
        for neu1 in self.neuOcultasY1:
            if self.funcionE == 's':
                neu1.calcularSalidaSigmoide()
            elif self.funcionE == 'r':
                neu1.calcularSalidaReLU()
            elif self.funcionE == 't':
                neu1.calcularSalidaHiperbolica()

        # Recolecta las salidas de Y1 para usarlas como entradas en Y2
        auxN = [neu1.salida for neu1 in self.neuOcultasY1]

        # --- Capa oculta Y2 ---
        for neu2 in self.neuOcultasY2:
            neu2.setValoresEntras(auxN)
            if self.funcionO == 's':
                neu2.calcularSalidaSigmoide()
            elif self.funcionO == 'r':
                neu2.calcularSalidaReLU()
            elif self.funcionO == 't':
                neu2.calcularSalidaHiperbolica()

        # Recolecta las salidas de Y2 para usarlas como entradas en la Capa de Salida
        aux = [neu2.salida for neu2 in self.neuOcultasY2]

        # --- Capa de salida ---
        for neu_salida in self.neuSalidas:
            neu_salida.setValoresEntras(aux)
            if self.funcionS == 's':
                neu_salida.calcularSalidaSigmoide()
            elif self.funcionS == 'r':
                neu_salida.calcularSalidaReLU()
            elif self.funcionS == 't':
                neu_salida.calcularSalidaHiperbolica()

    def salidaFinal(self):
        """
        Obtiene y devuelve en una lista los valores calculados por la capa de salida.
        """
        aux = []
        for x in range(len(self.neuSalidas)):
            aux.append(self.neuSalidas[x].getSalida())
        return aux

    # oportunidad de mejora, código viejo que puede ser actualizado y optmizado 
    def setMemoria(self, var):
        """
        Carga un conjunto de pesos (memoria) previamente guardados 
        a todas las neuronas de la red de forma secuencial.
        
        Args:
            var (list): Lista unidimensional con todos los pesos de la red.
        """
        memoria = var
        memoriaC = 0

        for x in range(len(self.neuOcultasY1)):
            aux = []
            for y in range(len(self.neuOcultasY1[x].pesos)):
                aux.append(memoria[memoriaC])
                memoriaC += 1
            self.neuOcultasY1[x].pesos = aux


        for x in range(len(self.neuOcultasY2)):

            aux = []
            for y in range(len(self.neuOcultasY2[x].pesos)):
                aux.append(memoria[memoriaC])
                memoriaC += 1
            self.neuOcultasY2[x].pesos = aux


        for x in range(len(self.neuSalidas)):
            aux = []
            for y in range(len(self.neuSalidas[x].pesos)):
                aux.append(memoria[memoriaC])
                memoriaC += 1
            self.neuSalidas[x].pesos = aux


    # oportunidad de mejora, código viejo que puede ser actualizado y optmizado
    def getMemoria(self):
        """
        Extrae y devuelve todos los pesos actuales de la red en una lista plana (unidimensional).
        Útil para guardar el modelo entrenado.
        """
        memoria = []

        for x in range(len(self.neuOcultasY1)):
            aux = self.neuOcultasY1[x].pesos
            for y in aux:
                memoria.append(y)

        for x in range(len(self.neuOcultasY2)):
            aux = self.neuOcultasY2[x].pesos
            for y in aux:
                memoria.append(y)

        for x in range(len(self.neuSalidas)):
            aux = self.neuSalidas[x].pesos
            for y in aux:
                memoria.append(y)


        return memoria

    def crearRed(self, aprendizaje, cantidadEntradas, cantidadCapOculta, cantidadSalidas, f1, f2, f3):
        """
        Método unificador (facade) que inicializa y estructura toda la red de una sola vez.
        
        Args:
            aprendizaje (float): Tasa de aprendizaje de la red.
            cantidadEntradas (int): Número de neuronas en la capa Y1 (entradas).
            cantidadCapOculta (int): Número de neuronas en la capa Y2.
            cantidadSalidas (int): Número de neuronas en la capa de salida.
            f1, f2, f3 (str): Funciones de activación para Y1, Y2 y Salida, respectivamente.
        """
        self.setRangosRandom(-1, 1)
        self.setAprendizaje(aprendizaje)
        self.numNeuCapasOcultas(cantidadCapOculta)
        self.numNueCapasSalida(cantidadSalidas)
        self.numEntras(cantidadEntradas)
        self.funcionActivacion(f1, f2, f3)