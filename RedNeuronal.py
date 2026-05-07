from Neurona import Neurona

class Red:
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
        self.neuOcultasY1 = []
        self.neuOcultasY2 = []
        self.neuSalidas = []
        self.valorDeseado = 0
        self.ciclos = 0
        self.entredas = []

    def reset(self):
        self.neuOcultasY1 = []
        self.neuOcultasY2 = []
        self.neuSalidas = []
        self.valorDeseado = 0
        self.ciclos = 0
        self.entredas = []

    def funcionActivacion(self, x, y, z):
        self.funcionE = x
        self.funcionO = y
        self.funcionS = z

    def setRangosRandom(self, mini, maxi):
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setRangosRandom(mini, maxi)

        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setRangosRandom(mini, maxi)

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setRangosRandom(mini, maxi)

    def setPesosRandom(self):
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setPesosRandom()

        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setPesosRandom()

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setPesosRandom()

    def setAprendizaje(self, var):
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setAprendizaje(var)

        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setAprendizaje(var)

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setAprendizaje(var)

    def numNeuCapasOcultas(self, num):
        for x in range(num):
            obj = Neurona()
            self.neuOcultasY2.append(obj)

    def numNueCapasSalida(self, num):
        for x in range(num):
            obj = Neurona()
            self.neuSalidas.append(obj)

    def numEntras(self, num):
        for x in range(num):
            obj = Neurona()
            self.neuOcultasY1.append(obj)

        for x in range(len(self.neuOcultasY1)):
            # SE DEFINE CON VALOR "1" POR EL DISEÑO DE RED, CADA ENTRA ES UNA NEURONA
            # POR LO CUAL SOLO TIENE UNA RAMA DE PESO.
            self.neuOcultasY1[x].setNumEntras(1)  # NO SOLO SE ASIGNADA LA ENTRA, TAMBIEN EL PESO DE LA MISMA.


        for x in range(len(self.neuOcultasY2)):
            self.neuOcultasY2[x].setNumEntras(len(self.neuOcultasY1))

        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].setNumEntras(len(self.neuOcultasY2))

    def setValorEntradas(self, var):
        self.entradas = len(var)
        for x in range(len(self.neuOcultasY1)):
            self.neuOcultasY1[x].setValoresEntras([var[x]])

    def valorEntrenamiento(self, var):
        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].valorReal = var[x]

    def totalCiclos(self):
        return self.ciclos

    def getEntradasRed(self):
        return self.neuOcultasY1[0].entradas

    def estadisticaRed(self):
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
        for x in range(len(self.neuSalidas)):
            self.neuSalidas[x].errorReal()
            self.neuSalidas[x].calcularPeso()


        for x, neu2 in enumerate(self.neuOcultasY2):
            # Calcula el error acumulado para esta neurona de Y2
            error_acumulado = sum(
                neu_salida.pesos[x] * neu_salida.errorC
                for neu_salida in self.neuSalidas
            )

            # Propaga el error estimado a la neurona y ajusta sus pesos
            neu2.error_estimado_propagado(error_acumulado)
            neu2.calcularPeso()

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
    
         # --- Capa oculta Y1 ---
        for neu1 in self.neuOcultasY1:
            if self.funcionE == 's':
                neu1.calcularSalidaSigmoide()
            elif self.funcionE == 'r':
                neu1.calcularSalidaReLU()
            elif self.funcionE == 't':
                neu1.calcularSalidaHiperbolica()

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
        aux = []
        for x in range(len(self.neuSalidas)):
            aux.append(self.neuSalidas[x].getSalida())
        return aux

    def setMemoria(self, var):
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


    def getMemoria(self):
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
        self.setRangosRandom(-1, 1)
        self.setAprendizaje(aprendizaje)
        self.numNeuCapasOcultas(cantidadCapOculta)
        self.numNueCapasSalida(cantidadSalidas)
        self.numEntras(cantidadEntradas)
        self.funcionActivacion(f1, f2, f3)
