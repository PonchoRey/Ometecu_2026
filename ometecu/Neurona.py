import random
import math 

class Neurona:

    entradas = []
    valorReal = 1
    pesos = []
    errorEstimado = 0
    errorC = 0
    aprendizaje = 0.2
    salida = 0
    rawSalida = 0
    rangoMin = -1
    rangoMax = 1

    def __init__(self):
        self.entradas = []
        self.pesos = []

    def setRangosRandom(self, mini, maxi):
        self.rangoMin = mini
        self.rangoMax = maxi

    def setAprendizaje(self, var):
        self.aprendizaje = var

    def setNumEntras(self, num):
        self.pesos = [round(random.uniform(self.rangoMin, self.rangoMax), 10) for _ in range(num)]


    def setPesosRandom(self):
        self.pesos = [round(random.uniform(self.rangoMin, self.rangoMax), 10) for _ in range(len(self.pesos))]

    def setValoresEntras(self, valor):
        self.entradas = valor

    def setValoresPesos(self, valor):
        self.pesos = valor

    def setSalida(self, var):
        self.salida = var


    def calcularSalidaHiperbolica(self):
        self.salida = 0
        for x in range(len(self.entradas)):
            self.salida += self.entradas[x] * self.pesos[x]
        self.rawSalida = self.salida
        self.salida = round(math.tanh(self.salida), 10)
        if self.salida >= 5:
            self.salida = 5
        elif self.salida <= -5:
            self.salida = -5
      

    def calcularSalidaSigmoide(self):
        self.salida = 0
        for x in range(len(self.entradas)):
            self.salida += self.entradas[x] * self.pesos[x]
        self.rawSalida = self.salida
        self.salida = round(1 / (1 + math.exp(-self.salida)), 10)


    def calcularSalidaReLU(self):
        aux = len(self.entradas)
        self.salida = 0
        for x in range(aux):
            self.salida += self.pesos[x] * self.entradas[x]
        
        self.rawSalida = self.salida

        # ReLU Pura: Si es menor o igual a 0, se vuelve 0. Si es mayor, se queda igual.
        if self.salida <= 0:
            self.salida = 0

    def errorReal(self):
        self.errorC = round(self.salida * (1 - self.salida) * (self.valorReal - self.salida), 10)

    def errorEstimado(self, salidaPeso, errorReal):
        self.errorC = round(self.salida * (1 - self.salida) * salidaPeso * errorReal, 10)

    def error_estimado_propagado(self, error_propagado):
        derivada = self.salida * (1 - self.salida) 
        self.errorC = round(derivada * error_propagado, 10)

    def calcularPeso(self):
        for x in range(len(self.entradas)):
            self.pesos[x] = round(self.pesos[x] + self.aprendizaje * self.entradas[x] * self.errorC, 10)

    def getSalida(self):
        return self.salida

    def getPesos(self):
        return self.pesos

    def getEntradas(self):
        return self.entradas
