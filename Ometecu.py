import os
import sys
import time
import json
import random
import math
import numpy as np
from multiprocessing import Pool
from Memoria import Cerebro
from red_neuronal import Red

class Ometecu:

    def __init__(self):
        # 1. Instanciamos los objetos propios de esta clase
        self.cerebro = Cerebro()
        self.red = Red()

        # 2. Configuramos el cerebro
        self.cerebro.setVariables([0.9, 0.9, 0.1, 0.9])
        self.cerebro.setConfig([0.2, 1, 1, 1])
        
        # 3. Extraemos configuración necesaria para la Red
        config_neo = self.cerebro.getConfig()
        
        # 4. Inicializamos variables de control y estado
        self.variables_entrenamiento = []
        self.salidas = [0] * self.cerebro.numvar
        self.ciclos = 1
        self.entradas = [0.9]
        
        # 5. Rangos de escala
        self.rango_min_red = 0.1
        self.rango_max_red = 0.9
        self.rango_min_predic = 1
        self.rango_max_predic = 10

        # 6. Creamos la estructura de la red con los datos de config_neo
        self.red.crearRed(
            float(config_neo["apren"]), 
            int(config_neo["capa1"]), 
            int(config_neo["capa2"]), 
            int(config_neo["capa3"]), 
            's', 's', 's'
        )
    
    def estadisticaRed(self):
        return self.red.estadisticaRed()

    def set_valores_neurona_rangos(self, minimo, maximo):
        self.rango_max_predic = maximo
        self.rango_min_predic = minimo

    def set_entradas(self, var):
        self.entradas = var

    def set_ciclos(self, var):
        self.ciclos = var

    def set_valor_aprender(self, var):
        self.variablesEntrenamiento = var

    def funcionActivacion(self, r1, r2, r3):
        self.red.funcionActivacion(r1, r2, r3)
     

    def set_config_red(self, capa_inicial, capa_intermedia, capa_final, synapsis=True):
        self.cerebro.setConfig([0.2] + [capa_inicial, capa_intermedia, capa_final])
        self.configNeo = self.cerebro.getConfig()
        self.red.reset()
        self.red.crearRed(float(self.configNeo["apren"]), int(self.configNeo["capa1"]), int(self.configNeo["capa2"]),
                     int(self.configNeo["capa3"]), 's', 's', 's')
        if synapsis == True:
            if len(self.cerebro.getMemoria()) == len(self.red.getMemoria()):
                self.red.setMemoria(self.cerebro.getMemoria())

    def entrenamiento(self):
        for y in range(self.ciclos):
            #self.variablesEntrenamiento = self.cerebro.getVariables()
            self.red.setValorEntradas(self.entradas)
            self.red.valorEntrenamiento(self.variablesEntrenamiento)
            self.red.ejecutar()
            self.red.entrenar()
            self.salidas = self.red.salidaFinal()
            #print (self.salidas)

    def prediccion_old(self):
        self.red.setValorEntradas(self.entradas)
        self.red.ejecutar()
        self.salidas = self.red.salidaFinal()
        return self.salidas

    def prediccion(self, synapsis=True):
        self.set_memoria()
        if synapsis == True:  
            if len(self.cerebro.getMemoria()) == len(self.red.getMemoria()):
                self.red.setMemoria(self.cerebro.getMemoria())
        self.red.setValorEntradas(self.entradas)
        self.red.ejecutar()
        self.salidas = self.red.salidaFinal()    
           
        return self.salidas


    def set_memoria_genetico(self, memoria):
        self.red.setMemoria(memoria)

    def set_memoria(self):
        self.cerebro.setMemoria(self.red.getMemoria())
    
    def get_memoria(self):
        return self.red.getMemoria()



    # ejemplo lista de edades entre 5 a 80 años, normalizar 20 años entre este rango,
    # valor normalziar = 20, min_original = 5, maximo_original = 80.
    # para datos mas reales.
    def normalizar_dato_individual(self, valor_a_normalizar, min_original, max_original, min_destino=0.2, max_destino=0.8):
        rango_original = max_original - min_original
        rango_destino = max_destino - min_destino

        # Manejo de la división por cero si el rango original es cero
        if rango_original == 0:
            # Si el rango es cero, devuelve el punto medio del rango de destino
            return (min_destino + max_destino) / 2

        # Aplicación de la fórmula Min-Max
        valor_escalado_0_1 = (valor_a_normalizar - min_original) / rango_original
        
        valor_norm = (valor_escalado_0_1 * rango_destino) + min_destino
        
        return valor_norm



