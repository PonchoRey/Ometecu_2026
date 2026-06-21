from .Memoria import Cerebro
from .red_neuronal import Red

class Ometecu:

    def __init__(self):
        # 1. Instanciamos los objetos propios de esta clase
        # Se crean las instancias de la memoria (Cerebro) y de la estructura neuronal (Red)
        self.cerebro = Cerebro()
        self.red = Red()
        self.nombre_synapsis = ""
        
        # 4. Inicializamos variables de control y estado
        # Parámetros para almacenar vectores de entrenamiento, salidas temporales y ciclos de ejecución
        self.variables_entrenamiento = []
        self.salidas = [0] * self.cerebro.numvar
        self.ciclos = 1
        self.entradas = [0.9]
        
        # 5. Rangos de escala
        # Límites para los procesos de normalización y desnormalización de datos dentro del flujo
        self.rango_min_red = 0.1
        self.rango_max_red = 0.9
        self.rango_min_predic = 1
        self.rango_max_predic = 10

    
    
    def inicio_synapsis(self, nombre):
        # Configuramos el cerebro
        # Inicializa variables de entorno y configuración interna del objeto Cerebro        
        #self.cerebro.setVariables(nombre)
        self.nombre_synapsis = nombre
        self.cerebro.setConfig([0.2, 1, 1, 1], nombre)

        # Extraemos configuración necesaria para la Red
        # Obtiene un diccionario o estructura con los parámetros de la arquitectura de la red
        config_neo = self.cerebro.getConfig(nombre)

        # Creamos la estructura de la red con los datos de config_neo
        # Instancia las capas y neuronas usando la tasa de aprendizaje y las neuronas por capa (1, 2 y 3)
        self.red.crearRed(
            float(config_neo["apren"]), 
            int(config_neo["capa1"]), 
            int(config_neo["capa2"]), 
            int(config_neo["capa3"]), 
            's', 's', 's'
        )

    def estadisticaRed(self):
        # Retorna un resumen métrico o estado interno actual del rendimiento de la red
        return self.red.estadisticaRed()

    def set_valores_neurona_rangos(self, minimo, maximo):
        # Permite actualizar dinámicamente los límites de escala para las predicciones
        self.rango_max_predic = maximo
        self.rango_min_predic = minimo

    def set_entradas(self, var):
        # Asigna el vector de características de entrada que procesará la red
        self.entradas = var

    def set_ciclos(self, var):
        # Define la cantidad de épocas o iteraciones para el bucle de entrenamiento
        self.ciclos = var

    def set_valor_aprender(self, var):
        # Establece el vector de valores objetivos (target) esperados para el entrenamiento
        self.variablesEntrenamiento = var

    def funcionActivacion(self, r1, r2, r3):
        # Configura las funciones de transferencia para las capas de la red neuronal
        self.red.funcionActivacion(r1, r2, r3)
     

    def set_config_red(self, capa_inicial, capa_intermedia, capa_final):
        synapsis = True
        # Actualiza la configuración en el Cerebro con la nueva distribución de neuronas por capa
        self.cerebro.setConfig([0.2] + [capa_inicial, capa_intermedia, capa_final], self.nombre_synapsis)
        self.configNeo = self.cerebro.getConfig(self.nombre_synapsis)
        
        # Reinicia la red y la vuelve a construir desde cero con la nueva estructura
        self.red.reset()
        self.red.crearRed(float(self.configNeo["apren"]), int(self.configNeo["capa1"]), int(self.configNeo["capa2"]),
                     int(self.configNeo["capa3"]), 's', 's', 's')
        
        # Si está activo el flag y los tamaños coinciden, hereda los pesos guardados en la memoria
        if synapsis == True:
            try:
                self.cerebro.getMemoria(self.nombre_synapsis)
            except FileNotFoundError:
                self.set_memoria()

            if len(self.cerebro.getMemoria(self.nombre_synapsis)) == len(self.red.getMemoria()):
                self.red.setMemoria(self.cerebro.getMemoria(self.nombre_synapsis))

    def entrenamiento(self):
        # Ejecuta el proceso de aprendizaje repetidas veces según el número de ciclos configurados
        for y in range(self.ciclos):
            # Carga las entradas actuales y los valores esperados en la estructura de la red
            self.red.setValorEntradas(self.entradas)
            self.red.valorEntrenamiento(self.variablesEntrenamiento)
            
            # Realiza la pasada hacia adelante (feedforward) y aplica el algoritmo de backpropagation (entrenar)
            self.red.ejecutar()
            self.red.entrenar()
            
            # Almacena el resultado final obtenido de la capa de salida en esta iteración
            self.salidas = self.red.salidaFinal()

    def prediccion_old(self):
        # Versión previa de predicción: inyecta entradas, procesa la red y extrae los resultados directamente
        self.red.setValorEntradas(self.entradas)
        self.red.ejecutar()
        self.salidas = self.red.salidaFinal()
        return self.salidas

    def prediccion(self, synapsis=True):
        # Sincroniza la memoria actual de la red hacia el componente Cerebro
        self.set_memoria()
        
        # Si se solicita transferencia de sinapsis y las dimensiones de memoria coinciden,
        # inyecta los pesos almacenados del Cerebro de vuelta a la Red antes de evaluar
        if synapsis == True:  
            if len(self.cerebro.getMemoria(self.nombre_synapsis)) == len(self.red.getMemoria()):
                self.red.setMemoria(self.cerebro.getMemoria(self.nombre_synapsis))
                
        # Proceso estándar de inferencia: establece entradas, calcula y devuelve la salida predicha
        self.red.setValorEntradas(self.entradas)
        self.red.ejecutar()
        self.salidas = self.red.salidaFinal()    
           
        return self.salidas


    def set_memoria_genetico(self, memoria):
        # Permite inyectar directamente una lista de pesos externos (por ejemplo, provenientes del AG) a la red
        self.red.setMemoria(memoria)

    def set_memoria(self):
        # Transfiere los pesos sinápticos actuales de la red para respaldarlos en el objeto Cerebro
        self.cerebro.setMemoria(self.red.getMemoria(), self.nombre_synapsis)
    
    def get_memoria(self):
        # Recupera y devuelve el arreglo lineal de pesos sinápticos que posee la red en este momento
        return self.red.getMemoria()


    def normalizar_dato_individual(self, valor_a_normalizar, min_original, max_original, min_destino=0.2, max_destino=0.8):
        """
        Aplica una normalización Min-Max para transformar un dato real a una escala apta para la red.
        Ejemplo: Mapear una edad entre 5 y 80 años hacia un rango seguro como [0.2, 0.8].
        """
        # Calcula la amplitud de los rangos de origen y de destino
        rango_original = max_original - min_original
        rango_destino = max_destino - min_destino

        # Manejo de la división por cero si el rango original es cero (evita errores si min y max son iguales)
        if rango_original == 0:
            # Si el rango es cero, devuelve el punto medio del rango de destino
            return (min_destino + max_destino) / 2

        # Aplicación de la fórmula Min-Max: Escala el valor primero a una base [0, 1]
        valor_escalado_0_1 = (valor_a_normalizar - min_original) / rango_original
        
        # Desplaza y ajusta el valor escalado al rango de destino final (por defecto 0.2 a 0.8)
        valor_norm = (valor_escalado_0_1 * rango_destino) + min_destino
        
        return valor_norm