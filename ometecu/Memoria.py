import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_synapsis_path(alias):
    return os.path.join(BASE_DIR, 'modelos', 'pesos', f'synapsis_{alias}.json')

def get_variables_path(alias):
    return os.path.join(BASE_DIR, 'modelos', 'config', f'variables_{alias}.json')

# libreria con posibilidad de mejora. 
class Cerebro:

    numvar = 0
    matriz = []

    def setMemoria(self, memoria, alias):
        with open(get_synapsis_path(alias), 'w') as file:
            json.dump(memoria, file, indent=4)

    def getMemoria(self, alias):
        with open(get_synapsis_path(alias)) as file:
            memoria = json.load(file)
        return memoria

    # def setVariables(self, variables, nombre):
    #     try:
    #         aux = self.getConfig(nombre)
    #     except FileNotFoundError:

    #         aux = {"neo": {"apren": 0.2,"capa1": 5,"capa2": 6,"capa3": 2}}
    #     memoria = '{"neo": {"apren": '+str(aux["apren"])+\
    #               ',"capa1": '+str(aux["capa1"])+',"capa2": '+str(aux["capa2"])+',"capa3": '+str(aux["capa3"])+'}'

    #     memoria += "}"

    #     file3 = open(get_variables_path(nombre), "w")
    #     file3.write(memoria)
    #     file3.close()

    def getVariables(self):
        pass
        # with open(get_variables_path()) as file:
        #     memoria = json.load(file)
        # self.matriz = []
        # cont = -1
        # for y in memoria:
        #     cont += 1
        # self.numvar = cont
        # for x in range(self.numvar):
        #     valor = memoria["var" + str(x)]
        #     self.matriz.append(valor["valor"])

        # return self.matriz


    def getConfig(self, nombre):
        with open(get_variables_path(nombre)) as file:
            memoria = json.load(file)
        return memoria["neo"]

    def setConfig(self, matriz, nombre):
        memoria = '{"neo": {"apren": '+str(matriz[0])+',"capa1": '+str(matriz[1])+\
                  ',"capa2": '+str(matriz[2])+',"capa3": '+str(matriz[3])+'}'

        memoria += "}"
        file3 = open(get_variables_path(nombre), "w")
        file3.write(memoria)
        file3.close()
