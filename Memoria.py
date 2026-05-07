import json

class Cerebro:

    numvar = 0
    matriz = []

    def setMemoria(self, memoria):
        with open('synapsis.json', 'w') as file:
            json.dump(memoria, file, indent=4)

    def getMemoria(self):
        with open('synapsis.json') as file:
            memoria = json.load(file)
        return memoria

    def setVariables(self, data):
        aux = self.getConfig()
        memoria = '{"neo": {"apren": '+str(aux["apren"])+\
                  ',"capa1": '+str(aux["capa1"])+',"capa2": '+str(aux["capa2"])+',"capa3": '+str(aux["capa3"])+'}'

        memoria += "}"

        file3 = open("variables.json", "w")
        file3.write(memoria)
        file3.close()

    def getVariables(self):
        pass
        # with open('variables.json') as file:
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


    def getConfig(self):
        with open('variables.json') as file:
            memoria = json.load(file)
        return memoria["neo"]

    def setConfig(self, matriz):
        memoria = '{"neo": {"apren": '+str(matriz[0])+',"capa1": '+str(matriz[1])+\
                  ',"capa2": '+str(matriz[2])+',"capa3": '+str(matriz[3])+'}'

        memoria += "}"
        file3 = open("variables.json", "w")
        file3.write(memoria)
        file3.close()
