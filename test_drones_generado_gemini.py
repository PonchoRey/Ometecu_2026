import pygame
import math
import random
from Ometecu import Ometecu
from baseGenetico import AlgoritmoGenetico

# --- Configuración del Entorno ---
TAM_POBLACION = 40 
ANCHO, ALTO = 800, 600
pygame.init()
pantalla = pygame.display.set_mode((ANCHO, ALTO))
reloj = pygame.time.Clock()
fuente = pygame.font.SysFont("consolas", 14)

global MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones
contador_generaciones = 0
MODO_ENTRENAMIENTO_RAPIDO = True 

# --- Inicialización de las dos facciones (Cian vs Naranja) ---
poblacion_A = []
poblacion_B = []

for _ in range(TAM_POBLACION):
    for lista in [poblacion_A, poblacion_B]:
        red = Ometecu()
        # 6 Entradas: [MiX, MiY, MiAngulo, EnemigoX, EnemigoY, EnemigoAngulo]
        red.set_config_red(capa_inicial=6, capa_intermedia=12, capa_final=3)
        # ReLU en las internas ('r') y Lineal en la salida ('l')
        red.funcionActivacion("r", "r", "r")
        red.set_memoria_genetico([random.uniform(-1, 1) for _ in red.get_memoria()])
        lista.append(red)

ag_A = AlgoritmoGenetico(tam_poblacion=TAM_POBLACION, tasa_mutacion=0.12, num_generaciones=1)
ag_B = AlgoritmoGenetico(tam_poblacion=TAM_POBLACION, tasa_mutacion=0.12, num_generaciones=1)

class Bala:
    def __init__(self, x, y, angulo, color):
        self.x, self.y = x, y
        self.vx = math.cos(math.radians(angulo)) * 10
        self.vy = math.sin(math.radians(angulo)) * 10
        self.vida = 40
        self.color = color

    def actualizar(self):
        self.x += self.vx
        self.y += self.vy
        self.vida -= 1

class DronIA:
    def __init__(self, x, y, color, index_red, lista_redes):
        self.x_ini, self.y_ini = x, y
        self.color = color
        self.index_red = index_red
        self.lista_redes = lista_redes
        self.reset()

    def reset(self):
        self.x, self.y = self.x_ini, self.y_ini
        self.vx, self.vy = 0, 0
        self.angulo = random.randint(0, 360)
        self.balas = []
        self.cadencia = 0
        self.fitness = 0
        self.vivo = True

    def actuar(self, enemigo):
        if not self.vivo: 
            self.balas = [] # Limpiar balas si muere
            return

        # Normalización de entradas
        inputs = [self.x/ANCHO, self.y/ALTO, (self.angulo%360)/360, 
                  enemigo.x/ANCHO, enemigo.y/ALTO, (enemigo.angulo%360)/360]
        
        red = self.lista_redes[self.index_red]
        red.set_entradas(inputs)
        red.prediccion_old()
        
        empuje, giro, disparar = red.salidas

        # Aplicar físicas
        self.angulo += giro * 6
        rad = math.radians(self.angulo)
        if empuje > 0:
            self.vx += math.cos(rad) * 0.25
            self.vy += math.sin(rad) * 0.25
        
        self.vx *= 0.96
        self.vy *= 0.96
        self.x += self.vx
        self.y += self.vy

        # Disparo
        if self.cadencia > 0: self.cadencia -= 1
        if disparar > 0.5 and self.cadencia == 0:
            self.balas.append(Bala(self.x, self.y, self.angulo, self.color))
            self.cadencia = 25

        # Colisión de balas
        for b in self.balas[:]:
            b.actualizar()
            if enemigo.vivo:
                dist = math.sqrt((b.x - enemigo.x)**2 + (b.y - enemigo.y)**2)
                if dist < 20:
                    self.fitness += 2000 # Gran premio por derribo
                    enemigo.fitness -= 500 # Castigo por ser derribado
                    enemigo.vivo = False
                    enemigo.balas = []
                    self.balas.remove(b)
                    continue
            if b.vida <= 0: self.balas.remove(b)
        
        # Penalización por salirse del mapa
        if self.x < 0 or self.x > ANCHO or self.y < 0 or self.y > ALTO:
            self.fitness -= 300
            self.vivo = False
            self.balas = []

    def dibujar(self):
        if not self.vivo: return
        pygame.draw.circle(pantalla, self.color, (int(self.x), int(self.y)), 15, 2)
        rad = math.radians(self.angulo)
        pygame.draw.line(pantalla, self.color, (self.x, self.y), 
                         (self.x + math.cos(rad)*20, self.y + math.sin(rad)*20), 3)
        for b in self.balas:
            pygame.draw.circle(pantalla, b.color, (int(b.x), int(b.y)), 3)

def main():
    global poblacion_A, poblacion_B, MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones
    
    while True:
        drones_A = [DronIA(100, ALTO//2, (0, 255, 255), i, poblacion_A) for i in range(TAM_POBLACION)]
        drones_B = [DronIA(ANCHO-100, ALTO//2, (255, 128, 0), i, poblacion_B) for i in range(TAM_POBLACION)]

        corriendo = True
        tiempo_duelo = 0
        while corriendo:
            if not MODO_ENTRENAMIENTO_RAPIDO:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: pygame.quit(); return

            duelos_activos = 0
            for i in range(TAM_POBLACION):
                dA, dB = drones_A[i], drones_B[i]
                if dA.vivo or dB.vivo:
                    dA.actuar(dB)
                    dB.actuar(dA)
                    if dA.vivo and dB.vivo: duelos_activos += 1
            
            tiempo_duelo += 1
            if duelos_activos == 0 or tiempo_duelo > 700: corriendo = False

            if not MODO_ENTRENAMIENTO_RAPIDO:
                pantalla.fill((15, 15, 20))
                for i in range(TAM_POBLACION):
                    drones_A[i].dibujar()
                    drones_B[i].dibujar()
                
                info = fuente.render(f"Gen: {contador_generaciones} | Duelos: {duelos_activos}", True, (255,255,255))
                pantalla.blit(info, (10, 10))
                pygame.display.flip()
                reloj.tick(60)
            else:
                reloj.tick(0)

        # Evolución
        scores_A = [max(1, d.fitness) for d in drones_A]
        scores_B = [max(1, d.fitness) for d in drones_B]
        
        contador_generaciones += 1
        print(f"Gen {contador_generaciones} | Max A: {int(max(scores_A))} | Max B: {int(max(scores_B))}")
        
        # Desactivar modo rápido tras 150 generaciones para ver el combate
        if contador_generaciones >= 20: 
            MODO_ENTRENAMIENTO_RAPIDO = False
            ag_A.guardar_memoria("drones")
        
        poblacion_A = ag_A.redes_genetico(scores_A, poblacion_A)
        poblacion_B = ag_B.redes_genetico(scores_B, poblacion_B)


if __name__ == "__main__":
    main()