import pygame
import math
import random
from Ometecu import Ometecu
from baseGenetico import AlgoritmoGenetico

# --- Configuración ---
TAM_POBLACION = 22
ANCHO, ALTO = 1000, 700
pygame.init()
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pista_background = pygame.Surface((ANCHO, ALTO)) 

reloj = pygame.time.Clock()
fuente = pygame.font.SysFont("consolas", 16)

global MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones
contador_generaciones = 0
MODO_ENTRENAMIENTO_RAPIDO = True 

# --- Redes Ometecu ---
array_poblacion_red = []
for _ in range(TAM_POBLACION):
    red = Ometecu()
    red.set_config_red(capa_inicial=5, capa_intermedia=20, capa_final=2)
    red.funcionActivacion("s", "r", "s")
    red.set_memoria_genetico([random.uniform(-1, 1) for _ in red.get_memoria()])
    array_poblacion_red.append(red)

ag = AlgoritmoGenetico(tam_poblacion=TAM_POBLACION, tasa_mutacion=0.1, num_generaciones=1)

# Puntos de la pista para render
PISTA_PUNTOS = [(100, 350), (200, 150), (500, 100), (800, 150), (900, 350), (800, 550), (500, 600), (200, 550)]

def dibujar_pista_en_superficie(superficie):
    superficie.fill((0, 0, 0)) 
    if len(PISTA_PUNTOS) > 2:
        pygame.draw.polygon(superficie, (100, 100, 100), PISTA_PUNTOS)
        pygame.draw.circle(superficie, (0, 0, 0), (500, 350), 150)

dibujar_pista_en_superficie(pista_background)

class CocheIA:
    def __init__(self, index):
        self.index_red = index
        self.reset()

    def reset(self):
        self.x, self.y = 150, 350 
        self.angulo_coche = 270 
        self.velocidad = 0
        self.vivo = True
        self.max_angulo_alcanzado = 0 # Para medir progreso real en el circuito
        self.fitness = 0
        self.tiempo_sin_progreso = 0

    def cast_ray(self, offset):
        ray_ang = math.radians(self.angulo_coche + offset)
        for d in range(0, 150, 5): 
            tx = int(self.x + math.cos(ray_ang) * d)
            ty = int(self.y + math.sin(ray_ang) * d)
            if 0 <= tx < ANCHO and 0 <= ty < ALTO:
                if pista_background.get_at((tx, ty))[0] < 50: return d / 150.0
            else: return 0.0
        return 1.0

    def actualizar(self):
        if not self.vivo: return

        # Obtener ángulo relativo al centro del circuito (500, 350)
        # Esto nos dice en qué parte del círculo está el coche
        dx = self.x - 500
        dy = self.y - 350
        angulo_actual_circuito = math.degrees(math.atan2(dy, dx)) + 180 

        # Entradas y predicción
        sensores = [self.cast_ray(-90), self.cast_ray(-45), self.cast_ray(0), self.cast_ray(45), self.cast_ray(90)]
        red = array_poblacion_red[self.index_red]
        red.set_entradas(sensores)
        red.prediccion_old()
        
        acel, giro = red.salidas
        self.velocidad = max(0, min(self.velocidad + acel * 0.4, 6)) * 0.94
        self.angulo_coche += giro * 5
        self.x += math.cos(math.radians(self.angulo_coche)) * self.velocidad
        self.y += math.sin(math.radians(self.angulo_coche)) * self.velocidad

        # --- LÓGICA DE PENALIZACIÓN POR RETROCESO ---
        # Si el coche avanza en el sentido del circuito, premiamos
        if angulo_actual_circuito > self.max_angulo_alcanzado:
            # Solo premiamos si el salto no es un error de cálculo (cruce de 360 a 0)
            if angulo_actual_circuito - self.max_angulo_alcanzado < 50:
                self.fitness += (angulo_actual_circuito - self.max_angulo_alcanzado) * 10
                self.max_angulo_alcanzado = angulo_actual_circuito
                self.tiempo_sin_progreso = 0
        else:
            # Si se regresa o se queda en el mismo lugar, quitamos puntos
            self.fitness -= 5 
            self.tiempo_sin_progreso += 1

        # Si no progresa en 90 cuadros (1.5 seg), muere por cobarde
        if self.tiempo_sin_progreso > 90:
            self.vivo = False

        # Colisión
        if 0 <= int(self.x) < ANCHO and 0 <= int(self.y) < ALTO:
            if pista_background.get_at((int(self.x), int(self.y)))[0] < 50:
                self.vivo = False
        else: self.vivo = False

    def dibujar(self):
        if self.vivo: pygame.draw.circle(pantalla, (0, 255, 0), (int(self.x), int(self.y)), 5)

def main():
    global array_poblacion_red, MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones
    coches = [CocheIA(i) for i in range(TAM_POBLACION)]

    while True:
        corriendo = True
        tiempo_limite_gen = 2500
        while corriendo:
            if not MODO_ENTRENAMIENTO_RAPIDO:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: pygame.quit(); return

            vivos = 0
            for c in coches:
                if c.vivo:
                    c.actualizar()
                    vivos += 1
            
            tiempo_limite_gen -= 1
            if vivos == 0 or tiempo_limite_gen <= 0: corriendo = False

            if not MODO_ENTRENAMIENTO_RAPIDO:
                pantalla.blit(pista_background, (0,0))
                for c in coches: c.dibujar()
                pygame.display.flip()
                reloj.tick(60)
            else: reloj.tick(0)

        scores = [max(0, c.fitness) for c in coches]
        contador_generaciones += 1
        print(f"Gen {contador_generaciones} | Mejor Fitness: {int(max(scores))}")
        
        if contador_generaciones >= 200: MODO_ENTRENAMIENTO_RAPIDO = False
        array_poblacion_red = ag.redes_genetico(scores, array_poblacion_red)
        for c in coches: c.reset()

if __name__ == "__main__":
    main()