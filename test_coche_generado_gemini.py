import pygame
import math
import random
from Ometecu import Ometecu
from baseGenetico import AlgoritmoGenetico

# --- Configuración ---
TAM_POBLACION = 40
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
    red.set_config_red(capa_inicial=11, capa_intermedia=10, capa_final=2)
    red.funcionActivacion("r", "r", "r")
    red.set_memoria_genetico([random.uniform(-1, 1) for _ in red.get_memoria()])
    array_poblacion_red.append(red)

ag = AlgoritmoGenetico(tam_poblacion=TAM_POBLACION, tasa_mutacion=0.2, num_generaciones=1)

try:
    array_poblacion_red[0].set_memoria_genetico(ag.obtener_memoria("coche"))
except:
    pass
# --- PISTA CON CURVAS (Camino central) ---
PISTA_LINEA_CENTRAL = [
    (150, 350), (200, 200), (400, 150), (550, 300), 
    (700, 450), (850, 400), (900, 250), (750, 150),
    (550, 200), (400, 450), (250, 550), (120, 500)
]

def dibujar_pista_en_superficie(superficie):
    superficie.fill((0, 0, 0)) 
    if len(PISTA_LINEA_CENTRAL) > 2:
        pygame.draw.lines(superficie, (100, 100, 100), True, PISTA_LINEA_CENTRAL, 110)
        pygame.draw.lines(superficie, (100, 100, 100), True, PISTA_LINEA_CENTRAL, 1)

dibujar_pista_en_superficie(pista_background)

class CocheIA:
    def __init__(self, index):
        self.index_red = index
        self.reset()

    def reset(self):
        self.x, self.y = PISTA_LINEA_CENTRAL[0][0], PISTA_LINEA_CENTRAL[0][1]
        self.angulo_coche = 315 
        self.velocidad = 2.0  # Empieza con impulso inicial
        self.vivo = True
        self.fitness = 0
        self.checkpoint_actual = 0
        self.tiempo_sin_progreso = 0
        
        sig_x, sig_y = PISTA_LINEA_CENTRAL[1]
        self.mejor_distancia_al_checkpoint = math.hypot(sig_x - self.x, sig_y - self.y)

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

        # Entradas y predicción
        sensores = [self.cast_ray(-90), self.cast_ray(-45), self.cast_ray(0), self.cast_ray(45), self.cast_ray(90)]
        red = array_poblacion_red[self.index_red]
        red.set_entradas(sensores)
        red.prediccion_old()
        
        acel, giro = red.salidas
        
        # --- SOLUCIÓN 1: VELOCIDAD MÍNIMA OBLIGATORIA ---
        # Forzamos un piso de velocidad (2.0) y un techo de (6.0) para que no puedan quedarse parados
        self.velocidad = max(2.0, min(self.velocidad + acel * 0.5, 6)) * 0.96
        
        self.angulo_coche += giro * 6 # Un poco más de sensibilidad de giro para las curvas cerradas
        self.x += math.cos(math.radians(self.angulo_coche)) * self.velocidad
        self.y += math.sin(math.radians(self.angulo_coche)) * self.velocidad

        # --- LÓGICA DE RECOMPENSAS AJUSTADA ---
        siguiente_idx = (self.checkpoint_actual + 1) % len(PISTA_LINEA_CENTRAL)
        target_x, target_y = PISTA_LINEA_CENTRAL[siguiente_idx]
        distancia_actual = math.hypot(target_x - self.x, target_y - self.y)
        
        if distancia_actual < self.mejor_distancia_al_checkpoint:
            # --- SOLUCIÓN 2: PREMIO POR VELOCIDAD AL AVANZAR ---
            # Si avanza hacia adelante, le damos más puntos si va RÁPIDO
            self.fitness += 1 + (self.velocidad * 0.5) 
            self.mejor_distancia_al_checkpoint = distancia_actual
            self.tiempo_sin_progreso = 0
        else:
            # Castigo moderado por retroceder (reducido para no infundir miedo, pero sigue castigando)
            self.fitness -= 5  
            self.tiempo_sin_progreso += 1

        # Checkpoint alcanzado (Curva superada)
        if distancia_actual < 65: 
            self.checkpoint_actual = siguiente_idx
            self.fitness += 2000  # Recompensa masiva por pasar de zona
            self.tiempo_sin_progreso = 0
            
            nuevo_sig_idx = (self.checkpoint_actual + 1) % len(PISTA_LINEA_CENTRAL)
            nx, ny = PISTA_LINEA_CENTRAL[nuevo_sig_idx]
            self.mejor_distancia_al_checkpoint = math.hypot(nx - self.x, ny - self.y)

        # --- SOLUCIÓN 3: CONTROL DE ESTANCAMIENTO ---
        # Si en 150 frames (aprox 2.5 segundos) no ha cruzado al siguiente checkpoint, muere.
        # Esto elimina a los que dan vueltas en círculos pequeños o van extremadamente lento.
        if self.tiempo_sin_progreso > 150: 
            self.vivo = False
            self.fitness -= 100 

        # Colisión pared
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
                for p in PISTA_LINEA_CENTRAL:
                    pygame.draw.circle(pantalla, (0, 0, 255), p, 6)
                for c in coches: c.dibujar()
                pygame.display.flip()
                reloj.tick(60)
            else: reloj.tick(0)

        scores = [max(0, c.fitness) for c in coches]
        contador_generaciones += 1
        print(f"Gen {contador_generaciones} | Mejor Fitness: {int(max(scores))}")
        
        # Dejamos 35 generaciones en modo rápido para que tengan tiempo de mutar velocidad
        if contador_generaciones >= 300: 
            MODO_ENTRENAMIENTO_RAPIDO = False
            ag.guardar_memoria("coche")
        array_poblacion_red = ag.redes_genetico(scores, array_poblacion_red)
        for c in coches: c.reset()

if __name__ == "__main__":
    main()