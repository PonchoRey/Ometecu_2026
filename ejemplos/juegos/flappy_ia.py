import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pygame
import random
import math
from ometecu.Ometecu import Ometecu
from ometecu.baseGenetico import AlgoritmoGenetico

# --- Configuración ---
TAM_POBLACION = 30
ANCHO, ALTO = 400, 600
pygame.init()
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Flappy Bird IA - Ometecu")

reloj = pygame.time.Clock()
fuente = pygame.font.SysFont("consolas", 16)

global MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones
contador_generaciones = 0
MODO_ENTRENAMIENTO_RAPIDO = True

# --- Redes Ometecu ---
array_poblacion_red = []
for _ in range(TAM_POBLACION):
    red = Ometecu()
    red.inicio_synapsis("flappy")
    red.set_config_red(capa_inicial=12, capa_intermedia=12, capa_final=2)
    red.funcionActivacion("r", "r", "s")
    red.set_memoria_genetico([random.uniform(-1, 1) for _ in red.get_memoria()])
    array_poblacion_red.append(red)

ag = AlgoritmoGenetico(tam_poblacion=TAM_POBLACION, tasa_mutacion=0.03, num_generaciones=1)

try:
    # Intentar cargar sinapsis previa si existe
    for i in range(TAM_POBLACION - 5):
        array_poblacion_red[i].set_memoria_genetico(ag.obtener_memoria("flappy"))
    print("Sinapsis de Flappy cargada con éxito!")
except Exception as e:
    print("No se encontró sinapsis previa de Flappy, iniciando desde cero.")

# ==========================================
#      CONTROLADOR CENTRAL DE IA Y EVOLUCIÓN
# ==========================================
def controlar_ia(pajaro=None, tubo=None, evento="paso", lista_pajaros=None):
    global array_poblacion_red, MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones

    if evento == "paso":
        # 1. Recompensa por supervivencia (un frame más vivo)
        pajaro.fitness += 1

        # 2. Obtener y normalizar las entradas para la red neuronal
        tubo_dist_x = (tubo.x - pajaro.x) / ANCHO
        tubo_y_top = tubo.alto / ALTO
        tubo_y_bottom = (tubo.alto + tubo.espacio) / ALTO
        pajaro_y = pajaro.y / ALTO
        pajaro_vel_y = pajaro.velocidad / 10.0

        entradas = [pajaro_y, pajaro_vel_y, tubo_dist_x, tubo_y_top, tubo_y_bottom]

        # 3. Predicción de la red neuronal correspondiente al pájaro
        red = array_poblacion_red[pajaro.index_red]
        red.set_entradas(entradas)
        red.prediccion_old()

        # 4. Tomar decisión (si la salida de saltar es mayor que la de no saltar, aletear)
        saltar, no_saltar = red.salidas
        if saltar > no_saltar:
            pajaro.velocidad = pajaro.salto

    elif evento == "choque":
        # Penalización por colisionar contra un tubo o los límites
        pajaro.fitness -= 20

    elif evento == "punto":
        # Recompensa masiva por superar con éxito un tubo
        pajaro.fitness += 2000

    elif evento == "evolucionar":
        # Ejecutar el proceso del algoritmo genético al finalizar la generación
        scores = [max(1, p.fitness) for p in lista_pajaros]
        score_maximo = max(p.score for p in lista_pajaros)

        # Desactivar el entrenamiento rápido para pasar a visual
        if contador_generaciones == 100:
            MODO_ENTRENAMIENTO_RAPIDO = False

        # Aplicar cruzamiento y mutación para obtener la nueva población
        array_poblacion_red = ag.redes_genetico(scores, array_poblacion_red)

        # Guardar memoria/sinapsis si se alcanza una buena puntuación o cada 50 generaciones
        if contador_generaciones % 10 == 0 or score_maximo > 10:
            ag.guardar_memoria("flappy")

# --- Clases de Juego ---
class PajaroIA:
    def __init__(self, index):
        self.index_red = index
        self.reset()

    def reset(self):
        self.x = 80
        self.y = ALTO // 2
        self.velocidad = 0
        self.gravedad = 0.4
        self.salto = -6.5
        self.vivo = True
        self.fitness = 0
        self.score = 0

    def actualizar(self, proximo_tubo):
        if not self.vivo:
            return

        # Aplicar gravedad y actualizar posición física
        self.velocidad += self.gravedad
        if self.velocidad > 10:
            self.velocidad = 10
        self.y += self.velocidad

        # Procesar entradas, predicción y acción de salto a través del controlador único
        controlar_ia(self, proximo_tubo, evento="paso")

        # Verificar límites de la pantalla
        if self.y >= ALTO - 15 or self.y <= 0:
            self.vivo = False
            controlar_ia(self, evento="choque")

    def dibujar(self):
        if self.vivo:
            # Pájaro verde
            pygame.draw.circle(pantalla, (0, 255, 0), (int(self.x), int(self.y)), 12)
            pygame.draw.circle(pantalla, (255, 255, 255), (int(self.x) + 4, int(self.y) - 4), 4)
            pygame.draw.circle(pantalla, (0, 0, 0), (int(self.x) + 5, int(self.y) - 4), 1.5)

class Tubo:
    def __init__(self, x):
        self.x = x
        self.espacio = 140
        self.ancho = 60
        self.alto = random.randint(50, ALTO - self.espacio - 50)
        self.velocidad = 3
        self.pasado = False

    def actualizar(self):
        self.x -= self.velocidad

    def colisiona_con(self, pajaro):
        if pajaro.x + 12 > self.x and pajaro.x - 12 < self.x + self.ancho:
            if pajaro.y - 12 < self.alto or pajaro.y + 12 > self.alto + self.espacio:
                return True
        return False

    def dibujar(self):
        # Tubo superior
        pygame.draw.rect(pantalla, (50, 180, 50), (self.x, 0, self.ancho, self.alto))
        pygame.draw.rect(pantalla, (30, 120, 30), (self.x - 4, self.alto - 15, self.ancho + 8, 15))
        # Tubo inferior
        pygame.draw.rect(pantalla, (50, 180, 50), (self.x, self.alto + self.espacio, self.ancho, ALTO - (self.alto + self.espacio)))
        pygame.draw.rect(pantalla, (30, 120, 30), (self.x - 4, self.alto + self.espacio, self.ancho + 8, 15))

# --- Bucle Principal ---
def main():
    global MODO_ENTRENAMIENTO_RAPIDO, contador_generaciones
    pajaros = [PajaroIA(i) for i in range(TAM_POBLACION)]
    tubos = [Tubo(ANCHO + 100)]
    score_maximo = 0

    while True:
        corriendo = True
        while corriendo:
            if not MODO_ENTRENAMIENTO_RAPIDO:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_m:
                            MODO_ENTRENAMIENTO_RAPIDO = not MODO_ENTRENAMIENTO_RAPIDO

            # Encontrar el tubo objetivo más cercano
            proximo_tubo = None
            for t in tubos:
                if t.x + t.ancho > 50:
                    proximo_tubo = t
                    break
            if proximo_tubo is None:
                proximo_tubo = tubos[0]

            # Actualizar física y decisiones de los pájaros
            vivos = 0
            for p in pajaros:
                if p.vivo:
                    p.actualizar(proximo_tubo)
                    if proximo_tubo.colisiona_con(p):
                        p.vivo = False
                        controlar_ia(p, evento="choque")
                    else:
                        vivos += 1

            if vivos == 0:
                corriendo = False
                break

            # Actualizar tubos y otorgar puntos
            for t in tubos:
                t.actualizar()
                if not t.pasado and t.x + t.ancho < 80:
                    t.pasado = True
                    for p in pajaros:
                        if p.vivo:
                            p.score += 1
                            controlar_ia(p, evento="punto")
                            if p.score > score_maximo:
                                score_maximo = p.score

            # Mantenimiento de la cola de tubos
            if tubos[0].x < -tubos[0].ancho:
                tubos.pop(0)

            if tubos[-1].x < ANCHO - 220:
                tubos.append(Tubo(ANCHO))

            # Renderizado en pantalla
            if not MODO_ENTRENAMIENTO_RAPIDO:
                pantalla.fill((112, 197, 206))
                for t in tubos:
                    t.dibujar()
                for p in pajaros:
                    p.dibujar()

                # Textos de estado
                txt_gen = fuente.render(f"Gen: {contador_generaciones + 1}", True, (255, 255, 255))
                txt_vivos = fuente.render(f"Vivos: {vivos}/{TAM_POBLACION}", True, (255, 255, 255))
                txt_score = fuente.render(f"Max Score: {score_maximo}", True, (255, 255, 255))
                txt_modo = fuente.render("[M] Alternar modo rapido", True, (0, 0, 0))

                pantalla.blit(txt_gen, (10, 10))
                pantalla.blit(txt_vivos, (10, 30))
                pantalla.blit(txt_score, (10, 50))
                pantalla.blit(txt_modo, (10, ALTO - 30))

                pygame.display.flip()
                reloj.tick(60)
            else:
                reloj.tick(0)

        # Fin de la generación: procesar evolución a través del controlador central
        contador_generaciones += 1
        mejor_fit = int(max(p.fitness for p in pajaros))
        print(f"Gen {contador_generaciones} | Mejor Fitness: {mejor_fit} | Score Max: {score_maximo}")

        controlar_ia(evento="evolucionar", lista_pajaros=pajaros)
        
        # Reiniciar variables del juego para la nueva generación
        pajaros = [PajaroIA(i) for i in range(TAM_POBLACION)]
        tubos = [Tubo(ANCHO + 100)]

if __name__ == "__main__":
    main()
