import pygame
import time
import random
import numpy as np
from Ometecu import Ometecu

# --- Configuración Inicial ---
snake_speed = 300  # Velocidad
window_x = 800
window_y = 400

# Colores
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# Inicialización Pygame
pygame.init()
pygame.display.set_caption('Serpiente IA con Lógica de Supervivencia')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()

# --- Variables Globales ---
MAX_FRUTAS = 20
lista_frutas = []

snake_position = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
direction = 'RIGHT'
evento_teclado = ''
score = 0
juego_terminado = False
fruta_comida_ne = 0

# --- Configuración Red Neuronal ---
red_neuronal = Ometecu()
# Asegúrate de que estas capas coincidan con tu Ometecu.py
red_neuronal.set_config_red(capa_inicial=6, capa_intermedia=6, capa_final=5)


# --- Funciones Auxiliares ---

def generar_nueva_fruta():
    """Genera una nueva fruta en una posición aleatoria (grid de 10x10)."""
    nueva_fruta = [random.randrange(1, (window_x // 10)) * 10,
                   random.randrange(1, (window_y // 10)) * 10]
    lista_frutas.append(nueva_fruta)


def reiniciar_juego():
    """Resetea el juego tras un choque."""
    global snake_position, snake_body, direction, score, lista_frutas, juego_terminado
    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    direction = 'RIGHT'
    score = 0
    juego_terminado = False

    lista_frutas = []
    for _ in range(MAX_FRUTAS):
        generar_nueva_fruta()


def hay_colision(point, body):
    """Retorna un valor de peligro si el punto choca con pared o cuerpo."""
    # 0.4 para paredes, 0.9 para cuerpo (como en tu lógica original)
    if point[0] >= window_x or point[0] < 0 or point[1] >= window_y or point[1] < 0:
        return 0.4
    if point in body[1:]:
        return 0.9
    return False


def show_score(color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Puntuación : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)


# --- LÓGICA DE ESTADO (INPUTS) ---
def obtener_estado_del_juego(snake_head, snake_body, lista_de_frutas, direction, fruta_comida_ne):
    # 1. Buscar la fruta más cercana
    closest_fruit = None
    min_dist = float('inf')

    if not lista_de_frutas:
        return np.zeros(15, dtype=float)  # Retorno de seguridad si no hay frutas

    for fruta in lista_de_frutas:
        dist = abs(snake_head[0] - fruta[0]) + abs(snake_head[1] - fruta[1])
        if dist < min_dist:
            min_dist = dist
            closest_fruit = fruta

    # 2. Sensores de entorno inmediato
    point_l = [snake_head[0] - 10, snake_head[1]]
    point_r = [snake_head[0] + 10, snake_head[1]]
    point_u = [snake_head[0], snake_head[1] - 10]
    point_d = [snake_head[0], snake_head[1] + 10]

    dir_l = (direction == 'LEFT')
    dir_r = (direction == 'RIGHT')
    dir_u = (direction == 'UP')
    dir_d = (direction == 'DOWN')

    # Estado[0]: Peligro Frente
    # Estado[1]: Peligro Derecha Relativa
    # Estado[2]: Peligro Izquierda Relativa

    estado = [
        # Peligro Frente
        (dir_r and hay_colision(point_r, snake_body)) or
        (dir_l and hay_colision(point_l, snake_body)) or
        (dir_u and hay_colision(point_u, snake_body)) or
        (dir_d and hay_colision(point_d, snake_body)),

        # Peligro Derecha (Relativa)
        (dir_u and hay_colision(point_r, snake_body)) or
        (dir_d and hay_colision(point_l, snake_body)) or
        (dir_l and hay_colision(point_u, snake_body)) or
        (dir_r and hay_colision(point_d, snake_body)),

        # Peligro Izquierda (Relativa)
        (dir_d and hay_colision(point_r, snake_body)) or
        (dir_u and hay_colision(point_l, snake_body)) or
        (dir_r and hay_colision(point_u, snake_body)) or
        (dir_l and hay_colision(point_d, snake_body)),

        # Direccion actual
        dir_l, dir_r, dir_u, dir_d,

        # Longitud normalizada
        (len(snake_body) / 1000),

        # Posición cabeza normalizada
        (snake_head[0] / 1000), (snake_head[1] / 1000),

        # Ubicación comida (One-Hot logic)
        closest_fruit[0] < snake_head[0],  # Comida a la Izq
        closest_fruit[0] > snake_head[0],  # Comida a la Der
        closest_fruit[1] < snake_head[1],  # Comida Arriba
        closest_fruit[1] > snake_head[1],  # Comida Abajo

        fruta_comida_ne
    ]

    # Convertimos booleanos a float (0.0 o 1.0) y retornamos numpy array
    return np.array([float(x) for x in estado], dtype=float)


# --- LÓGICA NEURONAL (DECISIÓN Y ENTRENAMIENTO) ---
def reglas_neuronas(estado_actual):
    red_neuronal.set_entradas(estado_actual)

    # === AQUI ESTA LA MAGIA: Calculamos qué DEBERIA aprender ===
    # Formato de salida esperado: [LEFT, RIGHT, UP, DOWN, EXTRA]

    # 1. ¿A dónde sugiere ir la comida?
    # Indices en estado_actual: -5:Izq, -4:Der, -3:Arr, -2:Abj
    target = [0, 0, 0, 0, 0.8]

    if estado_actual[-5]: target[0] = 1  # Sugerir LEFT
    if estado_actual[-4]: target[1] = 1  # Sugerir RIGHT
    if estado_actual[-3]: target[2] = 1  # Sugerir UP
    if estado_actual[-2]: target[3] = 1  # Sugerir DOWN

    # 2. ¿Qué peligros existen? (Valores > 0 indican peligro)
    peligro_frente = estado_actual[0]
    peligro_der = estado_actual[1]
    peligro_izq = estado_actual[2]

    # 3. ¿Hacia dónde miramos?
    voy_izq = estado_actual[3]
    voy_der = estado_actual[4]
    voy_arr = estado_actual[5]
    voy_abj = estado_actual[6]

    # 4. FILTRO DE SUPERVIVENCIA:
    # Si la dirección sugerida por la comida choca con un peligro, la apagamos (target=0)

    # -- Analisis de Peligro FRENTE --
    if peligro_frente > 0:
        if voy_izq: target[0] = 0  # No sigas yendo a la izq
        if voy_der: target[1] = 0  # No sigas yendo a la der
        if voy_arr: target[2] = 0  # No sigas yendo arriba
        if voy_abj: target[3] = 0  # No sigas yendo abajo

    # -- Analisis de Peligro DERECHA RELATIVA --
    if peligro_der > 0:
        if voy_arr: target[1] = 0  # Si voy arriba, mi derecha es RIGHT. No ir RIGHT.
        if voy_abj: target[0] = 0  # Si voy abajo, mi derecha es LEFT. No ir LEFT.
        if voy_izq: target[2] = 0  # Si voy izq, mi derecha es UP. No ir UP.
        if voy_der: target[3] = 0  # Si voy der, mi derecha es DOWN. No ir DOWN.

    # -- Analisis de Peligro IZQUIERDA RELATIVA --
    if peligro_izq > 0:
        if voy_arr: target[0] = 0  # Si voy arriba, mi izq es LEFT.
        if voy_abj: target[1] = 0  # Si voy abajo, mi izq es RIGHT.
        if voy_izq: target[3] = 0  # Si voy izq, mi izq es DOWN.
        if voy_der: target[2] = 0  # Si voy der, mi izq es UP.

    # 5. Entrenamos con la decisión corregida (segura)
    red_neuronal.set_valor_aprender(target)
    red_neuronal.entrenamiento()

    # 6. Obtenemos la predicción real de la red
    salida = red_neuronal.salidas  # Usamos la propiedad, asumiendo que Ometecu la actualiza tras entrenamiento

    direction_aux = ''
    if salida[0] > 0.5: direction_aux = 'LEFT'
    if salida[1] > 0.5: direction_aux = 'RIGHT'
    if salida[2] > 0.5: direction_aux = 'UP'
    if salida[3] > 0.5: direction_aux = 'DOWN'

    return direction_aux


# --- MAIN LOOP ---
reiniciar_juego()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        # Control manual opcional
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                evento_teclado = 'UP'
            elif event.key == pygame.K_DOWN:
                evento_teclado = 'DOWN'
            elif event.key == pygame.K_LEFT:
                evento_teclado = 'LEFT'
            elif event.key == pygame.K_RIGHT:
                evento_teclado = 'RIGHT'

    # Lógica del juego
    estado_actual = obtener_estado_del_juego(snake_position, snake_body, lista_frutas, direction, fruta_comida_ne)

    # IA decide dirección
    direction_IA = reglas_neuronas(estado_actual)

    if fruta_comida_ne == 0.8:
        fruta_comida_ne = 0.0

    # Prioridad: Teclado > IA
    if evento_teclado != "":
        direction = evento_teclado
        evento_teclado = ""
    elif direction_IA != "":
        direction = direction_IA

    # Movimiento
    if direction == 'UP':    snake_position[1] -= 10
    if direction == 'DOWN':  snake_position[1] += 10
    if direction == 'LEFT':  snake_position[0] -= 10
    if direction == 'RIGHT': snake_position[0] += 10

    # Crecimiento
    snake_body.insert(0, list(snake_position))

    fruta_comida = None
    for fruta in lista_frutas:
        # Chequeo simple de colisión con fruta
        if snake_position[0] == fruta[0] and snake_position[1] == fruta[1]:
            fruta_comida = fruta
            break

    if fruta_comida:
        score += 10
        fruta_comida_ne = 0.8
        lista_frutas.remove(fruta_comida)
        generar_nueva_fruta()
    else:
        snake_body.pop()

    # Dibujado
    game_window.fill(black)

    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

    for fruta_pos in lista_frutas:
        pygame.draw.rect(game_window, white, pygame.Rect(fruta_pos[0], fruta_pos[1], 10, 10))

    # Chequeo de Muerte
    if hay_colision(snake_position, snake_body):
        reiniciar_juego()

    show_score(white, 'times new roman', 20)
    pygame.display.update()

    fps.tick(snake_speed)