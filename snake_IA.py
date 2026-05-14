import pygame
import time
import random
from Ometecu import Ometecu
from baseGenetico import AlgoritmoGenetico

# Inicializar pygame
pygame.init()

# --- GENETICO ---
TAM_POBLACION_NEURONAL = 12
NUM_GENERACIONES = 1
global array_poblacion_red
array_poblacion_red = []
poblacion_con_fitness = []
contador_ciclos = 0
global fitnees_score
fitnees_score = []
tiempo_vivo = 0.1
contador_generaciones = 0

# --- Configuración Red Neuronal ---
for x in range(TAM_POBLACION_NEURONAL):
    red_neuronal = Ometecu()
    red_neuronal.set_config_red(capa_inicial=21, capa_intermedia=10, capa_final=4)
    red_neuronal.funcionActivacion("s", "r", "r")
    valor_aleatoreo = [round(random.uniform(-0.9, 0.9), 10) for elem in red_neuronal.get_memoria()]
    red_neuronal.set_memoria_genetico(valor_aleatoreo)
    array_poblacion_red.append(red_neuronal)

ag = AlgoritmoGenetico(tam_poblacion=len(array_poblacion_red), 
                       longitud_genoma=len(array_poblacion_red[0].get_memoria()) + 1, 
                       tasa_mutacion=0.01, num_generaciones=NUM_GENERACIONES)

# --- DEFINICIÓN DE COLORES ---
BLANCO = (255, 255, 255)
AMARILLO = (255, 255, 102)
NEGRO = (0, 0, 0)
ROJO = (213, 50, 80)
VERDE = (0, 255, 0)
AZUL = (50, 153, 213)
GRIS_CLARO = (150, 170, 200)

# --- CONFIGURACIÓN DE LA PANTALLA ---
ANCHO_PANTALLA = 600
ALTO_PANTALLA = 400
pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
pygame.display.set_caption('Víbora IA: Control Automático y Genético')

# --- VARIABLES DEL JUEGO ---
reloj = pygame.time.Clock()
tamano_bloque = 10
velocidad_vibora = 300

# --- FUENTES DE TEXTO ---
fuente_mensaje = pygame.font.SysFont("bahnschrift", 20)
fuente_puntos = pygame.font.SysFont("comicsansms", 25)
fuente_ia = pygame.font.SysFont("consolas", 12)

def algoritmo_genetico(fitnees_score):
    # ag.set_poblacion_fitnees(fitnees_score, array_poblacion_red)
    # ag.ejecutar()
    # for index, red in enumerate(ag.get_poblacion()):
    #     array_poblacion_red[index].set_memoria_genetico(red[:-1])
    global array_poblacion_red
    array_poblacion_red = ag.redes_genetico(fitnees_score, array_poblacion_red)

def mostrar_puntuacion(puntos):
    texto_puntos = fuente_puntos.render("Puntos: " + str(puntos), True, AMARILLO)
    pantalla.blit(texto_puntos, [0, 0])

def dibujar_vibora(tamano_bloque, lista_vibora):
    for bloque in lista_vibora:
        pygame.draw.rect(pantalla, VERDE, [bloque[0], bloque[1], tamano_bloque, tamano_bloque])

def generar_comida():
    comida_x = round(random.randrange(0, ANCHO_PANTALLA - tamano_bloque) / 10.0) * 10.0
    comida_y = round(random.randrange(0, ALTO_PANTALLA - tamano_bloque) / 10.0) * 10.0
    return comida_x, comida_y

def leer_estado_ia(cabeza_x, cabeza_y, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida, pantalla):
    direcciones = [
        (0, -tamano_bloque), (tamano_bloque, -tamano_bloque), 
        (tamano_bloque, 0), (tamano_bloque, tamano_bloque),
        (0, tamano_bloque), (-tamano_bloque, tamano_bloque), 
        (-tamano_bloque, 0), (-tamano_bloque, -tamano_bloque)
    ]
    
    vision_comida = [0.0] * 8
    vision_paredes = [0.0] * 8
    
    for i, (dx, dy) in enumerate(direcciones):
        x_actual = cabeza_x
        y_actual = cabeza_y
        distancia = 0
        comida_detectada = False
        
        while True:
            x_actual += dx
            y_actual += dy
            distancia += 1
            
            if x_actual < 0 or x_actual >= ANCHO_PANTALLA or y_actual < 0 or y_actual >= ALTO_PANTALLA:
                vision_paredes[i] = 1.0 / distancia
                break 
            
            if not comida_detectada and x_actual == comida_x and y_actual == comida_y:
                vision_comida[i] = 1.0 / distancia
                comida_detectada = True

    dir_arriba = 1 if y_cambio == -tamano_bloque else 0
    dir_abajo = 1 if y_cambio == tamano_bloque else 0
    dir_izquierda = 1 if x_cambio == -tamano_bloque else 0
    dir_derecha = 1 if x_cambio == tamano_bloque else 0
    
    direccion_actual = [dir_arriba, dir_abajo, dir_izquierda, dir_derecha]

    estado_completo = vision_comida + vision_paredes + direccion_actual + [comida_obtenida]
    return estado_completo

def red_neuronal_simulada(estado, index):
    red_neuronal = array_poblacion_red[index]
    red_neuronal.set_entradas(estado)
    red_neuronal.prediccion_old()
    salida_ia = red_neuronal.salidas  
    return salida_ia

def procesar_accion_ia(salida_ia, x_cambio_actual, y_cambio_actual, tamano_bloque):
    decision = salida_ia.index(max(salida_ia))
    x_nuevo = x_cambio_actual
    y_nuevo = y_cambio_actual

    if decision == 0 and y_cambio_actual == 0:     # Arriba
        x_nuevo, y_nuevo = 0, -tamano_bloque
    elif decision == 1 and y_cambio_actual == 0:   # Abajo
        x_nuevo, y_nuevo = 0, tamano_bloque
    elif decision == 2 and x_cambio_actual == 0:   # Izquierda
        x_nuevo, y_nuevo = -tamano_bloque, 0
    elif decision == 3 and x_cambio_actual == 0:   # Derecha
        x_nuevo, y_nuevo = tamano_bloque, 0

    return x_nuevo, y_nuevo, decision


# ==========================================
#          VARIABLES GLOBALES
# ==========================================
global index_redes, tiempo_con_vida, comida_obtenida, modo_ia_activado_error
index_redes = 0
tiempo_con_vida = 0
comida_obtenida = 0
modo_ia_activado_error = 2

# ==========================================
#          BÚCLE PRINCIPAL
# ==========================================

def juego_principal():
    global index_redes, fitnees_score, array_poblacion_red, tiempo_con_vida, comida_obtenida, modo_ia_activado_error

    juego_terminado = False
    juego_cerrado = False

    x_actual = ANCHO_PANTALLA / 2
    y_actual = ALTO_PANTALLA / 2
    x_cambio = tamano_bloque 
    y_cambio = 0

    lista_vibora = []
    largo_vibora = 1

    comida_x, comida_y = generar_comida()
    modo_ia_activado = True 
    
    while not juego_terminado:

        while juego_cerrado:
            # 1. LEER EL TECLADO PRIMERO (Freno de emergencia y reinicio limpio)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    return False # Cierra el juego de forma limpia
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_q:
                        return False 
                    if evento.key == pygame.K_s: 
                        modo_ia_activado = False 
                    if evento.key == pygame.K_c and not modo_ia_activado:
                        return True # Retorna True para que el bucle exterior reinicie la partida

            global modo_ia_activado_error
            modo_ia_activado_error = 0
            # 2. AUTO-REINICIO DE LA IA LIMPÍSIMO
            if modo_ia_activado:
                return True

            # 3. PANTALLA DE GAME OVER
            pantalla.fill(NEGRO)
            texto = fuente_mensaje.render("Perdiste! 'C' jugar, 'Q' salir, 'S' apaga IA", True, ROJO)
            pantalla.blit(texto, (ANCHO_PANTALLA/2 - 190, ALTO_PANTALLA/2))
            mostrar_puntuacion(largo_vibora - 1)
            pygame.display.update()
            

        # --- LECTURA DE EVENTOS (Durante el juego) ---
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_q:
                    return False
                if evento.key == pygame.K_s:
                    modo_ia_activado_error = 1 
                    
                
        # --- LÓGICA DE CONTROL ---
        if modo_ia_activado_error == 0:
            estado_actual = leer_estado_ia(x_actual, y_actual, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida, pantalla)
            
            salida_ia = red_neuronal_simulada(estado_actual, index_redes)

            x_cambio, y_cambio, decision_tomada = procesar_accion_ia(salida_ia, x_cambio, y_cambio, tamano_bloque)

        
        tiempo_con_vida += 0.002
        
        # --- ACTUALIZACIÓN DE POSICIÓN ---
        x_actual += x_cambio
        y_actual += y_cambio

        cabeza_vibora = [x_actual, y_actual]
        lista_vibora.append(cabeza_vibora)

        if len(lista_vibora) > largo_vibora:
            del lista_vibora[0]

        # --- FÍSICAS Y COLISIONES (Modificado para no perder el Fitness) ---
        muerto = False
        if x_actual >= ANCHO_PANTALLA or x_actual < 0 or y_actual >= ALTO_PANTALLA or y_actual < 0:
            muerto = True
        
        for segmento in lista_vibora[:-1]:
            if segmento == cabeza_vibora:
                muerto = Truex

        if muerto:
            # Ahora la puntuación se guarda correctamente sin importar de qué muera
            index_redes += 1
            fitnees_score.append((largo_vibora * 100 ) + tiempo_con_vida )  # + tiempo_con_vida
            #print(largo_vibora)
            
            if index_redes >= TAM_POBLACION_NEURONAL:
                algoritmo_genetico(fitnees_score)
                print(fitnees_score)
                fitnees_score = []
                index_redes = 0
                tiempo_con_vida = 0
                
            juego_cerrado = True

        # --- DIBUJADO ---
        if not juego_cerrado:
            pantalla.fill(NEGRO)
            pygame.draw.rect(pantalla, AMARILLO, [comida_x, comida_y, tamano_bloque, tamano_bloque])
            dibujar_vibora(tamano_bloque, lista_vibora)
            
            if modo_ia_activado:
                try:
                    texto_estado = fuente_ia.render(f"Decisión IA (0=Ar,1=Ab,2=Iz,3=De): {decision_tomada}", True, AMARILLO)
                except:
                    texto_estado = fuente_ia.render(f"Decisión IA (0=Ar,1=Ab,2=Iz,3=De): ", True, AMARILLO)
                texto_red = fuente_ia.render(f"Red en entrenamiento: {index_redes + 1}/{TAM_POBLACION_NEURONAL}", True, AMARILLO)
                pantalla.blit(texto_estado, [5, 40])
                pantalla.blit(texto_red, [5, 55])

            mostrar_puntuacion(largo_vibora - 1)
            pygame.display.update()

        # --- LÓGICA DE COMIDA ---
        if x_actual == comida_x and y_actual == comida_y:
            comida_obtenida = 1
            comida_x, comida_y = generar_comida()
            largo_vibora += 1
        else:
            comida_obtenida = 0
            
        reloj.tick(velocidad_vibora)
        
    return False # Si sale del bucle 'while not juego_terminado', cierra el juego

# ==========================================
#     EL BUCLE QUE EVITA EL RECURSION ERROR
# ==========================================
if __name__ == "__main__":
    jugando = True
    # Mientras la función juego_principal() devuelva True, se iniciará una partida limpia.
    # Si devuelve False (cuando aprietas la 'Q' o cierras la ventana), el bucle se rompe.
    while jugando:
        jugando = juego_principal()
        
    pygame.quit()
    quit()