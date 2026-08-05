import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pygame
import time
import random
import math
import multiprocessing
from ometecu.Ometecu import Ometecu
from ometecu.baseGenetico import AlgoritmoGenetico

# Inicializar pygame
pygame.init()

# ==========================================
#         CONFIGURACIÓN DEL GENÉTICO
# ==========================================
TAM_POBLACION_NEURONAL = 6  
NUM_GENERACIONES = 1

global array_poblacion_red, contador_generaciones, MODO_ENTRENAMIENTO_RAPIDO
array_poblacion_red = []
contador_generaciones = 0
MODO_ENTRENAMIENTO_RAPIDO = True  

# --- Configuración Red Neuronal ---
def inicializar_redes():
    global array_poblacion_red, ag
    for x in range(TAM_POBLACION_NEURONAL):
        red_neuronal = Ometecu()
        red_neuronal.inicio_synapsis("snake")
        # 31 entradas: 24 vision (8 pared, 8 comida, 8 cuerpo) + 4 dir_actual + 1 obtenida + 2 orientación
        red_neuronal.set_config_red(capa_inicial=31, capa_intermedia=20, capa_final=4)
        red_neuronal.funcionActivacion("r", "r", "r")
        valor_aleatorio = [random.uniform(-0.9, 0.9) for _ in red_neuronal.get_memoria()]
        red_neuronal.set_memoria_genetico(valor_aleatorio)
        array_poblacion_red.append(red_neuronal)

    ag = AlgoritmoGenetico(
        tam_poblacion=len(array_poblacion_red), 
        tasa_mutacion=0.04,  
        num_generaciones=NUM_GENERACIONES
    )
    try: 
        memoria_cargada = ag.obtener_memoria("snake")
        for i in range(TAM_POBLACION_NEURONAL // 2):
            array_poblacion_red[i].set_memoria_genetico(memoria_cargada)
        print("synapsis cargada!")
    except Exception as e:
        print("Error al cargar la synpasis:", e)

# ==========================================
#         CONFIGURACIÓN GRÁFICA
# ==========================================
BLANCO = (255, 255, 255)
AMARILLO = (255, 255, 102)
NEGRO = (0, 0, 0)
ROJO = (213, 50, 80)
VERDE = (0, 255, 0)

ANCHO_PANTALLA = 600
ALTO_PANTALLA = 400

tamano_bloque = 10
fuente_mensaje = pygame.font.SysFont("bahnschrift", 20)
fuente_puntos = pygame.font.SysFont("comicsansms", 25)
fuente_ia = pygame.font.SysFont("consolas", 12)

# ==========================================
#             LÓGICA AUXILIAR
# ==========================================
def algoritmo_genetico_ciclo(scores):
    global array_poblacion_red, contador_generaciones, MODO_ENTRENAMIENTO_RAPIDO
    array_poblacion_red = ag.redes_genetico(scores, array_poblacion_red)
    #ag.guardar_memoria("snake")
    
    contador_generaciones += 1
    print(f"--- GENERACIÓN {contador_generaciones} COMPLETADA ---")
    print(f"Mejor Score de esta gen: {max(scores)}")
    
    if contador_generaciones >= 20:
        ag.guardar_memoria("snake")
    
    if contador_generaciones == 20:
        MODO_ENTRENAMIENTO_RAPIDO = False

def generar_comida():
    comida_x = round(random.randrange(0, ANCHO_PANTALLA - tamano_bloque) / 10.0) * 10.0
    comida_y = round(random.randrange(0, ALTO_PANTALLA - tamano_bloque) / 10.0) * 10.0
    return comida_x, comida_y

def leer_estado_ia(cabeza_x, cabeza_y, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida, lista_vibora):
    direcciones = [
        (0, -tamano_bloque), (tamano_bloque, -tamano_bloque), 
        (tamano_bloque, 0), (tamano_bloque, tamano_bloque),
        (0, tamano_bloque), (-tamano_bloque, tamano_bloque), 
        (-tamano_bloque, 0), (-tamano_bloque, -tamano_bloque)
    ]
    vision_comida = [0.0] * 8
    vision_paredes = [0.0] * 8
    vision_cuerpo = [0.0] * 8
    
    cuerpo_set = set(tuple(segmento) for segmento in lista_vibora[:-1])
    
    for i, (dx, dy) in enumerate(direcciones):
        x_actual = cabeza_x
        y_actual = cabeza_y
        distancia = 0
        comida_detectada = False
        cuerpo_detectado = False
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
            if not cuerpo_detectado and (x_actual, y_actual) in cuerpo_set:
                vision_cuerpo[i] = 1.0 / distancia
                cuerpo_detectado = True

    dir_arriba = 1 if y_cambio == -tamano_bloque else 0
    dir_abajo = 1 if y_cambio == tamano_bloque else 0
    dir_izquierda = 1 if x_cambio == -tamano_bloque else 0
    dir_derecha = 1 if x_cambio == tamano_bloque else 0
    
    direccion_actual = [dir_arriba, dir_abajo, dir_izquierda, dir_derecha]
    return vision_comida + vision_paredes + vision_cuerpo + direccion_actual + [comida_obtenida]

def procesar_accion_ia(salida_ia, x_cambio_actual, y_cambio_actual, tamano_bloque):
    opciones_ordenadas = sorted([(val, idx) for idx, val in enumerate(salida_ia)], key=lambda item: item[0], reverse=True)
    x_nuevo, y_nuevo = x_cambio_actual, y_cambio_actual
    decision_final = -1

    for valor, decision in opciones_ordenadas:
        if decision == 0 and y_cambio_actual != tamano_bloque:
            x_nuevo, y_nuevo = 0, -tamano_bloque
            decision_final = decision
            break
        elif decision == 1 and y_cambio_actual != -tamano_bloque:
            x_nuevo, y_nuevo = 0, tamano_bloque
            decision_final = decision
            break
        elif decision == 2 and x_cambio_actual != tamano_bloque:
            x_nuevo, y_nuevo = -tamano_bloque, 0
            decision_final = decision
            break
        elif decision == 3 and x_cambio_actual != -tamano_bloque:
            x_nuevo, y_nuevo = tamano_bloque, 0
            decision_final = decision
            break
            
    if decision_final == -1:
        decision_final = opciones_ordenadas[0][1]
    return x_nuevo, y_nuevo, decision_final

# ==========================================
#             SIMULACIÓN PURA (WORKER)
# ==========================================
def simular_vibora_worker(args):
    """
    Función headless para simular una víbora hasta que muera.
    Devuelve (index, fitness)
    """
    index, red_neuronal = args
    
    x_actual = ANCHO_PANTALLA / 2
    y_actual = ALTO_PANTALLA / 2
    x_cambio = tamano_bloque 
    y_cambio = 0

    lista_vibora = []
    largo_vibora = 1
    comida_obtenida = 0
    comida_x, comida_y = generar_comida()
    
    pasos_restantes = 200 
    pasos_dados = 0
    
    fitness_acumulado = 0.0
    distancia_anterior = abs(x_actual - comida_x) + abs(y_actual - comida_y)
    mejor_distancia_comida = distancia_anterior
    
    historial_decisiones = []
    forzado_por_bucle = False
    muerto = False

    while not muerto:
        comida_dir_x = 1.0 if comida_x > x_actual else (-1.0 if comida_x < x_actual else 0.0)
        comida_dir_y = 1.0 if comida_y > y_actual else (-1.0 if comida_y < y_actual else 0.0)
        
        estado_base = leer_estado_ia(x_actual, y_actual, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida, lista_vibora)
        estado_completo = estado_base + [comida_dir_x, comida_dir_y]
        
        red_neuronal.set_entradas(estado_completo)
        red_neuronal.prediccion_old()
        x_cambio, y_cambio, decision_tomada = procesar_accion_ia(red_neuronal.salidas, x_cambio, y_cambio, tamano_bloque)
        
        historial_decisiones.append(decision_tomada)
        if len(historial_decisiones) > 4:
            historial_decisiones.pop(0)
            
        if len(historial_decisiones) == 4 and len(set(historial_decisiones)) == 4:
            fitness_acumulado -= 300.0
            forzado_por_bucle = True
        
        pasos_dados += 1
        pasos_restantes -= 1
        
        x_actual += x_cambio
        y_actual += y_cambio
        cabeza_vibora = [x_actual, y_actual]
        lista_vibora.append(cabeza_vibora)

        if x_actual == comida_x and y_actual == comida_y:
            comida_obtenida = 1
            comida_x, comida_y = generar_comida()
            largo_vibora += 1
            pasos_restantes = 200  
            fitness_acumulado += 50000.0  
            distancia_anterior = abs(x_actual - comida_x) + abs(y_actual - comida_y)
            mejor_distancia_comida = distancia_anterior
            historial_decisiones = []
        else:
            comida_obtenida = 0

        if len(lista_vibora) > largo_vibora:
            del lista_vibora[0]

        distancia_actual = abs(x_actual - comida_x) + abs(y_actual - comida_y)
        if distancia_actual < distancia_anterior:
            fitness_acumulado += 5.0   
        else:
            fitness_acumulado -= 5.5   

        if distancia_actual < mejor_distancia_comida:
            mejor_distancia_comida = distancia_actual
            
        distancia_anterior = distancia_actual

        if forzado_por_bucle:
            muerto = True
            
        if x_actual >= ANCHO_PANTALLA or x_actual < 0 or y_actual >= ALTO_PANTALLA or y_actual < 0:
            muerto = True
            fitness_acumulado -= 100.0  
            
        if largo_vibora > 1:
            for segmento in lista_vibora[:-1]:
                if segmento == cabeza_vibora:
                    muerto = True  
                    fitness_acumulado -= 100.0  
                    
        if pasos_restantes <= 0:
            muerto = True  

    bono_cercania = 2000.0 / (mejor_distancia_comida + 1.0)
    fitness_final = fitness_acumulado + bono_cercania
    if fitness_final < 1:
        fitness_final = 1
        
    return index, round(fitness_final, 2)

# ==========================================
#             SIMULACIÓN VISUAL
# ==========================================
def simular_vibora_visual(pantalla, reloj, index, red_neuronal):
    """
    Simula una víbora con renderizado en pantalla.
    Devuelve el fitness, o None si el usuario cierra el juego.
    """
    global MODO_ENTRENAMIENTO_RAPIDO
    x_actual = ANCHO_PANTALLA / 2
    y_actual = ALTO_PANTALLA / 2
    x_cambio = tamano_bloque 
    y_cambio = 0

    lista_vibora = []
    largo_vibora = 1
    comida_obtenida = 0
    comida_x, comida_y = generar_comida()
    
    pasos_restantes = 200 
    pasos_dados = 0
    
    fitness_acumulado = 0.0
    distancia_anterior = abs(x_actual - comida_x) + abs(y_actual - comida_y)
    mejor_distancia_comida = distancia_anterior
    
    historial_decisiones = []
    forzado_por_bucle = False
    muerto = False
    decision_tomada = "N/A"

    while not muerto:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return None
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_q:
                    return None
                if evento.key == pygame.K_m:
                    MODO_ENTRENAMIENTO_RAPIDO = True
                    # Al activar modo rápido saltamos esta simulación para acelerar
                    fitness_temp = fitness_acumulado + (2000.0 / (mejor_distancia_comida + 1.0))
                    if fitness_temp < 1: fitness_temp = 1
                    return round(fitness_temp, 2)

        comida_dir_x = 1.0 if comida_x > x_actual else (-1.0 if comida_x < x_actual else 0.0)
        comida_dir_y = 1.0 if comida_y > y_actual else (-1.0 if comida_y < y_actual else 0.0)
        
        estado_base = leer_estado_ia(x_actual, y_actual, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida, lista_vibora)
        estado_completo = estado_base + [comida_dir_x, comida_dir_y]
        
        red_neuronal.set_entradas(estado_completo)
        red_neuronal.prediccion_old()
        x_cambio, y_cambio, decision_tomada = procesar_accion_ia(red_neuronal.salidas, x_cambio, y_cambio, tamano_bloque)
        
        historial_decisiones.append(decision_tomada)
        if len(historial_decisiones) > 4:
            historial_decisiones.pop(0)
            
        if len(historial_decisiones) == 4 and len(set(historial_decisiones)) == 4:
            fitness_acumulado -= 300.0
            forzado_por_bucle = True
        
        pasos_dados += 1
        pasos_restantes -= 1
        
        x_actual += x_cambio
        y_actual += y_cambio
        cabeza_vibora = [x_actual, y_actual]
        lista_vibora.append(cabeza_vibora)

        if x_actual == comida_x and y_actual == comida_y:
            comida_obtenida = 1
            comida_x, comida_y = generar_comida()
            largo_vibora += 1
            pasos_restantes = 200  
            fitness_acumulado += 50000.0  
            distancia_anterior = abs(x_actual - comida_x) + abs(y_actual - comida_y)
            mejor_distancia_comida = distancia_anterior
            historial_decisiones = []
        else:
            comida_obtenida = 0

        if len(lista_vibora) > largo_vibora:
            del lista_vibora[0]

        distancia_actual = abs(x_actual - comida_x) + abs(y_actual - comida_y)
        if distancia_actual < distancia_anterior:
            fitness_acumulado += 5.0   
        else:
            fitness_acumulado -= 5.5   

        if distancia_actual < mejor_distancia_comida:
            mejor_distancia_comida = distancia_actual
            
        distancia_anterior = distancia_actual

        if forzado_por_bucle:
            muerto = True
            
        if x_actual >= ANCHO_PANTALLA or x_actual < 0 or y_actual >= ALTO_PANTALLA or y_actual < 0:
            muerto = True
            fitness_acumulado -= 100.0  
            
        if largo_vibora > 1:
            for segmento in lista_vibora[:-1]:
                if segmento == cabeza_vibora:
                    muerto = True  
                    fitness_acumulado -= 100.0  
                    
        if pasos_restantes <= 0:
            muerto = True  

        # --- DIBUJADO ---
        pantalla.fill(NEGRO)
        pygame.draw.rect(pantalla, AMARILLO, [comida_x, comida_y, tamano_bloque, tamano_bloque])
        for bloque in lista_vibora:
            pygame.draw.rect(pantalla, VERDE, [bloque[0], bloque[1], tamano_bloque, tamano_bloque])
            
        texto_estado = fuente_ia.render(f"IA Decision: {decision_tomada}", True, AMARILLO)
        texto_red = fuente_ia.render(f"Generacion actual: {contador_generaciones + 1}", True, AMARILLO)
        texto_ind = fuente_ia.render(f"Víbora actual: {index + 1}/{TAM_POBLACION_NEURONAL}", True, AMARILLO)
        texto_energia = fuente_ia.render(f"Energia restante: {pasos_restantes}", True, AMARILLO)
        
        pantalla.blit(texto_estado, [5, 40])
        pantalla.blit(texto_red, [5, 55])
        pantalla.blit(texto_ind, [5, 70])
        pantalla.blit(texto_energia, [5, 85])
        texto_puntos = fuente_puntos.render("Manzanas: " + str(largo_vibora - 1), True, AMARILLO)
        pantalla.blit(texto_puntos, [0, 0])
        
        pygame.display.update()
        reloj.tick(30)

    bono_cercania = 2000.0 / (mejor_distancia_comida + 1.0)
    fitness_final = fitness_acumulado + bono_cercania
    if fitness_final < 1: fitness_final = 1
    return round(fitness_final, 2)

# ==========================================
#             CONTROL PRINCIPAL
# ==========================================
def simular_poblacion_entera(pool, pantalla, reloj):
    global array_poblacion_red, MODO_ENTRENAMIENTO_RAPIDO
    
    scores = [1] * TAM_POBLACION_NEURONAL
    
    if MODO_ENTRENAMIENTO_RAPIDO:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                MODO_ENTRENAMIENTO_RAPIDO = False
                
        # Evaluación en paralelo
        args_list = [(i, red) for i, red in enumerate(array_poblacion_red)]
        resultados = pool.map(simular_vibora_worker, args_list)
        
        # Ordenar y guardar
        for idx, fit in resultados:
            scores[idx] = fit
    else:
        # Evaluación secuencial visual (como el original)
        for i, red in enumerate(array_poblacion_red):
            fit = simular_vibora_visual(pantalla, reloj, i, red)
            if fit is None: return False
            scores[i] = fit
            if MODO_ENTRENAMIENTO_RAPIDO: 
                # Terminar de evaluar al resto de la población inmediatamente en paralelo
                redes_restantes = [(idx, array_poblacion_red[idx]) for idx in range(i + 1, TAM_POBLACION_NEURONAL)]
                if redes_restantes:
                    resultados = pool.map(simular_vibora_worker, redes_restantes)
                    for idx, f in resultados:
                        scores[idx] = f
                break 
                
    algoritmo_genetico_ciclo(scores)
    return True

def main():
    inicializar_redes()
    pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
    pygame.display.set_caption('Víbora IA Multi-Procesos')
    reloj = pygame.time.Clock()
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_cores) as pool:
        jugando = True
        while jugando:
            jugando = simular_poblacion_entera(pool, pantalla, reloj)
            
    pygame.quit()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
