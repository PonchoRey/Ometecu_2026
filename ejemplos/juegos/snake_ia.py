import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pygame
import time
import random
import math
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
fitnees_score = []
index_redes = 0

contador_generaciones = 0
MODO_ENTRENAMIENTO_RAPIDO = True  # True para entrenar a ciegas a máxima velocidad

# --- Configuración Red Neuronal ---
for x in range(TAM_POBLACION_NEURONAL):
    red_neuronal = Ometecu()
    red_neuronal.inicio_synapsis("snake")
    # 23 entradas debido al radar de orientación directa de la manzana
    red_neuronal.set_config_red(capa_inicial=23, capa_intermedia=20, capa_final=4)
    # Híbrido: ReLU en capas ocultas, Lineal en la salida para desempate preciso
    red_neuronal.funcionActivacion("r", "r", "s")
    valor_aleatorio = [random.uniform(-0.9, 0.9) for _ in red_neuronal.get_memoria()]
    red_neuronal.set_memoria_genetico(valor_aleatorio)
    array_poblacion_red.append(red_neuronal)

ag = AlgoritmoGenetico(
    tam_poblacion=len(array_poblacion_red), 
    tasa_mutacion=0.03,  
    num_generaciones=NUM_GENERACIONES
)
try: 
    array_poblacion_red[0].set_memoria_genetico(ag.obtener_memoria("snake"))
    array_poblacion_red[1].set_memoria_genetico(ag.obtener_memoria("snake"))
    array_poblacion_red[2].set_memoria_genetico(ag.obtener_memoria("snake"))
    
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
pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
pygame.display.set_caption('Víbora IA: Sistema Anti-Bucles Integrado')

reloj = pygame.time.Clock()
tamano_bloque = 10

fuente_mensaje = pygame.font.SysFont("bahnschrift", 20)
fuente_puntos = pygame.font.SysFont("comicsansms", 25)
fuente_ia = pygame.font.SysFont("consolas", 12)

# ==========================================
#             LÓGICA AUXILIAR
# ==========================================
def algoritmo_genetico(scores):
    global array_poblacion_red, contador_generaciones, MODO_ENTRENAMIENTO_RAPIDO
    array_poblacion_red = ag.redes_genetico(scores, array_poblacion_red)
    ag.guardar_memoria("snake")

    
    contador_generaciones += 1
    print(f"--- GENERACIÓN {contador_generaciones} COMPLETADA ---")
    print(f"Mejor Score de esta gen: {max(scores)}")
    
    # Después de 100 generaciones a ciegas, activa la pantalla para evaluar visualmente
    if contador_generaciones >= 100:
        MODO_ENTRENAMIENTO_RAPIDO = False

def mostrar_puntuacion(puntos):
    texto_puntos = fuente_puntos.render("Manzanas: " + str(puntos), True, AMARILLO)
    pantalla.blit(texto_puntos, [0, 0])

def dibujar_vibora(tamano_bloque, lista_vibora):
    for bloque in lista_vibora:
        pygame.draw.rect(pantalla, VERDE, [bloque[0], bloque[1], tamano_bloque, tamano_bloque])

def generar_comida():
    comida_x = round(random.randrange(0, ANCHO_PANTALLA - tamano_bloque) / 10.0) * 10.0
    comida_y = round(random.randrange(0, ALTO_PANTALLA - tamano_bloque) / 10.0) * 10.0
    return comida_x, comida_y

def leer_estado_ia(cabeza_x, cabeza_y, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida):
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
    return vision_comida + vision_paredes + direccion_actual + [comida_obtenida]

def red_neuronal_simulada(estado, index):
    red_neuronal = array_poblacion_red[index]
    red_neuronal.set_entradas(estado)
    red_neuronal.prediccion_old()
    return red_neuronal.salidas  

def procesar_accion_ia(salida_ia, x_cambio_actual, y_cambio_actual, tamano_bloque):
    opciones_ordenadas = sorted([(val, idx) for idx, val in enumerate(salida_ia)], key=lambda item: item[0], reverse=True)
    x_nuevo, y_nuevo = x_cambio_actual, y_cambio_actual
    decision_final = -1

    for valor, decision in opciones_ordenadas:
        if decision == 0 and y_cambio_actual != tamano_bloque:
            x_nuevo, y_nuevo = 0, -tamano_bloque
            decision_final = decision
            break
        elif decision == 1 and y_cambio_actual != -tamano_bloque: # CORREGIDO: Removido el texto intruso 'sheeting'
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
#             JUEGO PRINCIPAL
# ==========================================
def juego_principal():
    global index_redes, fitnees_score, array_poblacion_red, contador_generaciones, MODO_ENTRENAMIENTO_RAPIDO

    juego_terminado = False
    juego_cerrado = False

    x_actual = ANCHO_PANTALLA / 2
    y_actual = ALTO_PANTALLA / 2
    x_cambio = tamano_bloque 
    y_cambio = 0

    lista_vibora = []
    largo_vibora = 1
    comida_obtenida = 0
    comida_x, comida_y = generar_comida()
    modo_ia_activado = True 
    
    pasos_restantes = 200 
    pasos_dados = 0
    
    # --- DETALLES DEL FITNESS AGRESIVO ---
    fitness_acumulado = 0.0
    distancia_anterior = abs(x_actual - comida_x) + abs(y_actual - comida_y)
    mejor_distancia_comida = distancia_anterior
    decision_tomada = "N/A"
    
    # --- DETECTOR ANTI-BUCLES ---
    historial_decisiones = []
    forzado_por_bucle = False
    
    while not juego_terminado:

        while juego_cerrado:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    return False
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_q:
                        return False 
                    if evento.key == pygame.K_c:
                        return True 

            if modo_ia_activado:
                return True

            pantalla.fill(NEGRO)
            texto = fuente_mensaje.render("Perdiste! 'C' jugar, 'Q' salir", True, ROJO)
            pantalla.blit(texto, (ANCHO_PANTALLA/2 - 190, ALTO_PANTALLA/2))
            mostrar_puntuacion(largo_vibora - 1)
            pygame.display.update()

        if not MODO_ENTRENAMIENTO_RAPIDO:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    return False
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_q:
                        return False
                    
        # 1. LA IA TOMA LA DECISIÓN (CON RADAR DE ORIENTACIÓN DIRECTA)
        if modo_ia_activado:
            comida_dir_x = 1.0 if comida_x > x_actual else (-1.0 if comida_x < x_actual else 0.0)
            comida_dir_y = 1.0 if comida_y > y_actual else (-1.0 if comida_y < y_actual else 0.0)
            
            estado_base = leer_estado_ia(x_actual, y_actual, comida_x, comida_y, x_cambio, y_cambio, tamano_bloque, comida_obtenida)
            estado_completo = estado_base + [comida_dir_x, comida_dir_y]
            
            salida_ia = red_neuronal_simulada(estado_completo, index_redes)
            x_cambio, y_cambio, decision_tomada = procesar_accion_ia(salida_ia, x_cambio, y_cambio, tamano_bloque)
            
            # --- EVALUACIÓN ANTI-BUCLE ---
            historial_decisiones.append(decision_tomada)
            if len(historial_decisiones) > 4:
                historial_decisiones.pop(0)
                
            # Si los últimos 4 movimientos son completamente distintos, completó un giro circular cerrado
            if len(historial_decisiones) == 4 and len(set(historial_decisiones)) == 4:
                fitness_acumulado -= 300.0  # Castigo de peso directo
                forzado_por_bucle = True
        
        pasos_dados += 1
        pasos_restantes -= 1
        
        # 2. SE MUEVE LA CABEZA DE LA VÍBORA
        x_actual += x_cambio
        y_actual += y_cambio
        cabeza_vibora = [x_actual, y_actual]
        lista_vibora.append(cabeza_vibora)

        # 3. VERIFICACIÓN INMEDIATA DE COMIDA
        if x_actual == comida_x and y_actual == comida_y:
            comida_obtenida = 1
            comida_x, comida_y = generar_comida()
            largo_vibora += 1
            pasos_restantes = 200  
            fitness_acumulado += 50000.0  # PREMIO SUPREMO POR COMER DE VERDAD
            distancia_anterior = abs(x_actual - comida_x) + abs(y_actual - comida_y)
            mejor_distancia_comida = distancia_anterior
            historial_decisiones = []     # Limpiamos historial al comer para no penalizar giros de acomodo
        else:
            comida_obtenida = 0

        # Mantener la proporción del cuerpo
        if len(lista_vibora) > largo_vibora:
            del lista_vibora[0]

        # 4. SISTEMA DINÁMICO DE RECOMPENSA Y CASTIGO PASO A PASO
        distancia_actual = abs(x_actual - comida_x) + abs(y_actual - comida_y)
        
        if distancia_actual < distancia_anterior:
            fitness_acumulado += 5.0   # Recompensa por cazar la manzana
        else:
            fitness_acumulado -= 5.5   # Castigo por perder el tiempo o alejarse

        if distancia_actual < mejor_distancia_comida:
            mejor_distancia_comida = distancia_actual
            
        distancia_anterior = distancia_actual

        # 5. CONTROL DE COLISIONES Y FÍSICAS DE MUERTE
        muerto = False
        
        # Activar muerte si fue atrapada in-fraganti en un bucle
        if forzado_por_bucle:
            muerto = True
            
        # Límites del mapa
        if x_actual >= ANCHO_PANTALLA or x_actual < 0 or y_actual >= ALTO_PANTALLA or y_actual < 0:
            muerto = True
            fitness_acumulado -= 100.0  
            
        # Colisión corporal legítima
        if largo_vibora > 1:
            for segmento in lista_vibora[:-1]:
                if segmento == cabeza_vibora:
                    muerto = True  
                    fitness_acumulado -= 100.0  
                    
        # Muerte por falta de energía o inanición
        if pasos_restantes <= 0:
            muerto = True  

        if muerto:
            # Cálculo final balanceado
            bono_cercania = 2000.0 / (mejor_distancia_comida + 1.0)
            fitness_final = fitness_acumulado + bono_cercania
            
            # Forzamos un piso mínimo de 1 para evitar errores en la ruleta del AG
            if fitness_final < 1:
                fitness_final = 1
                
            fitnees_score.append(round(fitness_final, 2))
            index_redes += 1
            
            if index_redes >= TAM_POBLACION_NEURONAL:
                algoritmo_genetico(fitnees_score)
                fitnees_score = []
                index_redes = 0
                
            juego_cerrado = True

        # 6. DIBUJADO CONDICIONAL
        if not juego_cerrado and not MODO_ENTRENAMIENTO_RAPIDO:
            pantalla.fill(NEGRO)
            pygame.draw.rect(pantalla, AMARILLO, [comida_x, comida_y, tamano_bloque, tamano_bloque])
            dibujar_vibora(tamano_bloque, lista_vibora)
            
            texto_estado = fuente_ia.render(f"IA Decision: {decision_tomada}", True, AMARILLO)
            texto_red = fuente_ia.render(f"Generacion actual: {contador_generaciones + 1}", True, AMARILLO)
            texto_ind = fuente_ia.render(f"Víbora actual: {index_redes + 1}/{TAM_POBLACION_NEURONAL}", True, AMARILLO)
            texto_energia = fuente_ia.render(f"Energia restante: {pasos_restantes}", True, AMARILLO)
            
            pantalla.blit(texto_estado, [5, 40])
            pantalla.blit(texto_red, [5, 55])
            pantalla.blit(texto_ind, [5, 70])
            pantalla.blit(texto_energia, [5, 85])
            mostrar_puntuacion(largo_vibora - 1)
            pygame.display.update()

        # 7. CONTROL DE FPS
        if not MODO_ENTRENAMIENTO_RAPIDO:
            reloj.tick(30)  
        else:
            reloj.tick(0)   
        
    return False

if __name__ == "__main__":
    jugando = True
    while jugando:
        jugando = juego_principal()
    pygame.quit()
    quit()