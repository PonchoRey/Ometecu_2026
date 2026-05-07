import random
import time
# Asegúrate de que tu archivo Ometecu.py esté en el mismo directorio
from Ometecu import Ometecu

# --- 0. CLASE PARA SIMULAR LA LÍNEA DE ENSAMBLAJE ---

class LineaDeEnsamblaje:
    """
    Simula una línea de ensamblaje de coches que produce vehículos
    con diferentes estados de calidad para ser inspeccionados.
    """
    def __init__(self, estado_produccion='optima'):
        self.estado = ''
        self.cambiar_estado(estado_produccion, primera_vez=True)

    def inspeccionar_nuevo_coche(self):
        """Devuelve las lecturas de los sensores del siguiente coche en la línea."""
        # Simular lectura del sensor de temperatura
        temp_motor = self.temp_base + random.uniform(-2.0, 2.0)
        temp_motor = max(70, min(120, temp_motor))

        # Simular lectura del sensor de alineación
        desviacion = self.alineacion_base + random.uniform(-0.5, 0.5)
        desviacion = max(0, min(10, desviacion))
        
        # Simular sensor de defectos de pintura
        defectos = self.pintura_base + random.uniform(-0.05, 0.05)
        defectos = max(0, min(1, defectos))

        return {'temperatura': temp_motor, 'alineacion': desviacion, 'pintura': defectos}

    def cambiar_estado(self, nuevo_estado, primera_vez=False):
        """Simula un cambio en la línea de producción (ej: una máquina se descalibra)."""
        if not primera_vez:
            print(f"\n--- 🏭 ¡Alerta en Producción! Estado de la línea: '{nuevo_estado}' ---\n")
        
        self.estado = nuevo_estado
        if self.estado == 'optima':
            self.temp_base = 90.0   # Temperatura ideal
            self.alineacion_base = 1.0  # Desviación mínima
            self.pintura_base = 0.1   # Casi sin defectos
        elif self.estado == 'sobrecalentamiento':
            self.temp_base = 105.0  # Problema de refrigeración
            self.alineacion_base = 1.5
            self.pintura_base = 0.2
        elif self.estado == 'desalineado':
            self.temp_base = 92.0
            self.alineacion_base = 6.0  # Máquina de alineación descalibrada
            self.pintura_base = 0.3


# --- 1. DATOS DE ENTRENAMIENTO (Conocimiento del Inspector de Calidad) ---
# Entradas: [Temperatura Motor (°C), Desviación Alineación (mm), Defectos Pintura (0-1)]
# Salida:   [Diagnóstico (0.1=Pasa Control, 0.9=Falla Control)]

datos_brutos = [
    # Coches que PASAN el control ✅
    {'entrada': [90, 1.0, 0.1], 'salida': [0.1]}, # Coche perfecto
    {'entrada': [93, 1.5, 0.2], 'salida': [0.1]}, # Dentro de tolerancias aceptables
    {'entrada': [88, 0.5, 0.05], 'salida': [0.1]},# Coche de alta calidad

    # Coches que FALLAN el control ❌
    {'entrada': [110, 2.0, 0.3], 'salida': [0.9]},# Falla por sobrecalentamiento
    {'entrada': [95, 8.0, 0.4], 'salida': [0.9]}, # Falla por mala alineación
    {'entrada': [90, 1.0, 0.8], 'salida': [0.9]}, # Falla por defectos de pintura
    {'entrada': [108, 7.5, 0.6], 'salida': [0.9]},# Múltiples fallas graves
]

# --- 2. PREPARAR DATOS (Normalización) ---

entradas_normalizadas = []
salidas_deseadas = []

for dato in datos_brutos:
    temp, alin, pint = dato['entrada']
    # Normalizamos cada entrada a un rango de 0.1 - 0.9
    temp_norm = ((temp - 80) / 40) * 0.8 + 0.1     # Rango esperado de temp: 80-120
    alin_norm = (alin / 10.0) * 0.8 + 0.1        # Rango esperado de alineación: 0-10
    pint_norm = pint * 0.8 + 0.1                   # Rango de pintura ya es 0-1
    
    entradas_normalizadas.append([temp_norm, alin_norm, pint_norm])
    salidas_deseadas.append(dato['salida'])

# --- 3. CONFIGURACIÓN Y ENTRENAMIENTO DE LA RED NEURONAL ---

inspector_ia = Ometecu()
# Arquitectura: 3 entradas, 4 neuronas ocultas, 1 salida
inspector_ia.set_config_red(capa_inicial=3, capa_intermedia=7, capa_final=1) 

epocas = 200000
print(f"🧠 Entrenando al Inspector de Calidad IA durante {epocas} épocas...")

for i in range(epocas):
    idx_aleatorio = random.randint(0, len(entradas_normalizadas) - 1)
    inspector_ia.set_entradas(entradas_normalizadas[idx_aleatorio])
    inspector_ia.set_valor_aprender(salidas_deseadas[idx_aleatorio])
    inspector_ia.entrenamiento()
    if (i + 1) % 10000 == 0:
        print("epocas:", i +1, "/", epocas)

print("✅ Entrenamiento finalizado. El inspector IA está listo.")
print("-" * 50)


# --- 4. INSPECCIÓN EN TIEMPO REAL EN LA LÍNEA DE ENSAMBLAJE ---

linea_actual = LineaDeEnsamblaje(estado_produccion='optima')
print("🤖 Iniciando inspección de calidad en la línea de producción...")
print("Presiona CTRL+C para detener la simulación.")
print("-" * 50)

try:
    coche_numero = 1
    while True:
        # 1. Un nuevo coche llega para ser inspeccionado
        lectura = linea_actual.inspeccionar_nuevo_coche()
        temp_actual = lectura['temperatura']
        alin_actual = lectura['alineacion']
        pint_actual = lectura['pintura']

        # 2. Normalizamos los datos para la red neuronal
        temp_norm = ((temp_actual - 80) / 40) * 0.8 + 0.1
        alin_norm = (alin_actual / 10.0) * 0.8 + 0.1
        pint_norm = pint_actual * 0.8 + 0.1
        
        # 3. La IA realiza su predicción
        inspector_ia.set_entradas([temp_norm, alin_norm, pint_norm])
        prediccion = inspector_ia.prediccion()[0]

        # 4. Interpretamos y mostramos el diagnóstico
        if prediccion > 0.6:
            diagnostico = "❌ FALLA DETECTADA"
        else:
            diagnostico = "✅ PASA CONTROL"
            
        print(f"Coche #{coche_numero}: Temp={temp_actual:.1f}°C, Alin={alin_actual:.1f}mm, Pint={pint_actual:.2f} | Predicción={prediccion:.2f} -> {diagnostico}")
        
        coche_numero += 1
        time.sleep(1.5)

        # --- Simulación de fallos en la línea de producción ---
        if coche_numero % 8 == 0:
            linea_actual.cambiar_estado('sobrecalentamiento')
        elif coche_numero % 13 == 0:
            linea_actual.cambiar_estado('desalineado')
        elif coche_numero % 17 == 0:
            linea_actual.cambiar_estado('optima')


except KeyboardInterrupt:
    print("\n👋 Simulación detenida por el usuario.")