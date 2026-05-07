import cv2
import numpy as np
from mss import mss
from PIL import Image

def draw_object_silhouettes_screen():
    with mss() as sct:
        # Definir la dimensión de la pantalla a capturar
        monitor = sct.monitors[1]  # Usar el primer monitor

        while True:
            # Capturar la pantalla
            screen_shot = sct.grab(monitor)
            img = np.array(Image.frombytes('RGB', (screen_shot.width, screen_shot.height), screen_shot.rgb))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Aplicar un desenfoque más intenso para reducir el ruido
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Ajustar los valores de umbralización para una mejor detección de bordes
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

            # Encontrar contornos utilizando un método más preciso
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Procesar cada contorno
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # Verificar si el área del contorno supera el umbral
                if area > 20000:
                    # Calcular el centro del contorno para colocar el texto
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0

                    # Dibujar el contorno y el número en la imagen
                    cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

                    # Dibujar el área del contorno en rojo
                    cv2.putText(img, str(area), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Mostrar la imagen resultante
            cv2.imshow('Screen with Object Silhouettes and Areas', img)

            # Interrumpir el bucle si se presiona 'q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

draw_object_silhouettes_screen()
