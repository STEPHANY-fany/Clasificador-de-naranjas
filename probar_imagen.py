import cv2
import joblib
import pandas as pd

from procesamiento import (
    extraer_caracteristicas,
    obtener_mascara,
    obtener_bordes
)

modelo = joblib.load("modelo_naranjas.pkl")
columnas = joblib.load("columnas_modelo.pkl")

print("Modelo cargado correctamente.")


ruta_imagen = r"C:\Users\steph\Desktop\Procesamiento de imagenes\clasificador naranjas\pruebas\naranja3.jpeg"

imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo abrir la imagen.")
    exit()


caracteristicas = extraer_caracteristicas(imagen)

if caracteristicas is None:
    print("No se detectó ninguna naranja.")
    exit()


X = pd.DataFrame([caracteristicas], columns=columnas)

prediccion = modelo.predict(X)[0]

probabilidades = modelo.predict_proba(X)[0]

indice = probabilidades.argmax()

confianza = probabilidades[indice] * 100


print("PREDICCIÓN")
print(prediccion)
print(f"Confianza: {confianza:.2f}%")


mascara = obtener_mascara(imagen)
bordes = obtener_bordes(imagen)


contornos, _ = cv2.findContours(
    mascara,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


if len(contornos) > 0:

    contorno = max(contornos, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contorno)


    # Caja alrededor de la naranja
    cv2.rectangle(
        imagen,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2
    )


    texto = f"Clase: {prediccion} | {confianza:.1f}%"


    # posición del texto
    pos_y = y - 10 if y > 60 else y + h + 40


    # tamaño del fondo del texto
    (tw, th), _ = cv2.getTextSize(
        texto,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        3
    )


    # fondo blanco
    cv2.rectangle(
        imagen,
        (x, pos_y-th-10),
        (x+tw+10, pos_y+8),
        (255,255,255),
        -1
    )


    # texto grande
    cv2.putText(
        imagen,
        texto,
        (x+5, pos_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,0,0),
        3,
        cv2.LINE_AA
    )



# Redimensionar imágenes
ALTO = 450


def redimensionar(img):

    h, w = img.shape[:2]

    escala = ALTO / h

    nuevo_ancho = int(w * escala)

    return cv2.resize(img, (nuevo_ancho, ALTO))


imagen_vista = redimensionar(imagen)

mascara_vista = redimensionar(
    cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
)

bordes_vista = redimensionar(
    cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
)



panel = cv2.hconcat([
    imagen_vista,
    mascara_vista,
    bordes_vista
])



# Títulos

cv2.putText(
    panel,
    "Original",
    (20,30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255,0,0),
    2
)


cv2.putText(
    panel,
    "Mascara",
    (imagen_vista.shape[1]+20,30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255,0,0),
    2
)


cv2.putText(
    panel,
    "Bordes",
    (imagen_vista.shape[1]*2+20,30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255,0,0),
    2
)



cv2.imshow(
    "Clasificador de Naranjas",
    panel
)


cv2.waitKey(0)

cv2.destroyAllWindows()