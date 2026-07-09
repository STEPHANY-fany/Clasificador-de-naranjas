import cv2
import numpy as np


def suavizar_hsv(imagen):

    suavizada = cv2.GaussianBlur(imagen, (5,5), 0)

    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)

    return hsv



def segmentar(hsv):

    bajo_naranja = np.array([0,80,80])
    alto_naranja = np.array([30,255,255])

    mascara = cv2.inRange(
        hsv,
        bajo_naranja,
        alto_naranja
    )

    return mascara



def aplicar_morfologia(mascara):

    kernel = np.ones((5,5),np.uint8)

    apertura = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel
    )

    cierre = cv2.morphologyEx(
        apertura,
        cv2.MORPH_CLOSE,
        kernel
    )

    return cierre



def detectar_bordes(mascara):

    bordes = cv2.Canny(
        mascara,
        100,
        200
    )

    return bordes



def extraer_caracteristicas(imagen):

    hsv = suavizar_hsv(imagen)

    mascara = segmentar(hsv)

    morfologia = aplicar_morfologia(mascara)

    bordes = detectar_bordes(morfologia)

    contornos,_ = cv2.findContours(
        morfologia,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos)==0:

        return None

    contorno=max(
        contornos,
        key=cv2.contourArea
    )

    
    area=cv2.contourArea(contorno)

    
    perimetro=cv2.arcLength(
        contorno,
        True
    )

    
    if perimetro!=0:

        circularidad=(
            4*np.pi*area
        )/(perimetro**2)

    else:

        circularidad=0

    
    media_hsv=cv2.mean(
        hsv,
        mask=morfologia
    )

    media_H=media_hsv[0]
    media_S=media_hsv[1]
    media_V=media_hsv[2]

    
    cantidad_bordes=np.sum(
        bordes>0
    )

   
    pixeles_totales=np.sum(
        morfologia>0
    )

    pixeles_blancos=np.sum(
        (morfologia>0)&
        (bordes>0)
    )

    if pixeles_totales!=0:

        porcentaje_manchas=(
            100*pixeles_blancos
        )/pixeles_totales

    else:

        porcentaje_manchas=0

    caracteristicas=[

        area,

        perimetro,

        circularidad,

        media_H,

        media_S,

        media_V,

        cantidad_bordes,

        porcentaje_manchas

    ]

    return caracteristicas



def obtener_mascara(imagen):

    hsv = suavizar_hsv(imagen)

    mascara = segmentar(hsv)

    mascara = aplicar_morfologia(mascara)

    return mascara



def obtener_bordes(imagen):

    hsv = suavizar_hsv(imagen)

    mascara = segmentar(hsv)

    mascara = aplicar_morfologia(mascara)

    bordes = detectar_bordes(mascara)

    return bordes