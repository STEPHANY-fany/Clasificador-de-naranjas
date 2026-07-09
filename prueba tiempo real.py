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

print("Modelo cargado correctamente")


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    imagen = frame.copy()


    try:

        caracteristicas = extraer_caracteristicas(imagen)


        if caracteristicas is not None:


            X = pd.DataFrame(
                [caracteristicas],
                columns=columnas
            )


            prediccion = modelo.predict(X)[0]


            probabilidades = modelo.predict_proba(X)[0]

            confianza = max(probabilidades)*100


            texto = f"{prediccion} | {confianza:.1f}%"


            cv2.putText(
                imagen,
                texto,
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )


            mascara = obtener_mascara(frame)
            bordes = obtener_bordes(frame)


            cv2.imshow(
                "Mascara",
                mascara
            )

            cv2.imshow(
                "Bordes",
                bordes
            )


        else:

            cv2.putText(
                imagen,
                "No se detecta naranja",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )


    except Exception as e:

        print("Error:",e)



    cv2.imshow(
        "Clasificador de naranjas",
        imagen
    )


    # ESC para salir
    if cv2.waitKey(1)==27:
        break



cap.release()
cv2.destroyAllWindows()