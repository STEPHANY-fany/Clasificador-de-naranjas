import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


ARCHIVO_CSV = "dataset_entrenamiento.csv"


print("=" * 60)
print("CARGANDO DATASET...")
print("=" * 60)

datos = pd.read_csv(ARCHIVO_CSV)

print("\nPrimeras filas:\n")
print(datos.head())

print("\nTotal de registros:", len(datos))

print("\nDistribución de clases:")
print(datos["clase"].value_counts())


X = datos.drop(columns=["imagen", "clase"])
y = datos["clase"]

# Guardar nombres de columnas para el sistema en tiempo real
joblib.dump(list(X.columns), "columnas_modelo.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nRegistros para entrenamiento:", len(X_train))
print("Registros para prueba:", len(X_test))


modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nEntrenando Random Forest...")

modelo.fit(X_train, y_train)

print("Entrenamiento terminado.")


predicciones = modelo.predict(X_test)

accuracy = accuracy_score(y_test, predicciones)

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"\nAccuracy: {accuracy*100:.2f}%")

print("\nReporte de clasificación:\n")

print(classification_report(y_test, predicciones))

print("\nMatriz de confusión:\n")

print(confusion_matrix(y_test, predicciones))


print("\n" + "=" * 60)
print("IMPORTANCIA DE LAS CARACTERÍSTICAS")
print("=" * 60)

importancias = pd.DataFrame({

    "Característica": X.columns,
    "Importancia": modelo.feature_importances_

})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print(importancias)


joblib.dump(modelo, "modelo_naranjas.pkl")

print("\n" + "=" * 60)
print("MODELO GUARDADO")
print("=" * 60)

print("✔ modelo_naranjas.pkl")
print("✔ columnas_modelo.pkl")