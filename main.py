import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from matplotlib.colors import ListedColormap

# 1. Cargar el dataset
dataset = pd.read_csv("data/credit_dataset.csv")

# 2. Separar características (X) y objetivo (y)
X = dataset.drop("Aprobado", axis=1)
y = dataset["Aprobado"]

# 3. Definir columnas numéricas y categóricas
numeric_cols = ["Edad", "IngresosMensuales", "CantidadDeudas"]
categorical_cols = ["HistorialCrediticio", "NivelEducativo"]

# 4. Pipeline de preprocesamiento + modelo SVM completo
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first"), categorical_cols)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear", random_state=0))
])

# 5. División de datos para entrenamiento del modelo general
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 6. Entrenar el modelo completo
model.fit(X_train, y_train)

# --- BLOQUE DE VISUALIZACIÓN EN ESCALA ORIGINAL (solo test) ---

# 7. Preparar solo Edad e IngresosMensuales para el gráfico
X_vis = X[["Edad", "IngresosMensuales"]].values
y_vis = (y == "Sí").astype(int).values  # 1 = Sí, 0 = No

# 8. Dividir esos datos en train/test para la visualización
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
    X_vis, y_vis, test_size=0.25, random_state=0
)

# 9. Ajustar scaler solo con train
sc_vis = StandardScaler().fit(X_vis_train)

# 10. Entrenar SVM solo con estas dos features escaladas
classifier_vis = SVC(kernel="linear", random_state=0)
classifier_vis.fit(sc_vis.transform(X_vis_train), y_vis_train)

# 11. Crear meshgrid en escala ORIGINAL
x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
y_min, y_max = X_vis_train[:, 1].min() - 500, X_vis_train[:, 1].max() + 500
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# 12. Predecir sobre el meshgrid (escala antes)
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = sc_vis.transform(grid)
Z = classifier_vis.predict(grid_scaled).reshape(xx.shape)

# 13. Dibujar boundary y puntos de test (originales)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.scatter(
    X_vis_test[:, 0], X_vis_test[:, 1],
    c=y_vis_test, cmap=ListedColormap(("red", "green")),
    edgecolors="k"
)
plt.title("SVM - Conjunto de Prueba")
plt.xlabel("Edad")
plt.ylabel("Ingresos Mensuales")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# ANALISIS
# Predicción sobre el conjunto de prueba (completo, con todas las features)
y_pred = model.predict(X_test)

# Binarizar para métricas: 1 = "Sí", 0 = "No"
y_test_bin = (y_test == "Sí").astype(int)
y_pred_bin = (y_pred == "Sí").astype(int)

# Métricas clave
cm = confusion_matrix(y_test_bin, y_pred_bin)
acc = accuracy_score(y_test_bin, y_pred_bin)
prec = precision_score(y_test_bin, y_pred_bin)
rec = recall_score(y_test_bin, y_pred_bin)
f1 = f1_score(y_test_bin, y_pred_bin)

# Conclusión
print(
    f"""
MODELO DE CLASIFICACIÓN SVM – ANÁLISIS DE RESULTADOS

Este modelo se ha desarrollado utilizando datos de solicitudes de crédito,
con el objetivo de predecir si un solicitante será aprobado o no aprobado para un crédito,
en función de cinco variables:
  - Edad (años, numérica)
  - IngresosMensuales (USD, numérica)
  - CantidadDeudas (número de deudas, numérica)
  - HistorialCrediticio (Bueno/Regular/Malo, categórica)
  - NivelEducativo (Secundaria/Universitario/Postgrado, categórica)

El algoritmo utilizado fue Support Vector Machine.
La variable objetivo fue binarizada: 1 para "Sí" (aprobado) y 0 para "No" (no aprobado).

Los resultados obtenidos fueron:

  • Exactitud (accuracy):        {acc:.3f}
  • Precisión (clase Sí):        {prec:.3f}
  • Recall (sensibilidad):       {rec:.3f}
  • F1-score:                    {f1:.3f}

La matriz de confusión muestra {cm[1,1]} verdaderos positivos (TP) y {cm[0,0]} verdaderos negativos (TN),
con {cm[0,1]} falsos positivos (FP) y {cm[1,0]} falsos negativos (FN).

Estos resultados indican que el modelo identifica correctamente aproximadamente el {acc*100:.1f}% de los casos
y logra una sensibilidad del {rec*100:.1f}% para detectar solicitudes aprobadas.
"""
)
