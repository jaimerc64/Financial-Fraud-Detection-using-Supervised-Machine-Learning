
# 1. LIBRERÍAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

plt.style.use("ggplot")


# 2. CARGA DE DATOS

data = pd.read_csv("creditcard.csv")


# 3. PREPROCESAMIENTO

scaler = StandardScaler()
data["Amount"] = scaler.fit_transform(data[["Amount"]])

# Eliminar variable Time
data = data.drop(columns=["Time"])


# 4. BALANCEO

fraud = data[data["Class"] == 1]
normal = data[data["Class"] == 0]

normal_sample = normal.sample(len(fraud)*3, random_state=42)
data_balanced = pd.concat([fraud, normal_sample])


# 5. TRAIN / TEST

X = data_balanced.drop("Class", axis=1)
y = data_balanced["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 6. MODELO REGRESIÓN LOGÍSTICA

model_log = LogisticRegression(max_iter=1000)
model_log.fit(X_train, y_train)

y_pred_log = model_log.predict(X_test)
y_prob_log = model_log.predict_proba(X_test)[:, 1]

print("REGRESIÓN LOGÍSTICA")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("AUC:", roc_auc_score(y_test, y_prob_log))


# 7. MODELO KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:, 1]

print("\n KNN")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("AUC:", roc_auc_score(y_test, y_prob_knn))


# 8. GRÁFICAS

# MATRIZ DE CONFUSIÓN 
cm = confusion_matrix(y_test, y_pred_log)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure()
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - KNN")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

# CURVA ROC 
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)

auc_log = roc_auc_score(y_test, y_prob_log)
auc_knn = roc_auc_score(y_test, y_prob_knn)

plt.figure()
plt.plot(fpr_log, tpr_log, label=f"Regresión Logística (AUC = {auc_log:.2f})")
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.2f})")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC - Comparación de Modelos")
plt.legend()
plt.show()

# COMPARACIÓN DE MODELOS
results = pd.DataFrame({
    "Modelo": ["Regresión Logística", "KNN"],
    "Recall": [0.91, 0.89],      
    "Precisión": [0.94, 0.96]
})

results.set_index("Modelo").plot(kind="bar")
plt.title("Comparación de Modelos")
plt.ylabel("Valor")
plt.xticks(rotation=0)
plt.show()

# IMPORTANCIA DE VARIABLES
coefficients = pd.Series(model_log.coef_[0], index=X.columns)
top_features = coefficients.abs().sort_values(ascending=False).head(10)

plt.figure()
top_features.plot(kind="barh")
plt.title("Variables Más Influyentes")
plt.xlabel("Magnitud del Coeficiente")
plt.show()
