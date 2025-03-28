import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image

# ---- 1. Datos Mejorados ----
data = {
    "nombre": ["Superman", "Iron Man", "Spider-Man", "Hulk", "Thor", "Batman", "Wonder Woman", "Doctor Strange", "Captain America", "Black Panther",
               "Flash", "Green Lantern", "Aquaman", "Cyclops", "Deadpool",
               "Thanos", "Joker", "Lex Luthor", "Green Goblin", "Venom", "Ultron", "Loki", "Magneto", "Darkseid", "Red Skull",
               "Hela", "Reverse Flash", "Kingpin", "Doctor Doom", "Mystique"],
    "fuerza": [10, 7, 6, 10, 9, 5, 9, 4, 8, 7, 9, 6, 8, 5, 7,
               4, 3, 2, 4, 6, 5, 3, 5, 2, 3, 9, 4, 8, 6, 7],
    "inteligencia": [9, 10, 9, 5, 9, 10, 8, 10, 7, 9, 7, 6, 5, 7, 8,
                     6, 9, 10, 7, 3, 8, 10, 9, 5, 8, 2, 10, 3, 4, 9],
    "altura": [191, 175, 178, 244, 198, 188, 183, 180, 188, 182, 180, 185, 190, 175, 177,
               201, 180, 183, 178, 190, 210, 191, 180, 260, 180, 170, 200, 210, 190, 175],
    "clase": ["heroe"]*15 + ["villano"]*15
}

df = pd.DataFrame(data)
df["clase"] = df["clase"].map({"heroe": 0, "villano": 1})

# Mostrar el dataset inicial
st.write("### 📊 Dataset de Héroes y Villanos")
st.dataframe(df)

# ---- 2. División Train/Test ----
X = df[["fuerza", "inteligencia"]]
y = df["clase"]
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, df["nombre"], test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 3. Split train y test ----
st.write("### 🏋️‍♂️ Separamos los datos entrenamientos y los datos de validación")
st.write("Entrenamiento")
cols = 6
rows = len(X_train) // cols
if len(X_train) % cols != 0:
    rows += 1

fig, axs = plt.subplots(rows, cols, figsize=(15, 8))
if len(X_train) == 1:
    axs = [axs]
for i, (name, label) in enumerate(zip(names_train, y_train)):
    tmp_row = i // cols
    tmp_col = i % cols
    img_path = f"images/{name.replace(' ', '_')}.jpeg"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axs[tmp_row, tmp_col].imshow(img)
        axs[tmp_row, tmp_col].axis("off")
        axs[tmp_row, tmp_col].set_title(name, fontsize=10, color="blue" if label == 0 else "red")
st.pyplot(fig)

st.write("Validación")

fig, axs = plt.subplots(1, len(X_test), figsize=(15, 5))
if len(X_test) == 1:
    axs = [axs]
for i, (name, label) in enumerate(zip(names_test, y_test)):
    img_path = f"images/{name.replace(' ', '_')}.jpeg"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(name, fontsize=10, color="blue" if label == 0 else "red")
st.pyplot(fig)

# ---- 3. Entrenar el Modelo (Random Forest) ----
st.write("### 🏋️‍♂️ Entrenar el Modelo de Random Forest")
#n_estimators = st.slider("🌲 Número de árboles", min_value=10, max_value=200, value=100, step=10)
#max_depth = st.slider("📏 Profundidad máxima", min_value=1, max_value=20, value=10, step=1)
#clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

C_value = st.slider("🔧 Complejidad del modelo (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
gamma_value = st.slider("🔬 Flexibilidad del modelo (Gamma)", min_value=0.01, max_value=5.0, value=0.5, step=0.1)
clf = SVC(kernel="rbf", C=C_value, gamma=gamma_value)

clf.fit(X_train_scaled, y_train)


# ---- 4. Frontera de Decisión ----
x_min, x_max = X_train_scaled[:, 0].min() - 0.3, X_train_scaled[:, 0].max() + 0.3
y_min, y_max = X_train_scaled[:, 1].min() - 0.3, X_train_scaled[:, 1].max() + 0.3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cols = 6
rows = len(X_train) // cols
if len(X_train) % cols != 0:
    rows += 1

fig, axs = plt.subplots(rows, cols, figsize=(15, 8))
if len(X_train) == 1:
    axs = [axs]
for i, (name, label) in enumerate(zip(names_train, y_train)):
    tmp_row = i // cols
    tmp_col = i % cols
    img_path = f"images/{name.replace(' ', '_')}.jpeg"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axs[tmp_row, tmp_col].imshow(img)
        axs[tmp_row, tmp_col].axis("off")
        axs[tmp_row, tmp_col].set_title(name, fontsize=10, color="blue" if label == 0 else "red")
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
ax.scatter(X_train_scaled[y_train == 0][:, 0], X_train_scaled[y_train == 0][:, 1], c='blue', edgecolors="k", label="Héroes - Train", alpha=0.7)
ax.scatter(X_train_scaled[y_train == 1][:, 0], X_train_scaled[y_train == 1][:, 1], c='red', edgecolors="k", label="Villanos - Train", alpha=0.7)
ax.set_xlabel("Fuerza")
ax.set_ylabel("Inteligencia")
ax.set_title("Frontera de Decisión del SVM")
ax.legend()
st.pyplot(fig)

# ---- 5. Visualización del Test Set con Imágenes ----
show_test_data = st.checkbox("📌 Mostrar datos de test con imágenes y puntos")
if show_test_data:
    fig, axs = plt.subplots(1, len(X_test), figsize=(15, 5))
    if len(X_test) == 1:
        axs = [axs]
    for i, (name, label) in enumerate(zip(names_test, y_test)):
        img_path = f"images/{name.replace(' ', '_')}.jpeg"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].axis("off")
            axs[i].set_title(name, fontsize=10, color="blue" if label == 0 else "red")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_train_scaled[y_train == 0][:, 0], X_train_scaled[y_train == 0][:, 1], c='blue', edgecolors="k", label="Héroes - Train", alpha=0.7)
    ax.scatter(X_train_scaled[y_train == 1][:, 0], X_train_scaled[y_train == 1][:, 1], c='red', edgecolors="k", label="Villanos - Train", alpha=0.7)
    ax.scatter(X_test_scaled[y_test == 0][:, 0], X_test_scaled[y_test == 0][:, 1], c='blue', marker='x', edgecolors="k", label="Héroes - Test", alpha=0.7)
    ax.scatter(X_test_scaled[y_test == 1][:, 0], X_test_scaled[y_test == 1][:, 1], c='red', marker='x', edgecolors="k", label="Villanos - Test", alpha=0.7)
    ax.legend()
    ax.set_xlabel("Fuerza")
    ax.set_ylabel("Inteligencia")
    ax.set_title("Frontera de Decisión del SVM")
    st.pyplot(fig)


