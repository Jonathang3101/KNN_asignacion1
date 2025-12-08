#!/usr/bin/env python3
"""
Asignación 3: Árboles de Decisión con MNIST
Comparación de profundidades para analizar underfitting y overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*60)
print("ÁRBOLES DE DECISIÓN - MNIST")
print("Comparación de Profundidades")
print("="*60)

# ============================================================================
# PASO 1: Cargar datos
# ============================================================================

print("\n1. Cargando datasets...")
train_df = pd.read_csv('../../datasets/mnist/mnist_train.csv')
test_df = pd.read_csv('../../datasets/mnist/mnist_test.csv')

print(f"✓ Train: {train_df.shape}")
print(f"✓ Test: {test_df.shape}")

# ============================================================================
# PASO 2: Preparar variables
# ============================================================================

print("\n2. Preparando variables...")

# Usar muestra para acelerar
sample_size = 10000
X_train_full = train_df.iloc[:sample_size, 1:].values
y_train_full = train_df.iloc[:sample_size, 0].values

X_test_full = test_df.iloc[:, 1:].values
y_test_full = test_df.iloc[:, 0].values

# División 80/20
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"✓ Train: {X_train.shape[0]:,} ejemplos")
print(f"✓ Val: {X_val.shape[0]:,} ejemplos")

# ============================================================================
# PASO 3: Entrenar modelos
# ============================================================================

print("\n3. Entrenando modelos...")

profundidades = [5, 10, 20]
modelos = {}
resultados = []

for prof in profundidades:
    print(f"\n  Profundidad {prof}...")
    
    modelo = DecisionTreeClassifier(
        max_depth=prof,
        random_state=42
    )
    
    modelo.fit(X_train, y_train)
    
    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_val = accuracy_score(y_val, modelo.predict(X_val))
    
    modelos[prof] = modelo
    resultados.append({
        'Profundidad': prof,
        'Acc Train': acc_train,
        'Acc Val': acc_val,
        'Diferencia': acc_train - acc_val
    })
    
    print(f"    Train: {acc_train:.4f}")
    print(f"    Val:   {acc_val:.4f}")

# ============================================================================
# PASO 4: Resultados
# ============================================================================

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)

df_res = pd.DataFrame(resultados)
print("\n", df_res.to_string(index=False))

mejor_idx = df_res['Acc Val'].idxmax()
mejor_prof = df_res.loc[mejor_idx, 'Profundidad']
mejor_acc = df_res.loc[mejor_idx, 'Acc Val']

print(f"\n✓ Mejor modelo: Profundidad {mejor_prof} (Acc = {mejor_acc:.4f})")

# ============================================================================
# PASO 5: Gráficas
# ============================================================================

print("\n4. Generando gráficas...")

# Gráfica 1: Desempeño
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_res['Profundidad'], df_res['Acc Train'], 
        marker='o', linewidth=2.5, markersize=10, label='Train')
ax.plot(df_res['Profundidad'], df_res['Acc Val'], 
        marker='s', linewidth=2.5, markersize=10, label='Validación')
ax.set_xlabel('Profundidad', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Desempeño vs Profundidad', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('desempeno.png', dpi=300, bbox_inches='tight')
print("✓ desempeno.png")
plt.close()

# Gráfica 2: Matriz de confusión
mejor_modelo = modelos[mejor_prof]
y_pred = mejor_modelo.predict(X_val)
cm = confusion_matrix(y_val, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title(f'Matriz de Confusión - Prof {mejor_prof}', fontsize=14)
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✓ matriz_confusion.png")
plt.close()

# Gráfica 3: Importancia de píxeles
importancias = mejor_modelo.feature_importances_.reshape(28, 28)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(importancias, cmap='hot')
ax.set_title('Importancia de Píxeles', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('importancia_pixeles.png', dpi=300, bbox_inches='tight')
print("✓ importancia_pixeles.png")
plt.close()

# ============================================================================
# PASO 6: Conclusiones
# ============================================================================

print("\n" + "="*60)
print("CONCLUSIONES")
print("="*60)

print(f"\n1. Desempeño general:")
print(f"   Accuracy promedio: {df_res['Acc Val'].mean():.4f}")

print(f"\n2. Mejor modelo:")
print(f"   Profundidad {mejor_prof} con {mejor_acc:.4f} accuracy")

print(f"\n3. Underfitting:")
prof_min = df_res['Profundidad'].min()
if df_res.loc[0, 'Acc Train'] < 0.80:
    print(f"   Profundidad {prof_min} muestra underfitting")
else:
    print(f"   No hay underfitting significativo")

print(f"\n4. Overfitting:")
max_diff = df_res['Diferencia'].max()
if max_diff > 0.10:
    prof_over = df_res.loc[df_res['Diferencia'].idxmax(), 'Profundidad']
    print(f"   Profundidad {prof_over} muestra overfitting")
else:
    print(f"   Overfitting controlado")

print(f"\n5. Recomendación:")
print(f"   Usar profundidad {mejor_prof} para este problema")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
