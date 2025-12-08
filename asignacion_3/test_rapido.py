#!/usr/bin/env python3
"""
Test rápido para verificar que todo funciona
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("="*60)
print("TEST RÁPIDO - Asignación 3")
print("="*60)

# Test 1: Cargar datos
print("\n1. Cargando datos...")
try:
    train_df = pd.read_csv('../../datasets/mnist/mnist_train.csv')
    print(f"   ✓ Train cargado: {train_df.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Preparar variables
print("\n2. Preparando variables...")
sample_size = 1000  # Muestra pequeña para test rápido
X = train_df.iloc[:sample_size, 1:].values
y = train_df.iloc[:sample_size, 0].values
print(f"   ✓ X: {X.shape}, y: {y.shape}")

# Test 3: División
print("\n3. División train/val...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train)}, Val: {len(X_val)}")

# Test 4: Entrenar modelo simple
print("\n4. Entrenando modelo (prof=5)...")
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)
print(f"   ✓ Modelo entrenado")

# Test 5: Evaluar
print("\n5. Evaluando...")
acc_train = accuracy_score(y_train, modelo.predict(X_train))
acc_val = accuracy_score(y_val, modelo.predict(X_val))
print(f"   ✓ Train accuracy: {acc_train:.4f}")
print(f"   ✓ Val accuracy: {acc_val:.4f}")

print("\n" + "="*60)
print("✓ TEST COMPLETADO - Todo funciona correctamente")
print("="*60)
print("\nAhora puedes ejecutar el notebook completo.")
