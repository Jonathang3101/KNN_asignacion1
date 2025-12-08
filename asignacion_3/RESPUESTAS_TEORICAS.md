# Respuestas Teóricas - Árboles de Decisión

## 📚 Conceptos Fundamentales

### 1. ¿Qué es un Árbol de Decisión?

Un **árbol de decisión** es un modelo de aprendizaje supervisado que realiza predicciones mediante una serie de decisiones binarias organizadas jerárquicamente.

**Estructura:**
```
                    [Raíz]
                 ¿pixel[X] <= 127?
                /                  \
             Sí                     No
            /                        \
    [Nodo Interno]              [Nodo Interno]
    ¿pixel[Y] <= 50?           ¿pixel[Z] <= 200?
      /        \                  /          \
   [Hoja]   [Hoja]            [Hoja]      [Hoja]
   Clase 0  Clase 1           Clase 2     Clase 3
```

**Componentes:**
- **Nodo raíz**: Primera decisión
- **Nodos internos**: Decisiones intermedias
- **Hojas**: Predicciones finales (clases)
- **Ramas**: Caminos de decisión

**Funcionamiento:**
1. Comienza en la raíz
2. Evalúa condición (ej: ¿pixel[X] <= valor?)
3. Sigue rama correspondiente (Sí/No)
4. Repite hasta llegar a una hoja
5. Retorna la clase de la hoja

---

### 2. Profundidad del Árbol (max_depth)

La **profundidad** es el número máximo de niveles desde la raíz hasta las hojas.

**Ejemplo:**
```
Profundidad 1:
    [Raíz]
    /    \
 [Hoja] [Hoja]

Profundidad 2:
        [Raíz]
       /      \
   [Nodo]    [Nodo]
   /   \      /   \
[Hoja][Hoja][Hoja][Hoja]

Profundidad 3:
            [Raíz]
          /        \
      [Nodo]      [Nodo]
      /    \       /    \
  [Nodo][Nodo][Nodo][Nodo]
   / \    / \    / \    / \
  ...  ...  ...  ...  ...  ...
```

**Impacto de la profundidad:**

| Profundidad | Nodos máximos | Complejidad | Riesgo |
|-------------|---------------|-------------|--------|
| 1 | 3 | Muy baja | Underfitting |
| 5 | 63 | Baja | Posible underfitting |
| 10 | 2,047 | Media | Balance |
| 20 | 2,097,151 | Alta | Overfitting |
| ∞ | Ilimitado | Muy alta | Overfitting severo |

---

### 3. Underfitting (Subajuste)

**Definición:**  
Modelo demasiado simple que no captura los patrones en los datos.

**Características:**
- Accuracy bajo en **entrenamiento**
- Accuracy bajo en **validación/test**
- Diferencia pequeña entre train y val
- Modelo "demasiado general"

**Causas:**
- Profundidad muy baja
- Pocas características
- Datos insuficientes
- Modelo inadecuado para el problema

**Ejemplo en MNIST:**
```python
# Profundidad 1 - Underfitting
modelo = DecisionTreeClassifier(max_depth=1)
# Resultado:
# Train accuracy: 0.45
# Val accuracy: 0.43
# → Ambos bajos, no aprende patrones
```

**Síntomas:**
- "El modelo no aprende nada útil"
- Predicciones casi aleatorias
- No mejora con más datos

**Solución:**
- ✓ Aumentar profundidad
- ✓ Agregar más características
- ✓ Usar modelo más complejo
- ✓ Feature engineering

**Gráfica típica:**
```
Accuracy
   |
0.5|  ●────●  (Train)
   |  ●────●  (Val)
   |
   +─────────> Profundidad
     1   2
```

---

### 4. Overfitting (Sobreajuste)

**Definición:**  
Modelo demasiado complejo que memoriza el ruido del entrenamiento en lugar de aprender patrones generales.

**Características:**
- Accuracy **alto** en entrenamiento
- Accuracy **bajo** en validación/test
- **Gran diferencia** entre train y val
- Modelo "memoriza" en vez de "aprender"

**Causas:**
- Profundidad muy alta
- Demasiadas características irrelevantes
- Datos de entrenamiento insuficientes
- Ruido en los datos

**Ejemplo en MNIST:**
```python
# Profundidad 50 - Overfitting
modelo = DecisionTreeClassifier(max_depth=50)
# Resultado:
# Train accuracy: 0.99
# Val accuracy: 0.75
# → Gran diferencia, memoriza training
```

**Síntomas:**
- "Funciona perfecto en train, mal en test"
- Predicciones específicas al training set
- No generaliza a datos nuevos

**Solución:**
- ✓ Reducir profundidad
- ✓ Poda (pruning)
- ✓ Regularización (min_samples_split, min_samples_leaf)
- ✓ Más datos de entrenamiento
- ✓ Cross-validation
- ✓ Ensemble methods (Random Forest)

**Gráfica típica:**
```
Accuracy
   |
1.0|      ●────●  (Train)
   |
0.7|  ●────●      (Val)
   |
   +─────────────> Profundidad
     10  20  30
```

---

### 5. Balance Óptimo (Sweet Spot)

**Definición:**  
Profundidad que maximiza la generalización sin underfitting ni overfitting.

**Características:**
- Accuracy **alto** en validación
- Diferencia **pequeña** entre train y val
- Modelo generaliza bien
- Balance sesgo-varianza

**Cómo encontrarlo:**

1. **Experimentación:**
   ```python
   profundidades = [1, 3, 5, 7, 10, 15, 20, 30]
   for prof in profundidades:
       modelo = DecisionTreeClassifier(max_depth=prof)
       # Evaluar y comparar
   ```

2. **Validación cruzada:**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {'max_depth': range(1, 31)}
   grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
   grid.fit(X_train, y_train)
   mejor_prof = grid.best_params_['max_depth']
   ```

3. **Análisis de curvas:**
   - Graficar accuracy vs profundidad
   - Buscar donde val accuracy es máximo
   - Verificar que diferencia train-val sea pequeña

**Ejemplo ideal:**
```
Profundidad 10:
  Train accuracy: 0.88
  Val accuracy: 0.85
  Diferencia: 0.03 ✓ (pequeña)
```

**Gráfica del balance:**
```
Accuracy
   |
1.0|        ●─────●  (Train)
   |      ●─────●    (Val)
0.8|    ●           
   |  ●             
   +─────────────────> Profundidad
     5   10  15  20
         ↑
      Balance
```

---

## 🎯 Aplicación a MNIST

### Características del Problema

**MNIST:**
- 784 features (píxeles)
- 10 clases (dígitos 0-9)
- Datos de alta dimensionalidad
- Patrones visuales complejos

**Desafíos para árboles:**
- Muchas características → árbol muy grande
- Píxeles correlacionados → redundancia
- Patrones no lineales → difícil de capturar

### Profundidades Recomendadas

| Profundidad | Uso | Resultado Esperado |
|-------------|-----|-------------------|
| 1-3 | Baseline | Underfitting (~40-50% acc) |
| 5-7 | Exploración | Moderado (~70-75% acc) |
| 10-15 | **Óptimo** | **Bueno (~80-85% acc)** |
| 20-30 | Experimental | Overfitting (train>95%, val~80%) |
| >30 | No recomendado | Overfitting severo |

### Comparación con Otros Modelos

| Modelo | Accuracy MNIST | Complejidad | Interpretabilidad |
|--------|----------------|-------------|-------------------|
| Árbol (prof=10) | ~85% | Media | Alta |
| Random Forest | ~95% | Alta | Media |
| SVM | ~98% | Alta | Baja |
| CNN | **>99%** | Muy alta | Muy baja |

**Conclusión:** Árboles son buenos para aprender, pero CNNs son mejores para imágenes.

---

## 📊 Análisis de Métricas

### Accuracy

**Fórmula:**
```
Accuracy = (Predicciones Correctas) / (Total de Predicciones)
```

**Interpretación:**
- 0.90 = 90% de predicciones correctas
- 0.50 = 50% (aleatorio para 2 clases)
- 0.10 = 10% (para 10 clases, aleatorio sería ~10%)

**Limitaciones:**
- No funciona bien con clases desbalanceadas
- No distingue tipos de errores
- Puede ser engañoso

### Diferencia Train-Val

**Fórmula:**
```
Diferencia = Accuracy_Train - Accuracy_Val
```

**Interpretación:**

| Diferencia | Significado |
|------------|-------------|
| < 0.05 | Excelente balance ✓ |
| 0.05 - 0.10 | Buen balance ✓ |
| 0.10 - 0.20 | Overfitting moderado ⚠️ |
| > 0.20 | Overfitting severo ✗ |

**Ejemplo:**
```python
# Modelo A
train_acc = 0.88
val_acc = 0.85
diff = 0.03  # ✓ Excelente

# Modelo B
train_acc = 0.99
val_acc = 0.75
diff = 0.24  # ✗ Overfitting severo
```

---

## 🔧 Técnicas de Mejora

### 1. Poda (Pruning)

**Pre-pruning (antes de entrenar):**
```python
modelo = DecisionTreeClassifier(
    max_depth=10,              # Limitar profundidad
    min_samples_split=20,      # Mínimo para dividir nodo
    min_samples_leaf=10,       # Mínimo en hojas
    max_features='sqrt'        # Limitar features por split
)
```

**Post-pruning (después de entrenar):**
```python
# Cost complexity pruning
path = modelo.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Probar diferentes alphas
for alpha in ccp_alphas:
    modelo_podado = DecisionTreeClassifier(ccp_alpha=alpha)
    # Evaluar
```

### 2. Ensemble Methods

**Random Forest:**
```python
from sklearn.ensemble import RandomForestClassifier

# Múltiples árboles → mejor generalización
rf = RandomForestClassifier(
    n_estimators=100,    # 100 árboles
    max_depth=10,
    random_state=42
)
```

**Ventajas:**
- Reduce overfitting
- Mejora accuracy
- Más robusto

### 3. Feature Engineering

**Para MNIST:**
```python
# PCA para reducir dimensionalidad
from sklearn.decomposition import PCA

pca = PCA(n_components=50)  # 784 → 50 features
X_reduced = pca.fit_transform(X)

# Entrenar con menos features
modelo = DecisionTreeClassifier(max_depth=10)
modelo.fit(X_reduced, y)
```

### 4. Validación Cruzada

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    modelo, X, y, 
    cv=5,              # 5 folds
    scoring='accuracy'
)

print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 💡 Consejos Prácticos

### Para Evitar Underfitting

1. ✓ Aumentar profundidad gradualmente
2. ✓ Verificar que el modelo aprende (train acc > random)
3. ✓ Agregar más características relevantes
4. ✓ Usar modelos más complejos si es necesario

### Para Evitar Overfitting

1. ✓ Usar validación cruzada
2. ✓ Limitar profundidad
3. ✓ Aplicar poda
4. ✓ Aumentar datos de entrenamiento
5. ✓ Regularización (min_samples_*)
6. ✓ Ensemble methods

### Para Encontrar el Balance

1. ✓ Probar múltiples profundidades
2. ✓ Graficar train vs val accuracy
3. ✓ Buscar donde val accuracy es máximo
4. ✓ Verificar diferencia train-val pequeña
5. ✓ Validar en test set final

---

## 📖 Resumen Ejecutivo

| Concepto | Definición Corta | Solución |
|----------|------------------|----------|
| **Underfitting** | Modelo muy simple | Aumentar complejidad |
| **Overfitting** | Modelo muy complejo | Reducir complejidad |
| **Balance** | Complejidad óptima | Experimentar y validar |
| **Profundidad** | Niveles del árbol | Ajustar según datos |
| **Generalización** | Funciona en datos nuevos | Validación cruzada |

**Regla de oro:**  
> "El mejor modelo no es el que mejor funciona en entrenamiento,  
> sino el que mejor generaliza a datos no vistos."

---

## 🎓 Preguntas Frecuentes

**P: ¿Cuál es la mejor profundidad para MNIST?**  
R: Típicamente 10-15, pero depende del tamaño del dataset y otras configuraciones.

**P: ¿Por qué no usar profundidad infinita?**  
R: Causaría overfitting severo. El árbol memorizaría todo el training set.

**P: ¿Los árboles son buenos para imágenes?**  
R: No son ideales. CNNs son mucho mejores (>99% vs ~85% accuracy).

**P: ¿Cómo sé si tengo overfitting?**  
R: Si train accuracy >> val accuracy (diferencia > 0.10).

**P: ¿Puedo usar árboles en producción?**  
R: Sí, pero Random Forest o Gradient Boosting son mejores opciones.

---

**Fin del documento teórico**
