# 🚀 Guía Rápida - Asignación 3

## ⚡ Inicio Rápido (5 minutos)

### 1. Instalar dependencias
```bash
cd IA/Asignaciones/Asignacion_3
pip install -r requirements.txt
```

### 2. Test rápido
```bash
python test_rapido.py
```

### 3. Ejecutar notebook
```bash
jupyter notebook Asignacion_3_SOLUCION.ipynb
```

Luego: **Cell → Run All**

---

## 📋 Checklist de Entrega

- [ ] Notebook ejecutado completamente
- [ ] Todas las gráficas generadas
- [ ] Tabla comparativa completa
- [ ] Conclusiones escritas
- [ ] Respuestas a preguntas teóricas

---

## 📊 Resultados Esperados

### Gráficas (5 archivos PNG)

1. ✅ `mnist_ejemplos.png` - Ejemplos de dígitos
2. ✅ `desempeno_profundidad.png` - Accuracy vs Profundidad
3. ✅ `arbol_decision.png` - Visualización del árbol
4. ✅ `importancia_pixeles.png` - Mapa de calor
5. ✅ `matriz_confusion.png` - Matriz de confusión

### Tabla de Resultados

| Profundidad | Acc Train | Acc Val | Diferencia |
|-------------|-----------|---------|------------|
| 5 | ~0.75 | ~0.72 | ~0.03 |
| 10 | ~0.88 | ~0.85 | ~0.03 |
| 20 | ~0.99 | ~0.82 | ~0.17 |

---

## 🎯 Puntos Clave para la Presentación

### 1. Introducción (2 min)
- Qué es un árbol de decisión
- Concepto de profundidad
- Underfitting vs Overfitting

### 2. Metodología (2 min)
- Dataset MNIST (784 píxeles, 10 clases)
- 3 profundidades: 5, 10, 20
- División 80/20 estratificada

### 3. Resultados (3 min)
- Mostrar tabla comparativa
- Gráfica de desempeño
- Identificar mejor modelo

### 4. Análisis (2 min)
- Profundidad 5: Posible underfitting
- Profundidad 10: Balance óptimo ✓
- Profundidad 20: Overfitting evidente

### 5. Conclusiones (1 min)
- Mejor profundidad: 10
- Accuracy: ~85%
- Árboles funcionan, pero CNNs son mejores

---

## ❓ Preguntas Frecuentes

**P: ¿Por qué usar solo 3 profundidades?**  
R: Para demostrar claramente underfitting, balance y overfitting.

**P: ¿Puedo usar más datos?**  
R: Sí, pero el entrenamiento será más lento. 10,000 ejemplos es suficiente.

**P: ¿Por qué no profundidad 1?**  
R: Sería demasiado simple (underfitting extremo). Profundidad 5 ya lo demuestra.

**P: ¿Qué accuracy es "bueno"?**  
R: Para árboles en MNIST: 80-85% es bueno. CNNs logran >99%.

**P: ¿Cómo interpreto la diferencia train-val?**  
R: < 0.05 = excelente, 0.05-0.10 = bueno, > 0.10 = overfitting.

---

## 🔧 Troubleshooting

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "File not found: mnist_train.csv"
Verifica que estás en el directorio correcto:
```bash
cd IA/Asignaciones/Asignacion_3
ls  # Debe mostrar mnist_train.csv
```

### Notebook muy lento
Reduce sample_size en el código:
```python
sample_size = 5000  # En vez de 10000
```

### Gráficas no aparecen
Agrega al inicio del notebook:
```python
%matplotlib inline
```

---

## 📚 Recursos Adicionales

- **README.md**: Documentación completa
- **RESPUESTAS_TEORICAS.md**: Conceptos detallados
- **solucion_completa.py**: Script alternativo
- **test_rapido.py**: Verificación rápida

---

## ✅ Criterios de Éxito

### Mínimo (70 pts)
- ✓ Código ejecuta sin errores
- ✓ 3 modelos entrenados
- ✓ Tabla comparativa
- ✓ Conclusiones básicas

### Excelente (90+ pts)
- ✓ Todo lo anterior
- ✓ Todas las gráficas generadas
- ✓ Análisis profundo
- ✓ Respuestas teóricas completas
- ✓ Visualización del árbol
- ✓ Matriz de confusión

---

## 🎓 Tips para Máxima Calificación

1. **Ejecuta TODO el notebook** - No dejes celdas sin ejecutar
2. **Comenta tus observaciones** - Agrega análisis personal
3. **Genera todas las gráficas** - Son parte de la evaluación
4. **Responde las preguntas teóricas** - Demuestra comprensión
5. **Revisa la rúbrica** - Asegúrate de cubrir todos los puntos

---

**Tiempo estimado total: 30-45 minutos**

¡Éxito! 🚀
