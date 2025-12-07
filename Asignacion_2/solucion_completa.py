"""
Solución completa para Asignación 2 - Regresión Lineal
Predicción de cash por empresa usando series temporales
"""

# ============================================================================
# PASO 1: Reconocer el dataset
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Cargar dataset
df = pd.read_csv('balanceSheetHistory_annually.csv')

# Explorar estructura
print("="*60)
print("PASO 1: RECONOCIMIENTO DEL DATASET")
print("="*60)
print(f"\nDimensiones: {df.shape}")
print(f"\nPrimeras filas:")
print(df.head())
print(f"\nInformación del dataset:")
print(df.info())
print(f"\nEstadísticas descriptivas:")
print(df.describe())

# Documentación de columnas
columnas_descripcion = {
    'stock': 'Ticker/símbolo bursátil de la empresa (string)',
    'endDate': 'Fecha de cierre del periodo fiscal (YYYY-MM-DD, anual)',
    'accountsPayable': 'Cuentas por pagar (float, USD)',
    'inventory': 'Inventario (float, USD)',
    'longTermDebt': 'Deuda a largo plazo (float, USD)',
    'netReceivables': 'Cuentas por cobrar netas (float, USD)',
    'netTangibleAssets': 'Activos tangibles netos (float, USD)',
    'longTermInvestments': 'Inversiones a largo plazo (float, USD)',
    'totalCurrentAssets': 'Activos corrientes totales (float, USD)',
    'propertyPlantEquipment': 'Propiedad, planta y equipo (float, USD)',
    'otherStockholderEquity': 'Otro capital accionario (float, USD)',
    'deferredLongTermAssetCharges': 'Cargos diferidos de activos LP (float, USD)',
    'totalCurrentLiabilities': 'Pasivos corrientes totales (float, USD)',
    'cash': 'Efectivo y equivalentes (float, USD) - VARIABLE OBJETIVO',
    'otherAssets': 'Otros activos (float, USD)',
    'treasuryStock': 'Acciones en tesorería (float, USD)',
    'goodWill': 'Plusvalía/Good will (float, USD)',
    'otherLiab': 'Otros pasivos (float, USD)',
    'retainedEarnings': 'Utilidades retenidas (float, USD)',
    'otherCurrentAssets': 'Otros activos corrientes (float, USD)',
    'commonStock': 'Acciones comunes (float, USD)',
    'totalAssets': 'Activos totales (float, USD)',
    'otherCurrentLiab': 'Otros pasivos corrientes (float, USD)',
    'deferredLongTermLiab': 'Pasivos diferidos a largo plazo (float, USD)',
    'totalStockholderEquity': 'Capital contable total (float, USD)',
    'totalLiab': 'Pasivos totales (float, USD)',
    'capitalSurplus': 'Superávit de capital (float, USD)',
    'intangibleAssets': 'Activos intangibles (float, USD)',
    'shortTermInvestments': 'Inversiones a corto plazo (float, USD)',
    'shortLongTermDebt': 'Deuda a corto/largo plazo (float, USD)',
    'minorityInterest': 'Interés minoritario (float, USD)'
}

print("\n" + "="*60)
print("DOCUMENTACIÓN DE COLUMNAS")
print("="*60)
for col, desc in columnas_descripcion.items():
    print(f"• {col}: {desc}")

# ============================================================================
# PASO 2: Seleccionar columnas relevantes
# ============================================================================

print("\n" + "="*60)
print("PASO 2: SELECCIÓN DE COLUMNAS")
print("="*60)

# Verificar que las columnas existen
required_cols = ['stock', 'endDate', 'cash']
print(f"\nColumnas requeridas: {required_cols}")
print(f"¿Existen todas? {all(col in df.columns for col in required_cols)}")

# Seleccionar solo las columnas necesarias
df_clean = df[required_cols].copy()
print(f"\nDataset reducido: {df_clean.shape}")
print(f"\nPrimeras filas:")
print(df_clean.head(10))

# Verificar valores nulos
print(f"\nValores nulos por columna:")
print(df_clean.isnull().sum())

# Eliminar filas con valores nulos en cash
df_clean = df_clean.dropna(subset=['cash'])
print(f"\nDespués de eliminar nulos en 'cash': {df_clean.shape}")

# ============================================================================
# PASO 3: Separar por empresas y elegir 3
# ============================================================================

print("\n" + "="*60)
print("PASO 3: SELECCIÓN DE 3 EMPRESAS")
print("="*60)

# Convertir endDate a datetime
df_clean['endDate'] = pd.to_datetime(df_clean['endDate'])

# Ordenar por stock y fecha
df_clean = df_clean.sort_values(['stock', 'endDate']).reset_index(drop=True)

# Contar observaciones por empresa
stock_counts = df_clean.groupby('stock').size().sort_values(ascending=False)
print(f"\nTop 10 empresas con más observaciones:")
print(stock_counts.head(10))

# Seleccionar 3 empresas con suficientes datos
selected_stocks = stock_counts.head(3).index.tolist()
print(f"\nEmpresas seleccionadas: {selected_stocks}")

# Filtrar solo las 3 empresas
df_selected = df_clean[df_clean['stock'].isin(selected_stocks)].copy()

# Mostrar información de cada empresa
for stock in selected_stocks:
    stock_data = df_selected[df_selected['stock'] == stock]
    print(f"\n{stock}:")
    print(f"  - Observaciones: {len(stock_data)}")
    print(f"  - Rango de fechas: {stock_data['endDate'].min()} a {stock_data['endDate'].max()}")
    print(f"  - Cash promedio: ${stock_data['cash'].mean():,.2f}")
    print(f"  - Cash min/max: ${stock_data['cash'].min():,.2f} / ${stock_data['cash'].max():,.2f}")

# ============================================================================
# PASO 4: Graficar tiempo vs dinero
# ============================================================================

print("\n" + "="*60)
print("PASO 4: VISUALIZACIÓN DE SERIES TEMPORALES")
print("="*60)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for idx, stock in enumerate(selected_stocks):
    stock_data = df_selected[df_selected['stock'] == stock].sort_values('endDate')
    
    ax = axes[idx]
    ax.plot(stock_data['endDate'], stock_data['cash'], marker='o', linewidth=2, markersize=6)
    ax.set_title(f'Serie Temporal de Cash - {stock}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Cash (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Formatear eje y con separadores de miles
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('cash_series_temporales.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficas guardadas en 'cash_series_temporales.png'")
plt.show()

# ============================================================================
# PASO 5: División 80/20
# ============================================================================

print("\n" + "="*60)
print("PASO 5: DIVISIÓN TEMPORAL 80/20")
print("="*60)

# Diccionarios para almacenar train/test por empresa
train_data = {}
test_data = {}

for stock in selected_stocks:
    stock_data = df_selected[df_selected['stock'] == stock].sort_values('endDate').reset_index(drop=True)
    
    # Calcular punto de corte (80%)
    split_idx = int(len(stock_data) * 0.8)
    
    train_data[stock] = stock_data.iloc[:split_idx].copy()
    test_data[stock] = stock_data.iloc[split_idx:].copy()
    
    print(f"\n{stock}:")
    print(f"  - Total: {len(stock_data)} observaciones")
    print(f"  - Train: {len(train_data[stock])} observaciones ({len(train_data[stock])/len(stock_data)*100:.1f}%)")
    print(f"  - Test: {len(test_data[stock])} observaciones ({len(test_data[stock])/len(stock_data)*100:.1f}%)")
    print(f"  - Última fecha train: {train_data[stock]['endDate'].max()}")
    print(f"  - Primera fecha test: {test_data[stock]['endDate'].min()}")

# ============================================================================
# PASO 6: Crear y entrenar modelos
# ============================================================================

print("\n" + "="*60)
print("PASO 6: ENTRENAMIENTO DE MODELOS")
print("="*60)

models = {}
predictions = {}

for stock in selected_stocks:
    print(f"\nEntrenando modelo para {stock}...")
    
    # Preparar features temporales
    train = train_data[stock].copy()
    test = test_data[stock].copy()
    
    # Crear features: índice temporal (días desde inicio)
    train['days_since_start'] = (train['endDate'] - train['endDate'].min()).dt.days
    test['days_since_start'] = (test['endDate'] - train['endDate'].min()).dt.days
    
    # Features adicionales
    train['year'] = train['endDate'].dt.year
    train['quarter'] = train['endDate'].dt.quarter
    test['year'] = test['endDate'].dt.year
    test['quarter'] = test['endDate'].dt.quarter
    
    # Preparar X e y
    feature_cols = ['days_since_start', 'year', 'quarter']
    X_train = train[feature_cols]
    y_train = train['cash']
    X_test = test[feature_cols]
    y_test = test['cash']
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Guardar
    models[stock] = model
    predictions[stock] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'dates_test': test['endDate']
    }
    
    print(f"  ✓ Modelo entrenado")
    print(f"  - Coeficientes: {model.coef_}")
    print(f"  - Intercepto: {model.intercept_:,.2f}")

# ============================================================================
# PASO 7: Graficar real vs predicho
# ============================================================================

print("\n" + "="*60)
print("PASO 7: VISUALIZACIÓN REAL VS PREDICHO")
print("="*60)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for idx, stock in enumerate(selected_stocks):
    pred_data = predictions[stock]
    
    ax = axes[idx]
    ax.plot(pred_data['dates_test'], pred_data['y_test'], 
            marker='o', linewidth=2, markersize=8, label='Real', color='blue')
    ax.plot(pred_data['dates_test'], pred_data['y_pred'], 
            marker='s', linewidth=2, markersize=6, label='Predicho', 
            color='red', linestyle='--', alpha=0.7)
    
    ax.set_title(f'Real vs Predicho - {stock}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Cash (USD)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('real_vs_predicho.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficas guardadas en 'real_vs_predicho.png'")
plt.show()

# ============================================================================
# PASO 8: Métricas de evaluación
# ============================================================================

print("\n" + "="*60)
print("PASO 8: MÉTRICAS DE EVALUACIÓN")
print("="*60)

results = []

for stock in selected_stocks:
    pred_data = predictions[stock]
    y_test = pred_data['y_test']
    y_pred = pred_data['y_pred']
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Stock': stock,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    })
    
    print(f"\n{stock}:")
    print(f"  - MSE:  {mse:,.2f}")
    print(f"  - RMSE: {rmse:,.2f}")
    print(f"  - R²:   {r2:.4f}")

# Crear DataFrame de resultados
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)
print(results_df.to_string(index=False))

# Conclusión
print("\n" + "="*60)
print("CONCLUSIÓN")
print("="*60)
best_r2 = results_df.loc[results_df['R²'].idxmax()]
worst_r2 = results_df.loc[results_df['R²'].idxmin()]

print(f"\n✓ Mejor modelo: {best_r2['Stock']} con R² = {best_r2['R²']:.4f}")
print(f"✗ Peor modelo: {worst_r2['Stock']} con R² = {worst_r2['R²']:.4f}")
print(f"\nInterpretación:")
print(f"- RMSE indica el error promedio en USD")
print(f"- R² cercano a 1 indica buen ajuste")
print(f"- R² negativo o cercano a 0 indica mal ajuste")
print(f"\nRecomendaciones para mejorar:")
print(f"- Agregar más features (rezagos, tendencias)")
print(f"- Probar modelos no lineales (polinomial, ARIMA)")
print(f"- Considerar variables exógenas del balance")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
