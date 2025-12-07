#!/usr/bin/env python3
"""
Script de prueba para verificar que las gráficas funcionan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TEST DE GRÁFICAS - Asignación 2")
print("="*60)

# Cargar datos
try:
    df = pd.read_csv('balanceSheetHistory_annually.csv')
    print(f"\n✓ Dataset cargado: {df.shape}")
except Exception as e:
    print(f"\n✗ Error cargando dataset: {e}")
    exit(1)

# Seleccionar columnas
df_clean = df[['stock', 'endDate', 'cash']].copy()
df_clean = df_clean.dropna(subset=['cash'])
df_clean['endDate'] = pd.to_datetime(df_clean['endDate'])
df_clean = df_clean.sort_values(['stock', 'endDate']).reset_index(drop=True)

print(f"✓ Datos limpios: {df_clean.shape}")

# Seleccionar 3 empresas
stock_counts = df_clean.groupby('stock').size().sort_values(ascending=False)
selected_stocks = stock_counts.head(3).index.tolist()
df_selected = df_clean[df_clean['stock'].isin(selected_stocks)].copy()

print(f"✓ Empresas seleccionadas: {selected_stocks}")

# Test: Crear gráfica simple
print("\nCreando gráfica de prueba...")

try:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for idx, stock in enumerate(selected_stocks):
        stock_data = df_selected[df_selected['stock'] == stock].sort_values('endDate')
        
        ax = axes[idx]
        ax.plot(stock_data['endDate'], stock_data['cash'], 
                marker='o', linewidth=2, markersize=6)
        ax.set_title(f'Cash - {stock}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Cash (USD)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_grafica.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfica guardada: test_grafica.png")
    plt.close()
    
except Exception as e:
    print(f"✗ Error creando gráfica: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETADO")
print("="*60)
