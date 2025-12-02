# ============================================
# Predicción diaria de PM2.5 (mediana del día siguiente)
# con Random Forest Regressor
# ============================================

# ==============
# 1) Librerías
# ==============
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ================================
# 0) Configuración de ejecución
# ================================
INPUT_CSV = "PM25_diario.csv"     # Cambia al nombre real de tu archivo
OUTPUT_DIR = "outputs"            # Carpeta de resultados
SEED = 42
TEST_SIZE = 0.2                   # 20% final para test (split temporal)
N_ESTIMATORS = 400
MAX_DEPTH = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================================
# 2) Comprensión y Carga de Datos (CRISP-DM 5.2)
# ===============================================
print(">> Cargando datos...")
df = pd.read_csv(INPUT_CSV)

# Asegurar nombres de columnas esperados
expected_cols = ['date','min','max','median','q1','q3','stdev','count']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Columnas faltantes en CSV: {missing}\nSe esperaban: {expected_cols}")

# Convertir fecha ISO 8601 a datetime y ordenar
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
# quitar tz si se desea trabajar horario "naive"
df['date'] = df['date'].dt.tz_convert(None)
df = df.sort_values('date').reset_index(drop=True)

print(">> Vista rápida de datos:")
print(df.head())

# ==================================================
# 3) Limpieza / Preparación de Datos (CRISP-DM 5.3)
# ==================================================
# Convertir numéricos por si alguna columna llega como texto
num_cols = ['min','max','median','q1','q3','stdev','count']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Interpolación lineal de faltantes, y forward/backward fill como respaldo
df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')

# Eliminar filas que sigan con NaN (muy raro si los datos son completos)
df = df.dropna(subset=num_cols)

# ======= Ingeniería de variables de tiempo =======
df['year']       = df['date'].dt.year
df['month']      = df['date'].dt.month
df['dayofweek']  = df['date'].dt.dayofweek    # 0=Lunes
df['dayofyear']  = df['date'].dt.dayofyear
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# ======= Objetivo: mediana del día siguiente =======
df['median_next'] = df['median'].shift(-1)

# (Opcional) Lags y rolling (mejoran series temporales)
# Si tu dataset es suficientemente largo, estas ayudan:
df['median_lag1'] = df['median'].shift(1)
df['max_lag1']    = df['max'].shift(1)
df['stdev_lag1']  = df['stdev'].shift(1)
df['median_roll3'] = df['median'].rolling(3).mean()
df['median_roll7'] = df['median'].rolling(7).mean()

# Eliminar últimas filas sin 'median_next' y las primeras con lags/rolling NaN
df = df.dropna().reset_index(drop=True)

# =======================
# 4) Conjunto de Modelado
# =======================
# Selección de variables predictoras (puedes ajustar)
features = [
    'min','max','q1','q3','stdev','count',
    'month','dayofweek','dayofyear','is_weekend',
    'median_lag1','max_lag1','stdev_lag1','median_roll3','median_roll7'
]
target = 'median_next'

X = df[features].copy()
y = df[target].copy()

# ================================
# 5) División Temporal Train/Test
# ================================
# Para series temporales es mejor separar cronológicamente:
n = len(df)
split_index = int(np.floor((1 - TEST_SIZE) * n))

X_train, X_test = X.iloc[:split_index, :], X.iloc[split_index:, :]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f">> Registros totales: {n} | Train: {len(X_train)} | Test: {len(X_test)}")

# ==========================
# 6) Modelado (CRISP-DM 5.4)
# ==========================
rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=SEED,
    n_jobs=-1,
    oob_score=False
)

print(">> Entrenando Random Forest...")
rf.fit(X_train, y_train)

# ===========================
# 7) Evaluación (CRISP-DM 5.5)
# ===========================
print(">> Evaluando modelo...")
y_pred = rf.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))

print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

# Guardar métricas
with open(os.path.join(OUTPUT_DIR, "metricas_rf.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Guardar predicciones vs reales
pred_df = pd.DataFrame({
    "date": df['date'].iloc[split_index:].values,
    "real": y_test.values,
    "predicho": y_pred
})
pred_path = os.path.join(OUTPUT_DIR, "predicciones_rf_PM25.csv")
pred_df.to_csv(pred_path, index=False)
print(f">> Predicciones guardadas en: {pred_path}")

# ===========================
# 8) Visualizaciones y Reporte
# ===========================
# a) Dispersión Real vs Predicho
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Real (PM2.5 median día+1)")
plt.ylabel("Predicción (PM2.5 median día+1)")
plt.title("Random Forest - Real vs Predicho (test)")
plt.grid(alpha=0.3)
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "scatter_real_vs_predicho.png")
plt.savefig(fig1_path, dpi=160)
plt.close()

# b) Serie temporal (últimos N puntos del set de prueba)
N = min(120, len(pred_df))  # mostrar últimos ~4 meses si hay datos diarios
plt.figure(figsize=(10,4))
plt.plot(pred_df['date'].iloc[-N:], pred_df['real'].iloc[-N:], label="Real")
plt.plot(pred_df['date'].iloc[-N:], pred_df['predicho'].iloc[-N:], label="Predicho")
plt.title(f"Serie temporal - últimos {N} días (test)")
plt.xlabel("Fecha")
plt.ylabel("PM2.5 (mediana)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "serie_temporal_real_vs_predicho.png")
plt.savefig(fig2_path, dpi=160)
plt.close()

# c) Importancia de variables
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(7,6))
plt.barh(importances.index, importances.values)
plt.title("Importancia de variables - Random Forest")
plt.xlabel("Importancia")
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "importancia_variables.png")
plt.savefig(fig3_path, dpi=160)
plt.close()

print(f">> Figuras guardadas en: {fig1_path}, {fig2_path}, {fig3_path}")

# ==================================
# 9) Implementación (Despliegue 5.6)
# ==================================
# Guardar el modelo entrenado para uso posterior
model_path = os.path.join(OUTPUT_DIR, "modelo_random_forest_PM25.pkl")
joblib.dump(rf, model_path)
print(f">> Modelo guardado en: {model_path}")

# Guardar conjunto de features utilizado (para reproducibilidad)
with open(os.path.join(OUTPUT_DIR, "features_usadas.json"), "w", encoding="utf-8") as f:
    json.dump({"features": features, "target": target}, f, indent=2, ensure_ascii=False)

print(">> Implementación terminada.")
print("Resumen de métricas:", metrics)

