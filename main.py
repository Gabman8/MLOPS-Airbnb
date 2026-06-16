import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import joblib
from datetime import datetime

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# =========================
# CONFIGURACIÓN DE LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURACIÓN DE MLFLOW
# =========================

EXPERIMENT_NAME = "Airbnb-Price-Prediction"
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Cambia si usas un servidor remoto
MODEL_REGISTRY_NAME = "AirbnbRandomForestModel"

# Crear directorio para modelos si no existe
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Configurar MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

logger.info("========== INICIANDO PIPELINE MLOPS ==========")
logger.info(f"Experimento: {EXPERIMENT_NAME}")

# =========================
# 1. CARGA DE DATOS
# =========================

logger.info("Iniciando carga de datos...")
airbnb_data = pd.read_csv("./airbnb-listings-extract.csv", sep=";", decimal='.')
logger.info(f"Datos cargados: {airbnb_data.shape[0]} filas, {airbnb_data.shape[1]} columnas")

# =========================
# 2. LIMPIEZA DE PRICE
# =========================

logger.info("Limpiando variable Price...")
initial_rows = len(airbnb_data)
airbnb_data['Price'] = (
    airbnb_data['Price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
)

airbnb_data['Price'] = pd.to_numeric(airbnb_data['Price'], errors='coerce')
airbnb_data = airbnb_data.dropna(subset=['Price'])
removed_rows = initial_rows - len(airbnb_data)
logger.info(f"Filas removidas por valores nulos en Price: {removed_rows}")

# =========================
# 3. LIMPIEZA DE %
# =========================

logger.info("Limpiando variables de porcentaje...")
for col in ['Host Response Rate', 'Host Acceptance Rate']:
    airbnb_data[col] = (
        airbnb_data[col]
        .astype(str)
        .str.replace('%','')
    )
    airbnb_data[col] = pd.to_numeric(airbnb_data[col], errors='coerce')

# =========================
# 4. FEATURES
# =========================

logger.info("Seleccionando features...")
columns = [
    'Accommodates',
    'Bathrooms',
    'Bedrooms',
    'Square Feet',
    'Room Type',
    'Latitude',
    'Longitude',
    'Review Scores Rating',
    'Number of Reviews',
    'Reviews per Month',
    'Availability 30',
    'Availability 90',
    'Availability 365',
    'Host Listings Count',
    'Host Total Listings Count',
]

X = airbnb_data[columns]
y = airbnb_data['Price']
logger.info(f"Features seleccionadas: {len(columns)}")
logger.info(f"Tamaño dataset: X={X.shape}, y={y.shape}")

# =========================
# 5. LOG TRANSFORM
# =========================

logger.info("Aplicando log transform a target...")
y = np.log1p(y)
logger.info(f"Log transform completado. Target stats - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")

# =========================
# INICIAR RUN DE MLFLOW
# =========================

with mlflow.start_run(run_name=f"airbnb_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # =========================
    # 6. TRAIN / TEST SPLIT
    # =========================
    
    logger.info("Realizando train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    logger.info(f"Train set: {X_train.shape[0]} muestras")
    logger.info(f"Test set: {X_test.shape[0]} muestras")
    
    # =========================
    # 7. ELIMINACIÓN DE OUTLIERS 
    # =========================
    
    logger.info("Eliminando outliers...")
    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    initial_train_size = len(y_train)
    mask = (y_train >= lower) & (y_train <= upper)
    
    X_train = X_train[mask]
    y_train = y_train[mask]
    outliers_removed = initial_train_size - len(y_train)
    logger.info(f"Outliers eliminados: {outliers_removed} ({(outliers_removed/initial_train_size*100):.2f}%)")
    
    # =========================
    # 8. ONE HOT ENCODING
    # =========================
    
    logger.info("Aplicando one-hot encoding...")
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Alinear columnas
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    logger.info(f"Features después de encoding: {X_train.shape[1]}")
    
    # =========================
    # 9. IMPUTACIÓN
    # =========================
    
    logger.info("Aplicando imputación de valores faltantes...")
    imputer = SimpleImputer(strategy="median")
    
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    logger.info("Imputación completada")
    
    # =========================
    # 10. MODELO + GRID SEARCH
    # =========================
    
    logger.info("Iniciando GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
    
    # Log de parámetros
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("cv_folds", 3)
    mlflow.log_dict(param_grid, "param_grid.json")
    
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    
    logger.info("GridSearchCV completado")
    logger.info(f"Mejores parámetros: {grid.best_params_}")
    
    # Log de mejores parámetros
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # =========================
    # 11. PREDICCIÓN
    # =========================
    
    logger.info("Realizando predicciones...")
    pred = model.predict(X_test)
    
    pred_real = np.expm1(pred)
    y_test_real = np.expm1(y_test)
    logger.info("Predicciones completadas")

# =========================
# 12. EVALUACIÓN
# =========================
    
    logger.info("Evaluando modelo...")
    mae = mean_absolute_error(y_test_real, pred_real)
    r2 = r2_score(y_test_real, pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred_real))
    
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R2: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    
    # Log de métricas a MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("RMSE", rmse)
    
    # =========================
    # 13. CROSS VALIDATION
    # =========================
    
    logger.info("Realizando validación cruzada...")
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    
    cv_mae_mean = -scores.mean()
    cv_mae_std = scores.std()
    logger.info(f"MAE medio CV: {cv_mae_mean:.4f} (+/- {cv_mae_std:.4f})")
    
    mlflow.log_metric("CV_MAE_mean", cv_mae_mean)
    mlflow.log_metric("CV_MAE_std", cv_mae_std)
    
    # =========================
    # 14. GRÁFICO REAL VS PREDICHO
    # =========================
    
    logger.info("Generando gráficos...")
    
    # Gráfico 1: Real vs Predicho
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_real, pred_real, alpha=0.5, label='Predicciones')
    plt.plot(
        [y_test_real.min(), y_test_real.max()],
        [y_test_real.min(), y_test_real.max()],
        'r--',
        lw=2,
        label='Predicción Perfecta'
    )
    plt.xlabel("Precio Real ($)", fontsize=12)
    plt.ylabel("Precio Predicho ($)", fontsize=12)
    plt.title("Real vs Predicho", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path_1 = "plots/real_vs_predicho.png"
    plt.savefig(plot_path_1, dpi=100)
    mlflow.log_artifact(plot_path_1)
    logger.info(f"Gráfico guardado: {plot_path_1}")
    plt.close()
    
    # =========================
    # 15. RESIDUOS
    # =========================
    
    residuals = y_test_real - pred_real
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_real, residuals, alpha=0.5, label='Residuos')
    plt.axhline(0, color='r', linestyle='--', lw=2, label='Línea Base')
    plt.xlabel("Precio Predicho ($)", fontsize=12)
    plt.ylabel("Error ($)", fontsize=12)
    plt.title("Análisis de Residuos", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path_2 = "plots/residuos.png"
    plt.savefig(plot_path_2, dpi=100)
    mlflow.log_artifact(plot_path_2)
    logger.info(f"Gráfico guardado: {plot_path_2}")
    plt.close()
    
    # =========================
    # 16. IMPORTANCIA VARIABLES
    # =========================
    
    importances = model.feature_importances_
    
    feature_importance = pd.Series(
        importances,
        index=X_train.columns
    )
    
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values().tail(15).plot.barh()
    plt.xlabel("Importancia", fontsize=12)
    plt.title("Variables más importantes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path_3 = "plots/importancia_variables.png"
    plt.savefig(plot_path_3, dpi=100)
    mlflow.log_artifact(plot_path_3)
    logger.info(f"Gráfico guardado: {plot_path_3}")
    plt.close()
    
    # =========================
    # 17. GUARDANDO MODELO
    # =========================
    
    logger.info("Guardando modelo...")
    model_path = f"models/airbnb_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Modelo guardado en: {model_path}")
    
    # Log del modelo en MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log de artefactos adicionales
    mlflow.log_artifact(model_path)
    
    # =========================
    # RESUMEN FINAL
    # =========================
    
    summary_metrics = {
        "MAE": mae,
        "R2": r2,
        "RMSE": rmse,
        "CV_MAE_mean": cv_mae_mean,
        "CV_MAE_std": cv_mae_std,
        "Train_size": len(X_train),
        "Test_size": len(X_test),
        "Features": len(X_train.columns),
        "Outliers_removed": outliers_removed
    }
    
    logger.info("\n" + "="*50)
    logger.info("RESUMEN DEL EXPERIMENTO")
    logger.info("="*50)
    for metric, value in summary_metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    logger.info("="*50)
    logger.info(f"Experimento completado exitosamente")
    logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
    logger.info("="*50)
