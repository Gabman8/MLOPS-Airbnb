# MLOPS - Airbnb Price Prediction Pipeline

## 📋 Descripción General

Este proyecto implementa un pipeline de **Machine Learning Operations (MLOps)** para la predicción de precios de Airbnb. Utiliza **MLflow** para el seguimiento de experimentos, el registro de métricas y la gestión del ciclo de vida del modelo, siguiendo las mejores prácticas de despliegue de algoritmos de ML en producción.

### 🎯 Objetivos del MLOps

Los objetivos principales son:

1. **Automatización del despliegue**: Implementación reproducible de modelos en entornos de producción
2. **Gestión del ciclo de vida**: Gestionar eficientemente todo el ciclo de vida del modelo
3. **Colaboración y comunicación**: Facilitar la colaboración entre equipos de desarrollo
4. **Monitorización y mantenimiento continuo**: Evaluar el rendimiento del modelo en tiempo real

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────┐
│  Data Extraction    │
│  (CSV Cargado)      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Data Preprocessing  │
│ - Limpieza          │
│ - Feature Eng.      │
│ - Codificación      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Train/Test Split   │
│  - 80/20 Split      │
│  - Outlier Removal  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Model Training     │
│  - GridSearchCV     │
│  - Random Forest    │
│  - MLflow Tracking  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Evaluation         │
│  - MAE, R2, RMSE    │
│  - Cross-Validation │
│  - Visualización    │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Model Registry     │
│  - MLflow Registry  │
│  - Artifact Storage │
│  - Logging          │
└─────────────────────┘
```

---

## 📦 Instalación

### Requisitos Previos

- Python 3.8+
- pip o conda
- Virtual environment (recomendado)

### Paso 1: Crear y Activar Entorno Virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 2: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Configurar MLflow (Opcional)

Para usar MLflow Server en lugar de archivo local:

```bash
# Instalar MLflow Server
pip install mlflow[postgres]

# Iniciar servidor MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Acceder a interfaz web
# http://localhost:5000
```

---

## 🚀 Ejecución

### Ejecutar Pipeline Completo

```bash
python main.py
```

El script ejecutará:

1. ✅ Carga y limpieza de datos
2. ✅ Ingeniería de características
3. ✅ Entrenamiento con GridSearchCV
4. ✅ Evaluación del modelo
5. ✅ Registro de métricas en MLflow
6. ✅ Generación de visualizaciones
7. ✅ Almacenamiento del modelo

---

## 📊 Métricas y Seguimiento

### Métricas Registradas en MLflow

El pipeline registra automáticamente:

#### Parámetros

- `test_size`: Porcentaje de datos para test (0.2)
- `random_state`: Seed para reproducibilidad (42)
- `cv_folds`: Número de folds en validación cruzada (3)
- `best_n_estimators`: Mejor número de árboles
- `best_max_depth`: Mejor profundidad máxima
- `best_min_samples_split`: Mejor número mínimo de muestras

#### Métricas

| Métrica         | Descripción                              |
| --------------- | ---------------------------------------- |
| **MAE**         | Error Absoluto Medio (Dollar difference) |
| **R2**          | Coeficiente de Determinación (0-1)       |
| **RMSE**        | Raíz del Error Cuadrático Medio          |
| **CV_MAE_mean** | MAE medio en validación cruzada          |
| **CV_MAE_std**  | Desviación estándar del MAE en CV        |

#### Artefactos

- `real_vs_predicho.png`: Gráfico de predicciones vs valores reales
- `residuos.png`: Análisis de residuos
- `importancia_variables.png`: Importancia de características
- `airbnb_model_YYYYMMDD_HHMMSS.pkl`: Modelo entrenado

---

## 📁 Estructura de Archivos

```
MLOPS-Airbnb/
├── main.py                              # Script principal del pipeline
├── requirements.txt                     # Dependencias del proyecto
├── README.md                           # Esta documentación
├── airbnb-listings-extract.csv         # Dataset de Airbnb
├── mlops_pipeline.log                  # Logs detallados de ejecución
│
├── models/                             # Directorio de modelos guardados
│   └── airbnb_model_YYYYMMDD_HHMMSS.pkl
│
├── plots/                              # Visualizaciones generadas
│   ├── real_vs_predicho.png
│   ├── residuos.png
│   └── importancia_variables.png
│
└── mlruns/                             # MLflow experiment tracking
    └── (Generado automáticamente por MLflow)
```

---

## 🔍 Detalles de Implementación

### 1. Logging Centralizado

El pipeline implementa logging en dos niveles:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
```

**Ventajas:**

- Trazabilidad completa de la ejecución
- Logs guardados en archivo para auditoría
- Información en consola para monitoreo en tiempo real

### 2. MLflow Integration

El pipeline envuelve todo el proceso de entrenamiento en un contexto de MLflow:

```python
with mlflow.start_run(run_name=f"airbnb_training_{timestamp}"):
    # Parámetros, métricas y artefactos se registran automáticamente
```

**Características:**

- Rastreo automático de parámetros e hiperparámetros
- Registro de todas las métricas de evaluación
- Guardado de modelos para reproducibilidad
- Interfaz web para comparación de experimentos

### 3. Preprocesamiento de Datos

#### Limpieza

- Conversión de formato de precios (string → float)
- Manejo de porcentajes en variables de host
- Eliminación de valores nulos

#### Ingeniería de Características

- 15 features seleccionadas manualmente
- One-hot encoding para variables categóricas
- Transformación logarítmica del target (log1p)

#### Manejo de Outliers

```python
# IQR method (Interquartile Range)
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

**Resultado:** Típicamente se eliminan ~5-10% de outliers

#### Imputación

```python
imputer = SimpleImputer(strategy="median")
```

### 4. Entrenamiento del Modelo

**Algoritmo:** Random Forest Regressor

**Hyperparameter Tuning:**

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
```

**Grid Search Configuration:**

- Cross-validation: 5-fold CV
- Scoring: neg_mean_absolute_error
- Parallelization: n_jobs=-1 (todos los cores disponibles)

### 5. Evaluación del Modelo

#### Test Set Metrics

- **MAE**: Métrica principal (en precios reales)
- **R2**: Proporción de varianza explicada
- **RMSE**: Raíz del error cuadrático medio

#### Cross-Validation

```python
# 5-fold cross-validation para validar robustez
scores = cross_val_score(model, X_train, y_train, cv=5)
cv_mae_mean = -scores.mean()
cv_mae_std = scores.std()
```

---

## 📈 Visualizaciones Generadas

### 1. Real vs Predicho

- **Propósito:** Evaluar calidad de predicciones
- **Interpretación:** Puntos cercanos a la línea diagonal indican predicciones precisas

### 2. Análisis de Residuos

- **Propósito:** Detectar sesgos y varianza heterogénea
- **Interpretación:** Residuos distribuidos aleatoriamente alrededor de cero = buen ajuste

### 3. Importancia de Variables

- **Propósito:** Identificar features más influyentes
- **Top 15 features:** Mostrados en gráfico de barras horizontal

---

## 🔄 Ciclo de Vida del Modelo (MLOps)

```
1. EXPERIMENTACIÓN (Esta ejecución)
   ├─ Entrenamiento con múltiples configs
   ├─ Registro de parámetros y métricas
   ├─ Evaluación y validación
   └─ Persistencia de artefactos

2. VALIDACIÓN
   ├─ Comparación de experimentos en MLflow UI
   ├─ Análisis de rendimiento
   └─ Selección del mejor modelo

3. STAGING
   ├─ Registro en MLflow Model Registry
   ├─ Pruebas en entorno de staging
   └─ Validación final

4. PRODUCCIÓN
   ├─ Despliegue en cloud (GCP Cloud Run, AWS)
   ├─ Monitorización continua
   ├─ Reentrenamiento automático si es necesario
   └─ Rollback en caso de degradación

5. MONITORIZACIÓN
   ├─ Seguimiento de datos de entrada
   ├─ Detección de data drift
   ├─ Alertas si rendimiento baja
   └─ Recolección de feedback
```

---

## 💾 Persistencia y Reproducibilidad

### Reproducibilidad Garantizada

- **Random State**: Fijado a 42 en todos los componentes estocásticos
- **Versioning**: Cada ejecución crea run único con timestamp
- **Versionado de Código**: Git para rastrear cambios en pipeline
- **Artifact Logging**: Todos los artefactos guardados en MLflow

### Cómo Reproductibilizar

```bash
# Acceder a MLflow UI
mlflow ui

# Seleccionar run específico
# Descargar modelo y parámetros exactos
# Ejecutar con los mismos datos
```

---

## 🛠️ Hardware Recomendado

**Tipo de Inferencia:** Batch Processing (recomendado para este dataset)

| Recurso        | Mínimo       | Recomendado | Producción |
| -------------- | ------------ | ----------- | ---------- |
| CPU            | 2 cores      | 4+ cores    | 8+ cores   |
| RAM            | 4 GB         | 8 GB        | 16+ GB     |
| GPU            | No necesario | Optional    | RTX 3060+  |
| Almacenamiento | 5 GB         | 20 GB       | 100+ GB    |

**Nota:** Random Forest requiere CPU principalmente. GPUs son útiles para Deep Learning.

---

## ☁️ Despliegue en Cloud (Próximos Pasos)

### Google Cloud Platform (GCP)

```bash
# 1. Exportar modelo desde MLflow
mlflow models serve -m "models:/AirbnbRandomForestModel/production"

# 2. Crear Dockerfile
# 3. Desplegar en Cloud Run
gcloud run deploy airbnb-predictor \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2
```

### AWS

```bash
# Usar SageMaker para hosting
# Crear endpoint REST
# Configurar auto-scaling
```

---

## 📝 Logs Disponibles

El sistema genera dos tipos de logs:

### 1. Console Output (Tiempo Real)

```
2024-01-15 10:30:45 - __main__ - INFO - Cargando datos...
2024-01-15 10:30:46 - __main__ - INFO - Datos cargados: 5000 filas
...
```

### 2. File Logs (`mlops_pipeline.log`)

```
2024-01-15 10:30:45,123 - __main__ - INFO - Iniciando pipeline MLOPS
2024-01-15 10:30:46,456 - __main__ - INFO - GridSearchCV completado
...
```

---

## 🔧 Troubleshooting

### Problema: MLflow no se conecta

```bash
# Solución: Verificar servidor MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### Problema: Memoria insuficiente

```bash
# Solución: Reducir tamaño de batch o GridSearch
# Cambiar param_grid en main.py
```

### Problema: Modelos no se guardan

```bash
# Verificar permisos en directorio models/
chmod -R 755 models/
```

---

## 📚 Referencias y Recursos

### Documentación Oficial

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

### Conceptos MLOps

1. **Experiment Tracking**: Registro automático de parámetros y métricas
2. **Model Registry**: Versionado y registro de modelos en producción
3. **Data Versioning**: Control de versiones de datasets
4. **Model Monitoring**: Seguimiento de rendimiento en producción
5. **CI/CD Pipelines**: Automatización de deployment

### Mejores Prácticas

- ✅ Siempre registrar parámetros y métricas
- ✅ Versionar datos y modelos
- ✅ Mantener logs detallados
- ✅ Implementar validación cruzada
- ✅ Monitorizar en producción
- ❌ Evitar hardcoding de rutas
- ❌ No ignorar outliers sin análisis
- ❌ Olvidar documentar cambios

---

## 👥 Autoría y Contribuciones

**Creado por:** KeepCoding MLOps Masterclass
**Adaptado para:** Airbnb Price Prediction
**Fecha:** 2024

---

## 📄 Licencia

Este proyecto está disponible bajo licencia MIT.

---

## 🎓 Conclusión

Este pipeline demuestra los **4 objetivos principales del MLOps**:

1. ✅ **Automatización**: El pipeline se ejecuta completamente sin intervención manual
2. ✅ **Gestión del Ciclo de Vida**: MLflow maneja experimentos, modelos y artefactos
3. ✅ **Colaboración**: Logs y MLflow UI facilitanCompartir resultados con equipos
4. ✅ **Monitorización**: Sistema de logging y métricas para seguimiento continuo

**Próximos Pasos:**

- 🚀 Desplegar modelo en GCP Cloud Run
- 📊 Implementar dashboard de monitorización
- 🔄 Configurar reentrenamiento automático
- 📈 Escalar a múltiples modelos
