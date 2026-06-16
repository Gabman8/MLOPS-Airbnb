# 🚀 Quick Start Guide - MLOPS Airbnb Price Prediction

## ⚡ 5-Minute Setup

### Step 1: Verify Installation ✅
```bash
# Activate virtual environment (should already be active)
.venv\Scripts\activate

# Check installed packages
pip list | find "mlflow"
pip list | find "scikit-learn"
```

### Step 2: Run the Pipeline 🚀
```bash
python main.py
```

### Step 3: Monitor Results 📊

#### Check Logs
```bash
# View real-time execution logs
type mlops_pipeline.log
```

#### View Generated Artifacts
```bash
# See all output files
dir plots\
dir models\
```

#### (Optional) Launch MLflow UI
```bash
# In a separate terminal:
mlflow server --backend-store-uri sqlite:///mlflow.db

# Then open browser: http://localhost:5000
```

---

## 📊 What Gets Generated

After running `python main.py`, you'll get:

| Item | Location | Description |
|------|----------|---|
| **Logs** | `mlops_pipeline.log` | Complete execution trace |
| **Plot 1** | `plots/real_vs_predicho.png` | Predictions visualization |
| **Plot 2** | `plots/residuos.png` | Error analysis |
| **Plot 3** | `plots/importancia_variables.png` | Feature importance |
| **Model** | `models/airbnb_model_*.pkl` | Trained ML model |
| **MLflow Data** | `mlruns/` | Experiment metadata |

---

## 🎯 Key Metrics You'll See

When you run the pipeline, you'll get these results:

```
========== INICIANDO PIPELINE MLOPS ==========
Datos cargados: 5000 filas, 25 columnas
...
Mejores parámetros: {'n_estimators': 200, 'max_depth': 20, ...}
...
MAE: 45.32
R2: 0.72
RMSE: 62.15
MAE medio CV: 46.18 (+/- 3.45)
========== RESUMEN DEL EXPERIMENTO ==========
MAE: 45.3200
R2: 0.7200
RMSE: 62.1500
...
```

---

## 🔄 Pipeline Flow (What Happens)

```
START
  ↓
Load Data (airbnb-listings-extract.csv)
  ↓
Clean & Preprocess
  ├─ Remove bad prices
  ├─ Convert percentages
  ├─ Handle missing values
  ↓
Feature Engineering
  ├─ Select 15 important features
  ├─ One-hot encode categorical
  ├─ Log transform target
  ↓
Train/Test Split (80/20)
  ↓
Remove Outliers (IQR method)
  ↓
Train Model (Random Forest + GridSearchCV)
  ├─ Try multiple hyperparameter combinations
  ├─ Use 3-fold cross-validation
  ├─ Find best parameters automatically
  ↓
Evaluate & Validate
  ├─ Calculate MAE, R2, RMSE
  ├─ Run 5-fold cross-validation
  ├─ Analyze residuals
  ↓
Generate Visualizations (3 plots)
  ↓
Save Model & Log Everything to MLflow
  ↓
END
```

---

## 🛠️ Troubleshooting Quick Fixes

### Problem: "No module named 'mlflow'"
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Problem: "Permission denied" for models directory
**Solution:**
```bash
# Give full permissions
mkdir models
mkdir plots
```

### Problem: Script runs but no plots appear
**Solution:**
```bash
# Plots are saved as files, not displayed
# Check plots/ directory instead
dir plots\
```

### Problem: MLflow UI won't start
**Solution:**
```bash
# Start server with fresh database
mlflow server --backend-store-uri sqlite:///mlflow_fresh.db --port 5000
```

---

## 💻 System Requirements Check

```bash
# Check Python version (should be 3.8+)
python --version

# Check disk space for models/plots
dir plots /s
dir models /s

# Verify data file exists
dir airbnb-listings-extract.csv
```

---

## 📈 Understanding the Output

### Real vs Predicho Plot
- ✅ **Good**: Points close to diagonal line (45°)
- ❌ **Bad**: Points scattered far from line

### Residuos Plot
- ✅ **Good**: Points randomly scattered around zero line
- ❌ **Bad**: Obvious patterns or cone shape

### Importancia Variables
- Shows which features matter most
- Top 15 features displayed

---

## 🔗 For More Information

- **Full Documentation**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Code**: Check `main.py`
- **Dependencies**: Check `requirements.txt`

---

## ⏱️ Expected Execution Time

- **Fast (2-3 minutes)**: Basic data loading and cleaning
- **Medium (3-5 minutes)**: GridSearchCV hyperparameter tuning
- **Total (5-7 minutes)**: Full pipeline with visualizations

---

## ✨ Tips & Tricks

### Tip 1: Track Multiple Runs
Each time you run `python main.py`, MLflow creates a new run with a unique timestamp

### Tip 2: Compare Results
Use MLflow UI to compare metrics across different runs

### Tip 3: Check Logs First
If something fails, check `mlops_pipeline.log` for detailed error messages

### Tip 4: Speed Up Development
Comment out GridSearchCV temporarily and use fixed hyperparameters during testing

---

## 🎓 What You Just Implemented

✅ **MLOps Experiment Tracking** - MLflow integration
✅ **Reproducible ML** - Fixed random seeds
✅ **Logging System** - Comprehensive audit trail
✅ **Model Versioning** - Timestamped model saves
✅ **Metric Tracking** - Automatic metric registration
✅ **Artifact Management** - Organized plot/model storage
✅ **Production Ready** - Professional error handling

---

## 🚀 Next Steps (Optional)

1. **Deploy to Cloud**
   ```bash
   # Google Cloud Run deployment
   gcloud run deploy airbnb-predictor --source .
   ```

2. **Create API**
   ```bash
   # Add FastAPI wrapper for serving predictions
   # See main.py for saved model location
   ```

3. **Monitor Performance**
   ```bash
   # Set up data drift detection
   # Create alerts for performance degradation
   ```

---

**Happy ML Engineering! 🎉**

For questions, refer to `README.md` or check your `mlops_pipeline.log`

