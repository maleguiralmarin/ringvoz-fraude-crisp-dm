# RingVoz — Detección de Fraude (CRISP-DM)

Proyecto de minería de datos para detección de transacciones fraudulentas en la plataforma RingVoz, siguiendo la metodología **CRISP-DM**. Incluye modelos predictivos (supervisado) y segmentación de transacciones (no supervisado).

---

## Estructura del proyecto

```
ringvoz-fraude-crisp-dm/
├── datos/
│   ├── mining_data_clean.arff          # Dataset crudo (89,890 registros, 24 columnas)
│   └── dataset_fraude_ringvoz_class_last.arff  # Dataset limpio y con features engineered
├── notebooks/
│   ├── fraude_ringvoz.ipynb            # Preparación de datos y feature engineering
│   ├── clustering_ringvoz.ipynb        # Análisis de clustering (no supervisado)
│   └── modelado_riesgo_ringvoz.ipynb   # Modelado predictivo (supervisado)
├── artifacts/
│   ├── modelo_riesgo.joblib            # Pipeline entrenado (RandomForest)
│   ├── feature_domains.json            # Dominios categóricos para la app
│   ├── resultados_modelado.json        # Métricas de los 3 modelos evaluados
│   ├── resultados_clustering.json      # Métricas de validación del clustering
│   ├── clusters_kmeans.csv             # Asignación de cluster por transacción
│   ├── report_RandomForest.txt
│   ├── report_XGBoost.txt
│   └── report_MLP.txt
├── streamlit_app.py                    # Aplicación de inferencia interactiva
├── train_and_export.py                 # Script de entrenamiento y exportación
└── requirements.txt
```

---

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `streamlit`, `joblib`, `matplotlib`.

---

## Ejecución

### 1. Entrenamiento del modelo

```bash
python train_and_export.py
```

Entrena tres modelos (RandomForest, XGBoost, MLP), selecciona el mejor por **PR-AUC** y exporta los artefactos a `artifacts/`.

### 2. Aplicación de inferencia (Streamlit)

```bash
streamlit run streamlit_app.py
```

Abre una interfaz web para ingresar los datos de una transacción y obtener la predicción de riesgo de fraude.

### 3. Notebooks (orden recomendado)

1. `fraude_ringvoz.ipynb` — limpieza, exploración y feature engineering
2. `clustering_ringvoz.ipynb` — segmentación no supervisada de transacciones
3. `modelado_riesgo_ringvoz.ipynb` — entrenamiento y evaluación de modelos predictivos

---

## Resultados

### Modelo predictivo (supervisado)

| Modelo | ROC-AUC | PR-AUC | Recall clase 1 |
|---|---|---|---|
| **RandomForest** | 0.9079 | **0.0122** | 0.25 |
| XGBoost | 0.8449 | 0.0120 | 0.25 |
| MLP | 0.5332 | 0.0002 | 0.00 |

**Modelo seleccionado: RandomForest** por mayor PR-AUC.
> El PR-AUC es la métrica clave dado el extremo desbalance de clases: solo 17 casos positivos (fraude) sobre 89,890 registros.

### Clustering (no supervisado)

| Métrica | K-Means (k=2) | Jerárquico (Ward) |
|---|---|---|
| Silhouette | 0.254 | 0.252 |
| Davies-Bouldin | 1.273 | 1.266 |
| Calinski-Harabasz | 20,863 | — |

**k óptimo = 2** — Cluster 0 (82%): transacciones con tarjeta. Cluster 1 (18%): pagos Wallet. Los 17 casos de fraude están en el Cluster 0 (esperado: las reglas de fraude aplican solo a tarjetas).

---

## Variable objetivo

`is_fraud` — clase binaria:
- `0`: transacción legítima
- `1`: transacción fraudulenta (positivo)

---

## Datos

El dataset crudo (`mining_data_clean.arff`, ~120 MB) no se incluye en el repositorio por su tamaño. El dataset procesado (`dataset_fraude_ringvoz_class_last.arff`) sí está disponible y es suficiente para reproducir el entrenamiento del modelo.
