from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
ARFF_PATH = ROOT / "datos" / "dataset_fraude_ringvoz_class_last.arff"
ARTIFACTS = ROOT / "artifacts"
TARGET = "is_fraud"
RANDOM_STATE = 42


def load_arff_to_df(path: Path) -> pd.DataFrame:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [
        "Account_Type",
        "Transaction_Type",
        "Merchant",
        "Transaction_Use",
        "day_of_week",
    ]
    num_cols = ["Amount", "pmntMethodId", "hour_of_day"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )


def metrics_dict(y_true, y_proba, y_pred) -> dict:
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall_clase_1": float(recall_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    return out


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_arff_to_df(ARFF_PATH)
    y = (df[TARGET].astype(str) == "1").astype(int)
    X = df.drop(columns=[TARGET])
    cat_cols = [
        "Account_Type",
        "Transaction_Type",
        "Merchant",
        "Transaction_Use",
        "day_of_week",
    ]
    cat_cols = [c for c in cat_cols if c in X.columns]

    domains = {c: sorted(X[c].astype(str).unique().tolist()) for c in cat_cols}
    domains["_amount_min"] = float(X["Amount"].min())
    domains["_amount_max"] = float(X["Amount"].max())
    with open(ARTIFACTS / "feature_domains.json", "w", encoding="utf-8") as f:
        json.dump(domains, f, ensure_ascii=False, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pre = build_preprocessor(X_train)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg / max(pos, 1))

    models: dict[str, Pipeline] = {
        "RandomForest": Pipeline(
            [
                ("prep", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=18,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", build_preprocessor(X_train)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=600,
                        max_depth=5,
                        min_child_weight=2,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.2,
                        gamma=0.05,
                        scale_pos_weight=spw,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("prep", build_preprocessor(X_train)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48),
                        activation="relu",
                        max_iter=300,
                        alpha=5e-4,
                        learning_rate_init=0.001,
                        batch_size=2048,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                        validation_fraction=0.12,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        ),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        results[name] = metrics_dict(y_test.to_numpy(), proba, pred)
        results[name]["report"] = classification_report(
            y_test,
            pred,
            digits=3,
            target_names=["clase_0", "clase_1"],
            zero_division=0,
        )

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_pipe = models[best_name]
    best_pipe.fit(X_train, y_train)

    joblib.dump(best_pipe, ARTIFACTS / "modelo_riesgo.joblib")
    meta = {
        "mejor_modelo": best_name,
        "motivo": "Mayor PR-AUC en conjunto de test (clase minoritaria).",
        "metricas_test": {
            k: {m: v for m, v in results[k].items() if m != "report"}
            for k in results
        },
        "n_registros": int(len(df)),
        "positivos_clase_1": int(y.sum()),
        "variable_objetivo": "is_fraud (clase 1 = positivos en el ARFF de entrenamiento).",
    }
    with open(ARTIFACTS / "resultados_modelado.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for name in results:
        p = ARTIFACTS / f"report_{name}.txt"
        p.write_text(results[name]["report"], encoding="utf-8")

    print("Mejor modelo:", best_name)
    print(json.dumps(meta["metricas_test"], indent=2))
    print("Artefactos en:", ARTIFACTS)


if __name__ == "__main__":
    main()
