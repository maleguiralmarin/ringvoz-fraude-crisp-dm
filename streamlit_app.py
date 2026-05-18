from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "modelo_riesgo.joblib"
DOMAINS_PATH = ROOT / "artifacts" / "feature_domains.json"
RESULTS_PATH = ROOT / "artifacts" / "resultados_modelado.json"


def load_domains() -> dict:
    if not DOMAINS_PATH.exists():
        return {}
    with open(DOMAINS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_meta() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(
        page_title="RingVoz — Modelo predictivo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RingVoz")
    st.subheader("Interfaz de inferencia del modelo")

    if not MODEL_PATH.exists():
        st.error(
            "No se encontró `artifacts/modelo_riesgo.joblib`. "
            "Ejecute en la raíz del proyecto: `python train_and_export.py`"
        )
        return

    pipe = joblib.load(MODEL_PATH)
    meta = load_meta()
    domains = load_domains()

    with st.sidebar:
        st.header("Modelo")
        if meta:
            st.write(f"**Algoritmo:** {meta.get('mejor_modelo', '—')}")
            st.caption(f"Registros de entrenamiento: {meta.get('n_registros', '—')}")
        st.divider()
        st.header("Datos de entrada")
        amt_min = float(domains.get("_amount_min", 1.0))
        amt_max = float(domains.get("_amount_max", 500.0))
        st.caption("Monto con centavos (paso 0,01).")
        amount = st.number_input(
            "Monto (USD)",
            min_value=amt_min,
            max_value=amt_max,
            value=float(min(25.13, amt_max)),
            step=0.01,
            format="%.2f",
        )
        pmnt = st.selectbox("Método de pago (pmntMethodId)", [1, 2, 3], index=0)
        hour = st.slider("Hora (0–23)", 0, 23, 12)

        def pick(key: str, default: str) -> str:
            opts = domains.get(key) or [default]
            return st.selectbox(key.replace("_", " "), opts, index=0)

        account_type = pick("Account_Type", "Retail")
        transaction_type = pick("Transaction_Type", "Mobile App Recharge")
        merchant = pick("Merchant", "Authorize.Net 2 [Nr. 1469355]")
        transaction_use = pick("Transaction_Use", "Intl Mobile Recharge")
        day_of_week = pick("day_of_week", "Monday")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Resumen de la transacción")
        st.json(
            {
                "Amount": amount,
                "Account_Type": account_type,
                "Transaction_Type": transaction_type,
                "Merchant": merchant,
                "Transaction_Use": transaction_use,
                "pmntMethodId": pmnt,
                "hour_of_day": hour,
                "day_of_week": day_of_week,
            }
        )

    with col_b:
        st.markdown("#### Salida del modelo")
        row = {
            "Account_Type": account_type,
            "Amount": amount,
            "Transaction_Type": transaction_type,
            "Merchant": merchant,
            "Transaction_Use": transaction_use,
            "pmntMethodId": int(pmnt),
            "hour_of_day": int(hour),
            "day_of_week": day_of_week,
        }
        X = pd.DataFrame([row])

        if st.button("Ejecutar predicción", type="primary", use_container_width=True):
            proba = float(pipe.predict_proba(X)[0, 1])
            clase = int(pipe.predict(X)[0])
            st.metric("Probabilidad estimada (clase 1)", f"{proba:.4f}")
            st.metric("Clase predicha", str(clase))
            st.progress(min(max(proba, 0.0), 1.0))
            if proba >= 0.5:
                st.warning("Clase 1 según umbral 0.5 (revisar en contexto operativo).")
            else:
                st.info("Clase 0 según umbral 0.5.")

    casos_path = ROOT / "docs" / "CASOS_PRUEBA_STREAMLIT.txt"
    if casos_path.exists():
        with st.expander("Casos de prueba sugeridos (qué poner y qué esperar)"):
            st.text(casos_path.read_text(encoding="utf-8"))

    if meta:
        with st.expander("Resultados de evaluación (test, entrenamiento previo)"):
            st.json(meta.get("metricas_test", {}))

    st.divider()
    st.caption(
        "Proyecto académico CRISP-DM · José Becerra · Marleny Guiral Marín · Geraldine Suárez"
    )


if __name__ == "__main__":
    main()
