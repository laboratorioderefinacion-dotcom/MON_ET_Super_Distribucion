#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# APP STREAMLIT – GASOLINA SUPER INTERIOR (MON_ET) – MODO PRO
# Gradient Boosting + Control de Confiabilidad Metrológica
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACIÓN UI
# ==========================================================

st.set_page_config(
    page_title="MON_ET – Gasolina Super Interior",
    page_icon="🧪",
    layout="centered"
)

st.markdown("""
<style>
a[href^="#"] { display: none !important; }
.block-container { padding-top: 2rem; }
.big-font { font-size:22px !important; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧪 Estimación de MON_ET")
st.markdown("##    Gasolina Super Plantas de Distribución")

# ==========================================================
# CRITERIOS METROLÓGICOS (igual a tu script)
# ==========================================================

REPRO_METODO = 0.83
UMBRAL_METODO = REPRO_METODO / 2
SIGMA_ANALITICO = 0.42
n_sim = 100  # igual que N_SIM

# ==========================================================
# MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    # Asegurate de que estos archivos estén en el repo junto a este .py
    modelo = load("Modelo_MON_ET_Interior.joblib")
    columnas = load("Columnas_MON_ET_Interior.joblib")
    return modelo, columnas

try:
    GBR, columnas_modelo = cargar_modelo()
    st.success("✅ Modelo Gradient Boosting con validación metrológica")
except Exception as e:
    st.error("❌ Error al cargar el modelo o las columnas (.joblib)")
    st.caption(f"Detalle técnico: {e}")
    st.stop()

# ==========================================================
# INPUT
# ==========================================================

archivo = st.file_uploader("📁 Cargar archivo CSV del LIMS", type=["csv"])

# ==========================================================
# FUNCIONES
# ==========================================================

def extraer_valor(df, nombre):
    fila = df[df[1] == nombre]
    if fila.empty:
        return np.nan
    return fila.iloc[0, 4]

def convertir_a_float(v):
    if pd.isna(v):
        return np.nan
    try:
        return float(str(v).replace(",", "."))
    except:
        return np.nan

def seleccionar_densidad(muestra, sampling_point):
    # Igual que tu lógica:
    # - TABLADA usa "Densidad promedio a 15º"
    # - otros usan "Densidad a 15ºC"
    if sampling_point == "R-TK_TABLADA":
        return extraer_valor(muestra, "Densidad promedio a 15º")
    else:
        return extraer_valor(muestra, "Densidad a 15ºC")

def armar_df_pred(muestra, columnas_modelo, sampling_point):
    densidad = seleccionar_densidad(muestra, sampling_point)

    datos = {
        'DENSIDAD': densidad,
        'PUNTO INICIAL': extraer_valor(muestra, "IBP"),
        'T5':  extraer_valor(muestra, "5% vol"),
        'T10': extraer_valor(muestra, "10% vol"),
        'T20': extraer_valor(muestra, "20% vol"),
        'T30': extraer_valor(muestra, "30% vol"),
        'T40': extraer_valor(muestra, "40% vol"),
        'T50': extraer_valor(muestra, "50% vol"),
        'T60': extraer_valor(muestra, "60% vol"),
        'T70': extraer_valor(muestra, "70% vol"),
        'T80': extraer_valor(muestra, "80% vol"),
        'T90': extraer_valor(muestra, "90% vol"),
        'T95': extraer_valor(muestra, "95% vol"),
        'PUNTO FINAL': extraer_valor(muestra, "Punto Final")
    }

    datos_convertidos = {k: convertir_a_float(v) for k, v in datos.items()}

    df_pred = pd.DataFrame([datos_convertidos])
    df_pred = df_pred.reindex(columns=columnas_modelo)

    return df_pred, datos_convertidos

def monte_carlo_std(modelo, df_base, n_sim, sigma):
    """
    Igual que tu código original:
    ruido ~ Normal(0, SIGMA_ANALITICO * 0.05)
    """
    preds = []
    for _ in range(n_sim):
        df_sim = df_base.copy()
        for col in df_sim.columns:
            ruido = np.random.normal(0, sigma * 0.05)
            df_sim[col] = df_sim[col] + ruido
        preds.append(modelo.predict(df_sim)[0])
    return np.asarray(preds).std()

# ==========================================================
# BOTÓN PRINCIPAL
# ==========================================================

if archivo is not None:

    if st.button("🚀 Calcular MON_ET"):

        with st.spinner("Procesando muestra..."):

            # Leer CSV
            try:
                muestra = pd.read_csv(archivo, sep=";", encoding="latin1", header=None)
            except Exception as e:
                st.error("❌ No se pudo leer el CSV (separador/encoding/formato)")
                st.caption(f"Detalle técnico: {e}")
                st.stop()

            # Campos clave
            try:
                celda_producto = muestra.loc[muestra[0] == "Producto", 4].values[0]
                celda_lims = muestra.loc[muestra[0] == "Número de Muestra", 4].values[0]
                celda_sp = muestra.loc[muestra[0] == "SamplingPoint", 4].values[0]
            except Exception:
                st.error("❌ Formato de archivo inválido (faltan campos: Producto / Número de Muestra / SamplingPoint)")
                st.stop()

            # Validación: este modelo NO es para FINAL TEJA
            if celda_sp == "R-TK_FINAL_TEJA":
                st.error("❌ Este modelo es para Plantas de Distribución (Interior).")
                st.warning("El archivo corresponde a TK FINAL TEJA. Use la app/modelo de La Teja.")
                st.stop()

            # Validación producto
            if celda_producto != "GAS_SUP_95":
                st.error("❌ La muestra NO corresponde a GASOLINA SUPER (GAS_SUP_95).")
                st.warning(f"Producto encontrado: {celda_producto}")
                st.stop()

            # Armar DF
            df_pred, datos_convertidos = armar_df_pred(muestra, columnas_modelo, celda_sp)

            # Chequeo faltantes
            faltantes = [k for k, v in datos_convertidos.items() if (isinstance(v, float) and np.isnan(v))]
            if faltantes:
                st.error("❌ Datos incompletos. No se puede estimar MON_ET con confiabilidad.")
                st.warning("Faltan ensayos / variables:")
                st.write(", ".join(faltantes))
                st.stop()

            # Predicción
            try:
                mon_et_estimado = float(GBR.predict(df_pred)[0])
                mon_et_estimado = np.round(mon_et_estimado, 1)
            except Exception as e:
                st.error("❌ Error al predecir (revisar columnas/valores).")
                st.caption(f"Detalle técnico: {e}")
                st.stop()

            # Monte Carlo (std)
            mon_et_std = monte_carlo_std(
                modelo=GBR,
                df_base=df_pred,
                n_sim=n_sim,
                sigma=SIGMA_ANALITICO
            )
            error_reportado = np.round(mon_et_std, 2)

            # Semáforo
            if mon_et_std <= UMBRAL_METODO:
                color = "green"
                estado = "ALTA CONFIABILIDAD"
                icono = "🟢"
            else:
                color = "red"
                estado = "BAJA CONFIABILIDAD"
                icono = "🔴"

            # ======================================================
            # RESULTADO VISUAL PRO
            # ======================================================
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔢 MON_ET estimado")

                if mon_et_std < UMBRAL_METODO:
                    valor = str(mon_et_estimado).replace(".", ",")
                    color_val = "black"
                else:
                    valor = "❌"
                    color_val = "red"

                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        font-size: 34px;
                        font-weight: bold;
                        color: {color_val};
                    ">
                        {valor}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(f"### 📋 LIMS: {celda_lims}")

            st.markdown("---")

            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h2 style="color:{color};">{icono} {estado}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

