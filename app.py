"""Segmentación de clientes con K-Means – Despliegue en Streamlit.

Ejecutar con:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "Mall_Customers.xls"
FEATURES = ["Ingreso", "Gasto"]
RANDOM_STATE = 42

DESCRIPCIONES = {
    "Alto valor": "Ingreso alto y gasto alto. Estrategia: fidelización premium, beneficios VIP y atención personalizada.",
    "Promedio": "Ingreso y gasto medios. Estrategia: promociones regulares, puntos y venta cruzada para subir el ticket.",
    "Bajo valor": "Ingreso bajo y gasto bajo. Estrategia: ofertas por precio, descuentos y marcas propias.",
    "Con potencial": "Ingreso alto pero gasto bajo. Estrategia: oferta premium, experiencias personalizadas y conocer por qué no compran.",
    "Gastador impulsivo": "Ingreso bajo pero gasto alto (jóvenes). Estrategia: marketing digital, tendencias accesibles y fidelización temprana.",
}
COLORES = {
    "Alto valor": "#2ca02c",
    "Promedio": "#1f77b4",
    "Bajo valor": "#7f7f7f",
    "Con potencial": "#ff7f0e",
    "Gastador impulsivo": "#d62728",
}


def asignar_perfil(ingreso, gasto):
    """Clasifica un centroide (en unidades originales) en un perfil de negocio."""
    if ingreso >= 70 and gasto >= 60:
        return "Alto valor"
    if ingreso >= 70 and gasto < 40:
        return "Con potencial"
    if ingreso < 40 and gasto < 40:
        return "Bajo valor"
    if ingreso < 40 and gasto >= 60:
        return "Gastador impulsivo"
    return "Promedio"


@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={
        "Annual Income (k$)": "Ingreso",
        "Spending Score (1-100)": "Gasto",
        "Age": "Edad",
        "Gender": "Genero",
    })


@st.cache_resource
def entrenar(k):
    df = cargar_datos()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].values)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
    perfiles = {c: asignar_perfil(r.Ingreso, r.Gasto) for c, r in centroides.iterrows()}
    return scaler, kmeans, centroides, perfiles, silhouette_score(X_scaled, kmeans.labels_)


@st.cache_data
def curvas_evaluacion():
    df = cargar_datos()
    X_scaled = StandardScaler().fit_transform(df[FEATURES].values)
    filas = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_) if k > 1 else None
        filas.append({"K": k, "Inercia": km.inertia_, "Silueta": sil})
    return pd.DataFrame(filas)


st.set_page_config(page_title="Segmentación de clientes", page_icon="🛍️", layout="wide")
st.title("🛍️ Segmentación de clientes con K-Means")
st.caption("Mall Customers Dataset · Variables: ingreso anual (k$) y puntaje de gasto (1-100), estandarizadas con StandardScaler.")

df = cargar_datos()
curvas = curvas_evaluacion()

with st.sidebar:
    st.header("Configuración")
    k = st.slider("Número de clusters (K)", 2, 10, 5, help="El codo y la silueta indican K = 5.")
    st.markdown("---")
    st.header("Nuevo cliente")
    ingreso_nuevo = st.number_input("Ingreso anual (k$)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
    gasto_nuevo = st.slider("Puntaje de gasto (1-100)", 1, 100, 20)
    clasificar = st.button("Clasificar cliente", type="primary", width="stretch")

scaler, kmeans, centroides, perfiles, sil = entrenar(k)
df["Cluster"] = kmeans.labels_
df["Perfil"] = df["Cluster"].map(perfiles)

tab_codo, tab_clusters, tab_nuevo = st.tabs(["📉 Método del codo", "🎯 Clusters", "🧑 Clasificar cliente"])

# ---------------------------------------------------------------- Codo
with tab_codo:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(curvas, x="K", y="Inercia", markers=True, title="Método del codo (inercia / WCSS)")
        fig.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="Codo K=5")
        fig.add_vline(x=k, line_dash="dot", line_color="gray", annotation_text=f"K elegido={k}",
                      annotation_position="bottom right")
        st.plotly_chart(fig, width="stretch")
    with col2:
        sil_df = curvas.dropna()
        k_best = int(sil_df.loc[sil_df["Silueta"].idxmax(), "K"])
        fig = px.line(sil_df, x="K", y="Silueta", markers=True, title="Coeficiente de silueta")
        fig.add_vline(x=k_best, line_dash="dash", line_color="red", annotation_text=f"Máximo K={k_best}")
        st.plotly_chart(fig, width="stretch")
    st.info(f"El codo de la curva aparece en **K = 5** y la silueta máxima también está en **K = {k_best}** "
            f"({sil_df['Silueta'].max():.3f}). Ambos métodos coinciden.")
    st.dataframe(curvas.round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Clusters
with tab_clusters:
    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters", k)
    m2.metric("Coeficiente de silueta", f"{sil:.3f}")
    m3.metric("Inercia", f"{kmeans.inertia_:.1f}")

    color_por = st.radio("Colorear por", ["Perfil", "Cluster"], horizontal=True)
    plot_df = df.assign(Cluster=df["Cluster"].astype(str))
    fig = px.scatter(
        plot_df, x="Ingreso", y="Gasto", color=color_por,
        color_discrete_map=COLORES if color_por == "Perfil" else None,
        hover_data=["CustomerID", "Edad", "Genero", "Cluster", "Perfil"],
        labels={"Ingreso": "Ingreso anual (k$)", "Gasto": "Puntaje de gasto (1-100)"},
        title=f"Clientes segmentados (K = {k})", height=550,
    )
    fig.add_trace(go.Scatter(x=centroides["Ingreso"], y=centroides["Gasto"], mode="markers",
                             marker=dict(symbol="x", size=16, color="black"), name="Centroides"))
    st.plotly_chart(fig, width="stretch")

    resumen = df.groupby("Cluster").agg(
        Perfil=("Perfil", "first"),
        Clientes=("CustomerID", "count"),
        Edad_media=("Edad", "mean"),
        Ingreso_medio=("Ingreso", "mean"),
        Gasto_medio=("Gasto", "mean"),
    ).round(1)
    st.subheader("Resumen por cluster")
    st.dataframe(resumen, width="stretch")

    if k == 5:
        st.subheader("Interpretación de negocio")
        for perfil in DESCRIPCIONES:
            st.markdown(f"- **{perfil}:** {DESCRIPCIONES[perfil]}")
    else:
        st.warning("Los perfiles de negocio se definieron para K = 5. Con otro K, la etiqueta de perfil es solo "
                   "aproximada (se asigna según el ingreso y gasto del centroide).")

# ---------------------------------------------------------------- Nuevo cliente
with tab_nuevo:
    if not clasificar:
        st.write("Ingresa los datos del cliente en la barra lateral y presiona **Clasificar cliente**.")
    else:
        nuevo = scaler.transform([[ingreso_nuevo, gasto_nuevo]])
        cluster = int(kmeans.predict(nuevo)[0])
        perfil = perfiles[cluster]

        st.success(f"El cliente pertenece al **Cluster {cluster}** → perfil **{perfil}**")
        st.markdown(DESCRIPCIONES[perfil])

        fig = px.scatter(df, x="Ingreso", y="Gasto", color="Perfil", color_discrete_map=COLORES,
                         opacity=0.45, height=500,
                         labels={"Ingreso": "Ingreso anual (k$)", "Gasto": "Puntaje de gasto (1-100)"})
        fig.add_trace(go.Scatter(x=centroides["Ingreso"], y=centroides["Gasto"], mode="markers",
                                 marker=dict(symbol="x", size=14, color="black"), name="Centroides"))
        fig.add_trace(go.Scatter(x=[ingreso_nuevo], y=[gasto_nuevo], mode="markers",
                                 marker=dict(symbol="star", size=24, color="gold",
                                             line=dict(width=2, color="black")),
                                 name="Nuevo cliente"))
        st.plotly_chart(fig, width="stretch")

        c = centroides.loc[cluster]
        st.caption(f"Centroide del cluster {cluster}: ingreso ≈ {c.Ingreso:.1f} k$, gasto ≈ {c.Gasto:.1f}.")
