import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans


def afiseaza_clusterizare(df_final_scaled):
    st.subheader("Clusterizare cu K-Means")
    st.write("Modelul grupează țările în clustere pe baza indicatorilor socio-economici.")

    coloane_features = [
        "PIB/capita",
        "ELECTRICITATE",
        "URBAN",
        "SERVICII",
        "DENSIT.POP."
    ]

    coloane_lipsa = [col for col in coloane_features if col not in df_final_scaled.columns]

    if len(coloane_lipsa) > 0:
        st.error(f"Lipsesc coloanele necesare clusterizării: {coloane_lipsa}")
        return

    df_model = df_final_scaled.copy()
    X = df_model[coloane_features].dropna()

    st.write("Variabile folosite pentru clusterizare:")
    st.write(coloane_features)

    nr_clustere = st.selectbox("Alege numărul de clustere:", [2, 3, 4, 5], index=1)

    model = KMeans(n_clusters=nr_clustere, random_state=42, n_init=10)
    etichete = model.fit_predict(X)

    df_cluster = df_model.loc[X.index].copy()
    df_cluster["Cluster"] = etichete.astype(str)

    st.write("Primele 10 rânduri după clusterizare:")
    coloane_afisare = ["TARA"] + coloane_features if "TARA" in df_cluster.columns else coloane_features
    coloane_afisare = [col for col in coloane_afisare if col in df_cluster.columns]
    coloane_afisare.append("Cluster")
    st.dataframe(df_cluster[coloane_afisare].head(10))

    distributie_cluster = df_cluster["Cluster"].value_counts().reset_index()
    distributie_cluster.columns = ["Cluster", "Număr țări"]

    fig_distributie = px.bar(
        distributie_cluster,
        x="Cluster",
        y="Număr țări",
        title="Distribuția observațiilor pe clustere"
    )
    st.plotly_chart(fig_distributie, use_container_width=True)

    if "TARA" in df_cluster.columns:
        fig_scatter = px.scatter(
            df_cluster,
            x="PIB/capita",
            y="ELECTRICITATE",
            color="Cluster",
            hover_name="TARA",
            title="Clusterizare: PIB/capita vs ELECTRICITATE"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        fig_scatter = px.scatter(
            df_cluster,
            x="PIB/capita",
            y="ELECTRICITATE",
            color="Cluster",
            title="Clusterizare: PIB/capita vs ELECTRICITATE"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    centre = pd.DataFrame(model.cluster_centers_, columns=coloane_features)
    centre["Cluster"] = [str(i) for i in range(nr_clustere)]

    st.write("Centrele clusterelor:")
    st.dataframe(centre)