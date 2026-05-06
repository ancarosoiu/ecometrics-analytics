import streamlit as st
import plotly.express as px


def afiseaza_analiza_descriptiva(df_raw, df_final):
    st.header("Analiză Descriptivă")
    st.write("Această secțiune ne ajută să înțelegem cum arată datele înainte și după preprocesare.")

    tab1, tab2 = st.tabs(["Date brute", "Date preprocesate"])

    with tab1:
        st.subheader("1. Structura setului de date brut")
        st.write(
            "Datele brute sunt afișate complet în pagina dedicată `Date Brute`. "
            "Aici păstrăm doar informații descriptive despre structura lor."
        )

        st.info(f"Setul brut are {df_raw.shape[0]} rânduri și {df_raw.shape[1]} coloane.")

        st.subheader("2. Tipurile de date")
        tipuri_date_raw = df_raw.dtypes.reset_index()
        tipuri_date_raw.columns = ["Coloană", "Tip de date"]
        st.dataframe(tipuri_date_raw, use_container_width=True)

        st.subheader("3. Valori lipsă în datele brute")
        valori_lipsa = df_raw.isna().sum().reset_index()
        valori_lipsa.columns = ["Coloană", "Număr valori lipsă"]
        valori_lipsa = valori_lipsa[valori_lipsa["Număr valori lipsă"] > 0]

        if valori_lipsa.empty:
            st.success("Nu există valori lipsă în setul de date brut.")
        else:
            st.warning("Există valori lipsă în setul de date brut.")
            st.dataframe(valori_lipsa, use_container_width=True)

    with tab2:
        st.subheader("1. Setul de date după preprocesare")
        st.write("Primele 10 rânduri din setul de date preprocesat:")
        st.dataframe(df_final.head(10))

        st.subheader("2. Statistici descriptive după preprocesare")
        st.dataframe(df_final.describe())

        coloane_numerice_final = df_final.select_dtypes(include="number").columns.tolist()

        if len(coloane_numerice_final) > 0:
            st.subheader("3. Histogramă pentru date preprocesate")
            variabila_hist_final = st.selectbox(
                "Alege o variabilă pentru histogramă (date preprocesate):",
                coloane_numerice_final,
                key="hist_final"
            )
            fig_hist_final = px.histogram(
                df_final,
                x=variabila_hist_final,
                title=f"Distribuția variabilei {variabila_hist_final} (date preprocesate)"
            )
            st.plotly_chart(fig_hist_final, use_container_width=True)

            st.subheader("4. Boxplot pentru o variabilă numerică")
            variabila_box = st.selectbox(
                "Alege o variabilă pentru boxplot:",
                coloane_numerice_final,
                key="box_final"
            )
            fig_box = px.box(
                df_final,
                y=variabila_box,
                title=f"Boxplot pentru {variabila_box}"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        if len(coloane_numerice_final) > 1:
            st.subheader("5. Corelații între variabile")
            matrice_corelatie = df_final[coloane_numerice_final].corr()
            st.dataframe(matrice_corelatie)

            st.subheader("6. Scatter plot")
            col1, col2 = st.columns(2)

            with col1:
                x_var = st.selectbox("Alege variabila X:", coloane_numerice_final, key="x_var")

            with col2:
                y_var = st.selectbox("Alege variabila Y:", coloane_numerice_final, key="y_var")

            fig_scatter = px.scatter(
                df_final,
                x=x_var,
                y=y_var,
                hover_name="TARA" if "TARA" in df_final.columns else None,
                title=f"{y_var} în funcție de {x_var}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        if "EMISII CO2" in df_final.columns and "TARA" in df_final.columns:
            st.subheader("7. Top 5 țări după EMISII CO2")
            top_emisii = df_final[["TARA", "EMISII CO2"]].sort_values(
                by="EMISII CO2", ascending=False
            ).head(5)
            st.dataframe(top_emisii)

        if "PIB/capita" in df_final.columns and "TARA" in df_final.columns:
            st.subheader("8. Top 5 țări după PIB/capita")
            top_pib = df_final[["TARA", "PIB/capita"]].sort_values(
                by="PIB/capita", ascending=False
            ).head(5)
            st.dataframe(top_pib)

        if "Emisii_Ridicate" in df_final.columns:
            st.subheader("9. Distribuția claselor pentru clasificare")
            distributie_clase = df_final["Emisii_Ridicate"].value_counts().reset_index()
            distributie_clase.columns = ["Clasa", "Număr țări"]

            fig_clase = px.bar(
                distributie_clase,
                x="Clasa",
                y="Număr țări",
                title="Distribuția variabilei Emisii_Ridicate"
            )
            st.plotly_chart(fig_clase, use_container_width=True)

        if "Emisii_Ridicate" in df_final.columns:
            st.subheader("10. Grupare și agregare după nivelul emisiilor")

            df_agregat_emisii = df_final.groupby("Emisii_Ridicate").agg({
                "EMISII CO2": ["count", "mean", "median", "min", "max"],
                "PIB/capita": ["mean", "median"],
                "ELECTRICITATE": ["mean"],
                "URBAN": ["mean"],
                "SERVICII": ["mean"]
            })

            df_agregat_emisii.index = df_agregat_emisii.index.map({
                0: "Emisii reduse",
                1: "Emisii ridicate"
            })

            st.write("Tabel agregat obținut cu `groupby()` și `agg()`:")
            st.dataframe(df_agregat_emisii, use_container_width=True)

            df_agregat_plot = df_final.groupby("Emisii_Ridicate", as_index=False).agg({
                "PIB/capita": "mean",
                "ELECTRICITATE": "mean",
                "URBAN": "mean",
                "SERVICII": "mean"
            })

            df_agregat_plot["Categorie emisii"] = df_agregat_plot["Emisii_Ridicate"].map({
                0: "Emisii reduse",
                1: "Emisii ridicate"
            })

            indicator_agregat = st.selectbox(
                "Alege indicatorul pentru comparația agregată:",
                ["PIB/capita", "ELECTRICITATE", "URBAN", "SERVICII"],
                key="indicator_agregat"
            )

            fig_agregat = px.bar(
                df_agregat_plot,
                x="Categorie emisii",
                y=indicator_agregat,
                title=f"Media indicatorului {indicator_agregat} pe grupe de emisii",
                text_auto=True
            )

            st.plotly_chart(fig_agregat, use_container_width=True)