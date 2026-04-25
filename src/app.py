import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

sys.path.append(str(Path(__file__).resolve().parent))

from preprocesare import load_data, obtine_coloane_numerice, aplica_preprocesare
from analiza_descriptiva import afiseaza_analiza_descriptiva
from modele.regresie import afiseaza_regresie
from modele.clasificare import afiseaza_clasificare
from modele.clusterizare import afiseaza_clusterizare


st.set_page_config(page_title="EcoMetrics Analytics", layout="wide")

df_raw = load_data()

st.title("EcoMetrics: Socio-Economic & CO2 Analysis")
st.sidebar.header("Meniu Proiect")
menu = st.sidebar.radio(
    "Navigare",
    ["Date Brute", "Preprocesare & Curățare", "Analiză Descriptivă", "Modele"]
)

if df_raw is not None:

    if menu == "Date Brute":
        st.header("Date Regionale (Sursă: Proiect Econometrie)")
        st.write("Vizualizarea setului de date inițial.")
        st.dataframe(df_raw.head(15))
        st.info(f"Setul de date este complet: {df_raw.shape[0]} rânduri și {df_raw.shape[1]} coloane.")

        missing = df_raw.isna().sum().sum()
        if missing == 0:
            st.success("Verificare finalizată: Nu s-au găsit valori lipsă în setul de date.")
        else:
            st.warning(f"Setul de date conține {missing} valori lipsă.")

    elif menu == "Preprocesare & Curățare":
        st.header("Laborator de Preprocesare")
        st.write("Pregătirea datelor pentru analizele de Regresie, Clasificare și Clusterizare.")

        st.subheader("1. Metode de Codificare (Categorical Encoding)")
        enc_col1, enc_col2 = st.columns(2)

        with enc_col1:
            method = st.radio(
                "Alege metoda de encodare pentru coloana 'TARA':",
                ["Label Encoding", "One-Hot Encoding (Dummy)"]
            )

        with enc_col2:
            if method == "Label Encoding":
                le = LabelEncoder()
                demo_df = df_raw[["TARA"]].copy()
                demo_df["TARA_Encoded"] = le.fit_transform(demo_df["TARA"])
                st.dataframe(demo_df.head())
            else:
                dummy_df = pd.get_dummies(df_raw["TARA"], prefix="Tara").iloc[:, :5]
                st.write("Primele 5 coloane Dummy:")
                st.dataframe(dummy_df.head())

        st.subheader("2. Analiza și Tratarea Valorilor Extreme (Outliers)")
        numeric_cols_all = obtine_coloane_numerice(df_raw)
        var_to_check = st.selectbox("Selectează variabila pentru analiză:", numeric_cols_all)

        fig_out = px.box(df_raw, y=var_to_check, title=f"Boxplot {var_to_check} - Identificare Outlieri")
        st.plotly_chart(fig_out, use_container_width=True)

        st.divider()
        st.subheader("Salvare Pipeline Preprocesare")
        st.write("Apasă butonul de mai jos pentru a aplica transformările matematice (Log, Scalare, Target).")

        if st.button("Finalizează și Salvează Datele"):
            df_final, df_final_scaled = aplica_preprocesare(df_raw)

            st.session_state["df_final"] = df_final
            st.session_state["df_final_scaled"] = df_final_scaled
            st.session_state["preprocesare_executata"] = True

            st.success(f"Pipeline executat! Date rămase după eliminarea outlierilor: {df_final.shape[0]} țări.")
            st.write("Coloane noi adăugate: `log_EMISII CO2`, `log_PIB/capita`, `log_DENSIT.POP.`, `Emisii_Ridicate`.")
            st.balloons()

    elif menu == "Analiză Descriptivă":
        if st.session_state.get("preprocesare_executata", False):
            df_final = st.session_state["df_final"]
            afiseaza_analiza_descriptiva(df_raw, df_final)
        else:
            st.warning("Mai întâi trebuie să rulezi secțiunea de preprocesare și să apeși pe butonul de salvare.")

    elif menu == "Modele":
        if st.session_state.get("preprocesare_executata", False):
            df_final = st.session_state["df_final"]
            df_final_scaled = st.session_state["df_final_scaled"]

            optiune_model = st.radio(
                "Alege modelul:",
                ["Regresie", "Clasificare", "Clusterizare"]
            )

            if optiune_model == "Regresie":
                afiseaza_regresie(df_final)

            elif optiune_model == "Clasificare":
                afiseaza_clasificare(df_final)

            elif optiune_model == "Clusterizare":
                afiseaza_clusterizare(df_final_scaled)
        else:
            st.warning("Mai întâi trebuie să rulezi secțiunea de preprocesare și să apeși pe butonul de salvare.")