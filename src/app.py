import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px

st.set_page_config(page_title="EcoMetrics Analytics", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/date-initiale-psw.xlsx", header=1, engine='openpyxl')
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Eroare la încărcarea fișierului: {e}")
        return None


df_raw = load_data()

st.title("EcoMetrics: Socio-Economic & CO2 Analysis")
st.sidebar.header("Meniu Proiect")
menu = st.sidebar.radio("Navigare", ["Date Brute", "Preprocesare & Curatare"])

if df_raw is not None:
    if menu == "Date Brute":
        st.header("Date Regionale (Sursă: Proiect Econometrie)")
        st.write("Vizualizarea setului de date inițial.")
        st.dataframe(df_raw.head(15))
        st.info(f"Setul de date este complet: {df_raw.shape[0]} rânduri și {df_raw.shape[1]} coloane.")

        missing = df_raw.isna().sum().sum()
        if missing == 0:
            st.success("Verificare finalizată: Nu s-au găsit valori lipsă în setul de date.")

    elif menu == "Preprocesare & Curatare":
        st.header("Laborator de Preprocesare")
        st.write("Pregătirea datelor pentru analizele de Regresie, Clasificare și Clusterizare.")

        st.subheader("1. Metode de Codificare (Categorical Encoding)")
        enc_col1, enc_col2 = st.columns(2)
        with enc_col1:
            method = st.radio("Alege metoda de encodare pentru coloana 'TARA':",
                              ["Label Encoding", "One-Hot Encoding (Dummy)"])
        with enc_col2:
            if method == "Label Encoding":
                le = LabelEncoder()
                demo_df = df_raw[['TARA']].copy()
                demo_df['TARA_Encoded'] = le.fit_transform(demo_df['TARA'])
                st.dataframe(demo_df.head())
            else:
                dummy_df = pd.get_dummies(df_raw['TARA'], prefix='Tara').iloc[:, :5]
                st.write("Primele 5 coloane Dummy:")
                st.dataframe(dummy_df.head())

        st.subheader("2. Analiza și Tratarea Valorilor Extreme (Outliers)")
        numeric_cols_all = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        var_to_check = st.selectbox("Selectează variabila pentru analiză:", numeric_cols_all)

        fig_out = px.box(df_raw, y=var_to_check, title=f"Boxplot {var_to_check} - Identificare Outlieri")
        st.plotly_chart(fig_out)

        st.divider()
        st.subheader("Salvare Pipeline Preprocesare")
        st.write("Apasă butonul de mai jos pentru a aplica transformările matematice (Log, Scalare, Target).")

        if st.button("Finalizează și Salvează Datele"):
            df_final = df_raw.copy()

            for col in ['EMISII CO2', 'PIB/capita']:
                limit = df_final[col].quantile(0.95)
                df_final = df_final[df_final[col] <= limit]

            for col in ['EMISII CO2', 'PIB/capita', 'DENSIT.POP.']:
                if col in df_final.columns:
                    df_final[f'log_{col}'] = np.log1p(df_final[col])

            median_val = df_final['EMISII CO2'].median()
            df_final['Emisii_Ridicate'] = (df_final['EMISII CO2'] > median_val).astype(int)

            scaler = StandardScaler()
            numeric_cols = ['PIB/capita', 'ELECTRICITATE', 'URBAN', 'SERVICII', 'DENSIT.POP.']
            df_final_scaled = df_final.copy()
            df_final_scaled[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])

            st.session_state['df_final'] = df_final
            st.session_state['df_final_scaled'] = df_final_scaled
            st.session_state['preprocesare_executata'] = True

            st.success(f"Pipeline executat! Date rămase după eliminarea outlierilor: {df_final.shape[0]} țări.")
            st.write("Coloane noi adăugate: `log_EMISII CO2`, `log_PIB/capita`, `Emisii_Ridicate`.")
            st.balloons()