import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_data():
    try:
        baza_proiect = Path(__file__).resolve().parent.parent
        cale_fisier = baza_proiect / "data" / "date-initiale-psw.xlsx"

        df = pd.read_excel(cale_fisier, header=1, engine="openpyxl")
        df.columns = [str(col).strip() for col in df.columns]
        return df

    except Exception as e:
        st.error(f"Eroare la încărcarea fișierului: {e}")
        return None


def obtine_coloane_numerice(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def aplica_preprocesare(df_raw):
    df_final = df_raw.copy()

    for col in ["EMISII CO2", "PIB/capita"]:
        if col in df_final.columns:
            limita = df_final[col].quantile(0.95)
            df_final = df_final[df_final[col] <= limita]

    for col in ["EMISII CO2", "PIB/capita", "DENSIT.POP."]:
        if col in df_final.columns:
            df_final[f"log_{col}"] = np.log1p(df_final[col])

    if "EMISII CO2" in df_final.columns:
        valoare_mediana = df_final["EMISII CO2"].median()
        df_final["Emisii_Ridicate"] = (df_final["EMISII CO2"] > valoare_mediana).astype(int)

    coloane_scalare = ["PIB/capita", "ELECTRICITATE", "URBAN", "SERVICII", "DENSIT.POP."]
    coloane_existente = [col for col in coloane_scalare if col in df_final.columns]

    df_final_scaled = df_final.copy()

    if len(coloane_existente) > 0:
        scaler = StandardScaler()
        df_final_scaled[coloane_existente] = scaler.fit_transform(df_final[coloane_existente])

    return df_final, df_final_scaled