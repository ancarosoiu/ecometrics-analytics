import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def afiseaza_regresie(df_final):
    st.subheader("Regresie liniară pe variabile logaritmate")
    st.write("Modelul estimează `log_EMISII CO2` folosind indicatori socio-economici selectați.")

    coloana_target = "log_EMISII CO2"
    coloane_features = [
        "log_PIB/capita",
        "ELECTRICITATE",
        "URBAN",
        "SERVICII",
        "log_DENSIT.POP."
    ]

    coloane_necesare = [coloana_target] + coloane_features
    coloane_lipsa = [col for col in coloane_necesare if col not in df_final.columns]

    if len(coloane_lipsa) > 0:
        st.error(f"Lipsesc coloanele necesare regresiei: {coloane_lipsa}")
        return

    df_model = df_final[coloane_necesare].dropna()

    st.write("**Variabila țintă (y):**", coloana_target)
    st.write("**Variabile explicative (X):**", coloane_features)

    st.write("Primele 10 rânduri din setul folosit pentru regresie:")
    st.dataframe(df_model.head(10))

    X = df_model[coloane_features]
    y = df_model[coloana_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.write(f"Observații train: {X_train.shape[0]}")
    st.write(f"Observații test: {X_test.shape[0]}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rezultate_test = pd.DataFrame({
        "Valoare reală": y_test.values,
        "Valoare prezisă": y_pred
    })

    st.write("Primele predicții:")
    st.dataframe(rezultate_test.head(10))

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("MAE", f"{mae:.4f}")

    coeficienti = pd.DataFrame({
        "Variabilă": coloane_features,
        "Coeficient": model.coef_
    })

    st.write("Coeficienții modelului:")
    st.dataframe(coeficienti)
    st.write(f"Intercept: {model.intercept_:.4f}")

    fig_reale_prezise = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Valori reale", "y": "Valori prezise"},
        title="Valori reale vs valori prezise"
    )
    st.plotly_chart(fig_reale_prezise, use_container_width=True)

    fig_coef = px.bar(
        coeficienti,
        x="Variabilă",
        y="Coeficient",
        title="Coeficienții regresiei"
    )
    st.plotly_chart(fig_coef, use_container_width=True)