import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def afiseaza_clasificare(df_final):
    st.subheader("Clasificare cu Logistic Regression")
    st.write("Modelul clasifică țările în funcție de variabila `Emisii_Ridicate`.")

    coloana_target = "Emisii_Ridicate"
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
        st.error(f"Lipsesc coloanele necesare clasificării: {coloane_lipsa}")
        return

    df_model = df_final[coloane_necesare].dropna()

    st.write("**Variabila țintă (y):**", coloana_target)
    st.write("**Variabile explicative (X):**", coloane_features)

    st.write("Primele 10 rânduri din setul folosit pentru clasificare:")
    st.dataframe(df_model.head(10))

    X = df_model[coloane_features]
    y = df_model[coloana_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.write(f"Observații train: {X_train.shape[0]}")
    st.write(f"Observații test: {X_test.shape[0]}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rezultate_test = pd.DataFrame({
        "Valoare reală": y_test.values,
        "Valoare prezisă": y_pred
    })

    st.write("Primele predicții:")
    st.dataframe(rezultate_test.head(10))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")

    matrice = confusion_matrix(y_test, y_pred)
    df_matrice = pd.DataFrame(
        matrice,
        index=["Real 0", "Real 1"],
        columns=["Prezis 0", "Prezis 1"]
    )

    st.write("Matricea de confuzie:")
    st.dataframe(df_matrice)

    coeficienti = pd.DataFrame({
        "Variabilă": coloane_features,
        "Coeficient": model.coef_[0]
    })

    st.write("Coeficienții modelului:")
    st.dataframe(coeficienti)

    fig_coef = px.bar(
        coeficienti,
        x="Variabilă",
        y="Coeficient",
        title="Coeficienții clasificării logistice"
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    distributie_pred = pd.DataFrame(pd.Series(y_pred).value_counts()).reset_index()
    distributie_pred.columns = ["Clasa prezisă", "Număr observații"]

    fig_pred = px.bar(
        distributie_pred,
        x="Clasa prezisă",
        y="Număr observații",
        title="Distribuția claselor prezise"
    )
    st.plotly_chart(fig_pred, use_container_width=True)