import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Health Risk Prediction", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff; /* Bleu clair pour le fond */
    }
    .stButton > button {
        background-color: #dc143c; /* Rouge pour les boutons */
        color: white;
    }
    .stTitle {
        color: #1e90ff; /* Bleu pour les titres */
    }
    .stMarkdown h3 {
        color: #ff4500; /* Rouge foncé pour les sous-titres */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("C:/Users/MSI/MLOps_Project/frontend/heart.gif", width=100)

def show_classification():
    st.markdown('<h1 class="stTitle">Prédiction des Risques de Santé</h1>', unsafe_allow_html=True)
    st.markdown('<h3>Entrez vos données pour estimer les risques de santé :</h3>', unsafe_allow_html=True)

    default_data = {
        "general_health": 3,
        "checkup": 1,
        "exercise": 1,
        "skin_cancer": 1,
        "other_cancer": 0,
        "diabetes": 1,
        "arthritis": 0,
        "sex": 1,
        "age_category": 5,
        "height_cm": 170.5,
        "weight_kg": 75.3,
        "bmi": 25.9,
        "smoking_history": 2,
        "alcohol_consumption": 1.3,
        "fruit_consumption": 3,
        "green_vegetables_consumption": 4
    }

    st.title("Formulaire")

    inputs = {
        key: st.number_input(key, value=value)
        for key, value in default_data.items()
    }

    if st.button("Prédire"):
        input_data = {key: float(value) for key, value in inputs.items()}

        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

        if response.status_code == 200:
            prediction = response.json().get("prediction", "Aucune prédiction disponible")
            st.write(f"Prédictions : {prediction}")
        else:
            st.write("Erreur lors de la prédiction. Veuillez réessayer plus tard.")

def show_problem_description():
    st.markdown('<h1 class="stTitle">Description du Problème</h1>', unsafe_allow_html=True)
    st.markdown("""
        ### Problématique
        La prédiction des risques pour la santé est un enjeu majeur dans le domaine de la santé publique.
        L'objectif de cette application est d'aider les utilisateurs à identifier les facteurs de risque
        en fonction de leurs habitudes et conditions de vie, afin de favoriser une prise en charge préventive.
    """)

def show_powerbi_dashboard():
    st.markdown('<h1 class="stTitle">Tableau de Bord Power BI</h1>', unsafe_allow_html=True)
    st.markdown("### Analyse détaillée avec Power BI")
    st.markdown(
        '[Voir le tableau de bord Power BI](https://app.powerbi.com/links/8-VWQ8UIZS?ctid=dbd6664d-4eb9-46eb-99d8-5c43ba153c61&pbi_source=linkShare)',
        unsafe_allow_html=True,
    )

    df = pd.read_csv('C:/Users/MSI/Desktop/mlops_project/data/CVD.csv')

    sex_filter = st.sidebar.multiselect("Sélectionner le sexe", options=df['Sex'].unique(), default=df['Sex'].unique())
    age_filter = st.sidebar.multiselect("Sélectionner la catégorie d'âge", options=df['Age_Category'].unique(), default=df['Age_Category'].unique())

    filtered_data = df[df['Sex'].isin(sex_filter) & df['Age_Category'].isin(age_filter)]

    fig1 = px.pie(filtered_data, names='Sex', values='Alcohol_Consumption', title="Consommation d'alcool par sexe")
    st.plotly_chart(fig1)

    fig2 = px.bar(filtered_data, x='Age_Category', color='Heart_Disease', barmode='group',
                title="Répartition des maladies cardiaques par catégorie d'âge")
    st.plotly_chart(fig2)

    fig3 = px.line(filtered_data, x='Age_Category', y=['Fruit_Consumption', 'Green_Vegetables_Consumption'], 
                title="Consommation de fruits et légumes par catégorie d'âge")
    st.plotly_chart(fig3)

    fig4 = px.bar(filtered_data, x='Sex', color='Heart_Disease', title="Répartition des maladies cardiaques par sexe")
    st.plotly_chart(fig4)

    st.write("Données filtrées selon vos critères :", filtered_data)

menu = st.sidebar.selectbox("Menu", ["Description du Problème", "Classification", "Tableau de Bord Power BI"])

if menu == "Description du Problème":
    show_problem_description()
elif menu == "Classification":
    show_classification()
elif menu == "Tableau de Bord Power BI":
    show_powerbi_dashboard()
