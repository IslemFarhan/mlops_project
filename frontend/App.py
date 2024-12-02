import streamlit as st
import requests
import pandas as pd
import json
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Health Risk Prediction", layout="wide")

# Style général avec CSS
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

# Ajouter un logo en haut
st.image("C:/Users/MSI/MLOps_Project/frontend/heart.gif", width=100)

def show_classification():
    st.markdown('<h1 class="stTitle">Prédiction des Risques de Santé</h1>', unsafe_allow_html=True)
    st.markdown('<h3>Entrez vos données pour estimer les risques de santé :</h3>', unsafe_allow_html=True)

    # Champs d'entrée utilisateur
    # general_health = st.selectbox("Santé Générale", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    # checkup = st.selectbox("Fréquence des Checkups Médicaux", ["Never", "5 or more years ago", "Within the past 5 years", "Within the past 2 years",'Within the past year'])
    # exercise = st.number_input("Heures d'exercice par semaine", min_value=0.0, max_value=50.0, value=3.0)
    # skin_cancer = st.selectbox("Antécédents de cancer de la peau ?", [("Yes", 1), ("No", 0)])
    # other_cancer = st.selectbox("Autres types de cancer ?", [("Yes", 1), ("No", 0)])
    # diabetes = st.selectbox("Diabète ?", [("Yes", 1), ("No", 0)])
    # arthritis = st.selectbox("Arthrite ?", [("Yes", 1), ("No", 0)])
    # sex = st.selectbox("Sexe", [("Male", 1), ("Female", 0)])
    # age_category = st.selectbox("Catégorie d'âge", ["18-24", "25-29", "30-34", "45-54", "35-39", "40-44","45-49",'50-54','55-59','60-64','65-69','70-74','75-79','80+'])
    # height_cm = st.number_input("Taille (cm)", min_value=100.0, max_value=250.0, value=170.0)
    # weight_kg = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0)
    # bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    # smoking_history = st.selectbox("Fumeur ?", [("Yes", 1), ("No", 0)])
    # alcohol_consumption = st.number_input("Consommation d'alcool (par semaine)", min_value=0.0, max_value=50.0, value=1.0)
    # fruit_consumption = st.number_input("Consommation régulière de fruits ?", min_value=00.0, max_value=100.0, value=17.0)
    # green_vegetables_consumption = st.number_input("Consommation régulière de légumes verts ?", min_value=00.0, max_value=250.0, value=10.0)
    # Default values
    default_data = {
        "General_Health": "Poor",
        "Checkup": "Within the past 2 years",
        "Exercise": "No",
        "Skin_Cancer": "No",
        "Other_Cancer": "No",
        "Diabetes": "No",
        "Arthritis": "Yes",
        "Sex": "Female",
        "Age_Category": "70-74",
        "Height_(cm)": 150.0,
        "Weight_(kg)": 32.66,
        "BMI": 14.54,
        "Smoking_History": "Yes",
        "Alcohol_Consumption": 0.0,
        "Fruit_Consumption": 30.0,
        "Green_Vegetables_Consumption": 16.0,
    }

    st.title("Health Data Input Form")

    # Create input fields with default values
    inputs = {
        key: st.text_input(key, value=str(value)) if isinstance(value, str) else st.number_input(key, value=value)
        for key, value in default_data.items()
    }
    input_data=inputs
    # Bouton pour prédire
    if st.button("Prédire"):
        # Préparer les données
        # input_data = {
        #     "general_health": general_health,
        #     "checkup": checkup,
        #     "exercise": exercise,
        #     "skin_cancer": skin_cancer[1],
        #     "other_cancer": other_cancer[1],
        #     "diabetes": diabetes[1],
        #     "arthritis": arthritis[1],
        #     "sex": sex[1],
        #     "age_category": age_category,
        #     "height_cm": height_cm,
        #     "weight_kg": weight_kg,
        #     "bmi": bmi,
        #     "smoking_history": smoking_history[1],
        #     "alcohol_consumption": alcohol_consumption,
        #     "fruit_consumption": fruit_consumption,
        #     "green_vegetables_consumption": green_vegetables_consumption,
        #}
        # Appel à l'API
        st.write(input_data)
        response = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(input_data))
        predictions = response.json().get("predictions")
        st.write(response)
        # if response.status_code == 200:
        #     result = response.json()
        #     st.success(f"Résultat de la prédiction : {'Risque élevé' if result['prediction'] == 1 else 'Risque faible'}")
        # else:
        #     st.error("Erreur lors de l'appel à l'API !")
        st.write(predictions)
        


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



menu = st.sidebar.selectbox("Menu", ["Description du Problème", "Classification", "Tableau de Bord Power BI"])

if menu == "Description du Problème":
    show_problem_description()
elif menu == "Classification":
    show_classification()
elif menu == "Tableau de Bord Power BI":
    show_powerbi_dashboard()
