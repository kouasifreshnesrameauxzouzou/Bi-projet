import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Configuration de la page
st.set_page_config(page_title="Prédiction Juridique", layout="wide", page_icon="⚖️")

# Fonction pour charger et préparer les données
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()  # Supprime les espaces dans les noms de colonnes
    return data

# Encodage des variables catégoriques
def preprocess_data(data, categorical_columns):
    label_encoders = {}
    encoded_data = data.copy()
    for col in categorical_columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        label_encoders[col] = le
    return encoded_data, label_encoders

# Charger les données
file_path = "data_imputationfinal.xlsx"  # Remplacez par votre chemin de fichier
data = load_data(file_path)

# Colonnes catégoriques existantes
categorical_columns = ['Numero IMPUTATION', 'ORIGINE/AFFAIRES', 'OBJET', 'JURIDICTIONS', 'INSTRUCTIONS ET DELAIS']

# Colonnes supplémentaires pour prédiction utilisateur
additional_columns = ['complexité', 'domaine_juridique', 'urgence']
additional_defaults = {
    'complexité': 'simple',
    'domaine_juridique': 'Droit civil',
    'urgence': 'Moyenne'
}

# Ajouter les nouvelles colonnes aux données
for col, default in additional_defaults.items():
    data[col] = default

# Prétraitement des données
all_columns = categorical_columns + additional_columns
encoded_data, label_encoders = preprocess_data(data, all_columns)

# Diviser les données en caractéristiques et cible
X = encoded_data[all_columns]
y = encoded_data['Numero DOSSIER']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Entraîner le modèle
model = RandomForestClassifier(random_state=50)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Interface Streamlit
st.title("🔍 Prédiction du Numéro de Dossier Juridique")

# En-tête et métriques
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Précision du Modèle", value=f"{accuracy * 100:.2f}%", delta="👍")
with col2:
    st.markdown("### Instructions :")
    st.markdown("""
    1. Remplissez les caractéristiques du dossier dans les menus déroulants.
    2. Cliquez sur **Prédire le numéro du dossier** pour obtenir une prédiction.
    """)

st.markdown("---")

# Formulaire principal
st.header("📝 Entrer les Caractéristiques du Dossier")
cols = st.columns(3)
user_inputs = {}

# Colonnes principales
for i, col in enumerate(categorical_columns):
    with cols[i % 3]:
        user_inputs[col] = st.selectbox(col, label_encoders[col].classes_)

# Colonnes supplémentaires
st.subheader("Caractéristiques Supplémentaires")
additional_cols = st.columns(3)
for i, col in enumerate(additional_columns):
    with additional_cols[i % 3]:
        user_inputs[col] = st.selectbox(col.capitalize(), label_encoders[col].classes_)

# Bouton de prédiction
st.markdown("---")
if st.button("📊 Prédire le Numéro du Dossier"):
    try:
        # Préparer les données d'entrée
        input_data = {
            col: label_encoders[col].transform([user_inputs[col]])[0] for col in all_columns
        }
        input_df = pd.DataFrame([input_data])

        # Prédiction
        prediction = model.predict(input_df)[0]
        st.success(f"🔑 Numéro de Dossier prédit : **{prediction}**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
