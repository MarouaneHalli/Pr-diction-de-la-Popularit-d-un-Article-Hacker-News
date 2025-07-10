import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date, time
import nltk

# Télécharger les stopwords (une fois)
nltk.download('stopwords')


# ========== 1. Chargement des objets sauvegardés ==========
with open('hn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
vectorizer = model_data['vectorizer']
scaler = model_data['scaler']
author_popularity_global = model_data['author_popularity']
average_author_popularity = np.mean(list(author_popularity_global.values()))

# ========== 2. Fonction de nettoyage du titre ==========
def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# ========== 3. Fonction de prédiction ==========
def predict_popularity(title, author, publication_time):
    temp_df = pd.DataFrame({
        'title': [title],
        'author': [author],
        'datetime': [publication_time],
        'descendants': [0]
    })

    temp_df['title_length'] = temp_df['title'].apply(lambda x: len(str(x)))
    temp_df['processed_title'] = temp_df['title'].apply(preprocess_text)
    temp_df['hour'] = temp_df['datetime'].dt.hour
    temp_df['weekday'] = temp_df['datetime'].dt.weekday
    temp_df['is_weekend'] = temp_df['weekday'].isin([5, 6]).astype(int)
    temp_df['author_popularity'] = author_popularity_global.get(author, average_author_popularity)
    temp_df['score_per_comment'] = 0

    X_new = temp_df[['title_length', 'hour', 'weekday', 'is_weekend', 
                     'author_popularity', 'score_per_comment']]

    title_features = vectorizer.transform(temp_df['processed_title'])
    title_features_df = pd.DataFrame(title_features.toarray(), 
                                     columns=vectorizer.get_feature_names_out())

    X_new = pd.concat([X_new.reset_index(drop=True), title_features_df], axis=1)
    X_scaled = scaler.transform(X_new)

    prediction_log = model.predict(X_scaled)[0]
    prediction_real = np.expm1(prediction_log)

    return round(prediction_real, 1), round(prediction_log, 4), temp_df

# ========== 4. Interface Streamlit ==========
st.set_page_config(page_title="Prédiction Popularité Hacker News", layout="centered")

st.title("🔮 Prédiction de la Popularité d’un Article Hacker News")
st.markdown("Prédit le **score (votes)** estimé pour un article à partir de son titre, auteur et date de publication.")

# Entrées utilisateur
title_input = st.text_input("📝 Titre de l’article", "New AI model beats human performance")
author_input = st.text_input("✍️ Auteur", "minimaxir")

# Correction ici : date_input + time_input
date_input = st.date_input("📅 Date de publication", date.today())
time_input = st.time_input("🕒 Heure de publication", datetime.now().time())
datetime_input = datetime.combine(date_input, time_input)

# Bouton de prédiction
if st.button("Prédire la popularité"):
    try:
        predicted_score, log_score, features = predict_popularity(title_input, author_input, datetime_input)

        st.success(f"🎯 Score prédit : **{predicted_score}**")
        st.caption(f"(Score en log: {log_score})")

        with st.expander("🔍 Détails des features utilisées"):
            st.write(features[['title_length', 'hour', 'weekday', 'is_weekend', 'author_popularity']].T)

    except Exception as e:
        st.error(f"🚨 Erreur lors de la prédiction : {e}")
