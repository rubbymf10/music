import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os

# Load dataset
DATA_DIR = "data"
ZIP_FILE_NAME = "spotify_songs.csv.zip"
data = load_data()

# Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['lyrics'])

# Classification for popularity (optional step for modeling only)
def train_model():
    df = data.copy()
    df['popularity_label'] = pd.cut(df['track_popularity'], bins=[-1, 50, 100], labels=['Low', 'High'])
    X = df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'tempo']]
    y = df['popularity_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

model = train_model()

# Save history (in session)
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Navigation
page = st.sidebar.selectbox("Pilih Halaman", ["Home", "Rekomendasi", "Histori"])

if page == "Home":
    st.title("🎵 Sistem Rekomendasi Musik")
    st.write("Selamat datang! Aplikasi ini merekomendasikan lagu berdasarkan konten dan popularitas menggunakan algoritma Random Forest.")
    st.write("Silakan pilih menu di samping untuk mulai menggunakan.")

elif page == "Rekomendasi":
    st.title("🔍 Rekomendasi Lagu")
    selected_title = st.selectbox("Pilih Lagu", data['track_name'].unique())
    
    if st.button("Rekomendasikan"):
        idx = data[data['track_name'] == selected_title].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        song_indices = [i[0] for i in sim_scores]
        recommendations = data.iloc[song_indices][['track_name', 'track_artist', 'track_popularity']]

        st.write("### Rekomendasi Lagu:")
        st.dataframe(recommendations)

        # Save to history
        for _, row in recommendations.iterrows():
            st.session_state['history'].append({
                'track_name': row['track_name'],
                'track_artist': row['track_artist'],
                'track_popularity': row['track_popularity']
            })

elif page == "Histori":
    st.title("📜 Histori Rekomendasi")
    if st.session_state['history']:
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist.drop_duplicates())
    else:
        st.write("Belum ada histori rekomendasi.")
