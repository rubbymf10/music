import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
@st.cache_data
def load_data():
    if not os.path.exists("spotify_songs.csv"):
        with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'playlist_genre', 'lyrics', 'track_popularity'], inplace=True)
    return df

df = load_data()

# Encode popularity as 'High' or 'Low'
df['popularity_label'] = pd.qcut(df['track_popularity'], q=2, labels=["Low", "High"])

# Fit Random Forest model
rf = RandomForestClassifier()
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
rf.fit(df[features], df['popularity_label'])

# TF-IDF for title and lyrics
tfidf_title = TfidfVectorizer(stop_words='english')
tfidf_matrix_title = tfidf_title.fit_transform(df['track_name'])

tfidf_lyrics = TfidfVectorizer(stop_words='english')
tfidf_matrix_lyrics = tfidf_lyrics.fit_transform(df['lyrics'])

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# ----------------- Streamlit Layout ------------------

st.set_page_config(page_title="Rekomendasi Musik", layout="wide")

menu = st.sidebar.radio("Navigasi", ["Home", "Rekomendasi", "Genre", "Histori"])

# ------------------ Halaman HOME ---------------------
if menu == "Home":
    st.title("🎵 10 Lagu Terpopuler")
    top10 = df.sort_values(by='track_popularity', ascending=False).head(10)
    st.table(top10[['track_name', 'track_artist', 'track_popularity']])

    st.header("🔥 5 Lagu Terpopuler Setiap Genre")
    for genre in df['playlist_genre'].unique():
        st.subheader(f"🎧 {genre}")
        top5 = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
        st.table(top5[['track_name', 'track_artist', 'track_popularity']])

# ------------------ Halaman GENRE ---------------------
elif menu == "Genre":
    st.title("🎼 Rekomendasi Berdasarkan Genre")

    selected_genre = st.selectbox("Pilih Genre Lagu", df['playlist_genre'].unique())

    if selected_genre:
        genre_recs = df[df['playlist_genre'] == selected_genre].sort_values(by='track_popularity', ascending=False).head(10)
        st.write(f"🎶 Rekomendasi Lagu Genre **{selected_genre}**")
        for _, row in genre_recs.iterrows():
            st.markdown(f"**{row['track_name']}** oleh *{row['track_artist']}*")
            st.caption(f"Album: {row['track_album_name']} | Popularitas: {row['track_popularity']}")

# ------------------ Halaman REKOMENDASI ------------------
elif menu == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik")

    option = st.radio("Metode Input", ["Pilih Lagu", "Masukkan Judul Manual"])
    
    if option == "Pilih Lagu":
        selected_song = st.selectbox("Pilih Lagu (maks 50)", df['track_name'].unique()[:50])
    else:
        selected_song = st.text_input("Masukkan Judul Lagu")

    if selected_song:
        if selected_song in df['track_name'].values:
            selected_index = df[df['track_name'] == selected_song].index[0]
        else:
            st.error("Judul tidak ditemukan di dataset.")
            selected_index = None

        if selected_index is not None:
            selected_genre = df.loc[selected_index, 'playlist_genre']
            genre_matches = df[df['playlist_genre'] == selected_genre]

            # Genre-based recommendations (sorted by popularity prediction)
            genre_recommendations = genre_matches.copy()
            genre_recommendations['popularity_pred'] = rf.predict(genre_recommendations[features])
            genre_recommendations = genre_recommendations.sort_values(by='popularity_pred', ascending=False).head(10)

            st.subheader("🎯 Rekomendasi Berdasarkan Genre yang Sama")
            for _, row in genre_recommendations.iterrows():
                st.markdown(f"**{row['track_name']}** oleh *{row['track_artist']}*")
                st.caption(f"Album: {row['track_album_name']} | Prediksi Popularitas: {row['popularity_pred']}")

            # Title similarity
            cosine_title = cosine_similarity(tfidf_matrix_title[selected_index], tfidf_matrix_title).flatten()
            similar_title_indices = cosine_title.argsort()[::-1][1:6]
            similar_titles = df.iloc[similar_title_indices].copy()
            similar_titles['popularity_pred'] = rf.predict(similar_titles[features])

            st.subheader("🔁 Rekomendasi Berdasarkan Kemiripan Judul")
            for _, row in similar_titles.iterrows():
                st.markdown(f"**{row['track_name']}** oleh *{row['track_artist']}*")
                st.caption(f"Album: {row['track_album_name']} | Prediksi Popularitas: {row['popularity_pred']}")

            # Lyrics similarity
            cosine_lyrics = cosine_similarity(tfidf_matrix_lyrics[selected_index], tfidf_matrix_lyrics).flatten()
            similar_lyrics_indices = cosine_lyrics.argsort()[::-1][1:6]
            similar_lyrics = df.iloc[similar_lyrics_indices].copy()
            similar_lyrics['popularity_pred'] = rf.predict(similar_lyrics[features])

            st.subheader("📝 Rekomendasi Berdasarkan Kemiripan Lirik")
            for _, row in similar_lyrics.iterrows():
                st.markdown(f"**{row['track_name']}** oleh *{row['track_artist']}*")
                st.caption(f"Album: {row['track_album_name']} | Prediksi Popularitas: {row['popularity_pred']}")

            # Simpan histori
            st.session_state.history.append(selected_song)

# ------------------ Halaman HISTORI ---------------------
elif menu == "Histori":
    st.title("📜 Riwayat Pencarian")
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            st.markdown(f"- {item}")
    else:
        st.write("Belum ada pencarian.")
