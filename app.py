import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API Setup
SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
))

@st.cache_data
def load_data():
    if not os.path.exists("spotify_songs.csv"):
        with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
            zip_ref.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'track_artist', 'playlist_genre', 'lyrics'], inplace=True)
    return df

# Ambil fitur dari link Spotify
def get_audio_features_from_link(url):
    try:
        track_id = url.split("/")[-1].split("?")[0]
        metadata = sp.track(track_id)
        features = sp.audio_features(track_id)[0]
        return {
            'track_name': metadata['name'],
            'track_artist': metadata['artists'][0]['name'],
            'track_album_name': metadata['album']['name'],
            'danceability': features['danceability'],
            'energy': features['energy'],
            'loudness': features['loudness'],
            'speechiness': features['speechiness'],
            'acousticness': features['acousticness'],
            'instrumentalness': features['instrumentalness'],
            'liveness': features['liveness'],
            'valence': features['valence'],
            'tempo': features['tempo']
        }
    except:
        return None

df = load_data()

# Labeling popularitas
pop_threshold = df['track_popularity'].median()
df['popularity_label'] = df['track_popularity'].apply(lambda x: 'High' if x >= pop_threshold else 'Low')

# TF-IDF judul + RandomForest
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['track_name'])

feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
x = df[feature_cols]
y = df['popularity_label']

model = RandomForestClassifier(random_state=42)
model.fit(x, y)

# Simpan histori
if 'history' not in st.session_state:
    st.session_state.history = []

# UI
st.set_page_config(page_title="Spotify Music Recommender", layout="wide")
menu = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Rekomendasi", "Rekomendasi Berdasarkan Genre", "Histori"])

if menu == "Beranda":
    st.title("🎵 10 Musik Terpopuler")
    st.dataframe(df.sort_values(by='track_popularity', ascending=False).head(10)[['track_name', 'track_artist', 'track_popularity']])

    st.subheader("🎶 5 Musik Terpopuler per Genre")
    for genre in df['playlist_genre'].unique():
        st.markdown(f"**🎧 Genre: {genre}**")
        top = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
        st.dataframe(top[['track_name', 'track_artist', 'track_popularity']])

elif menu == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik dari Judul atau Link")

    judul_input = st.selectbox("Pilih Judul Lagu", df['track_name'].unique()[:50])
    manual_input = st.text_input("Atau ketik judul lagu")
    link_input = st.text_input("Atau masukkan link Spotify")

    input_judul = manual_input if manual_input else judul_input

    if st.button("Cari Rekomendasi"):
        if link_input:
            st.info("🔗 Mengambil data dari link Spotify...")
            track_data = get_audio_features_from_link(link_input)
            if track_data:
                input_vector = np.array([track_data[col] for col in feature_cols]).reshape(1, -1)
                sim = cosine_similarity(input_vector, df[feature_cols]).flatten()
                idx = sim.argsort()[-10:][::-1]
                rekom = df.iloc[idx]
                rekom['popularity_pred'] = model.predict(rekom[feature_cols])
                st.subheader(f"🎧 Lagu Mirip dengan: {track_data['track_name']}")
                st.dataframe(rekom[['track_name', 'track_artist', 'playlist_genre', 'popularity_pred']])
                st.session_state.history.append({'input': track_data['track_name'], 'judul_result': rekom[['track_name', 'track_artist']].values.tolist(), 'genre_result': []})
            else:
                st.error("Gagal mengambil data dari link.")
        elif input_judul not in df['track_name'].values:
            st.warning("Judul tidak ditemukan.")
        else:
            idx = df[df['track_name'] == input_judul].index[0]
            sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            top_idx = sim.argsort()[-11:-1][::-1]
            rekom = df.iloc[top_idx]
            rekom['popularity_pred'] = model.predict(rekom[feature_cols])
            st.subheader("🎧 Rekomendasi Berdasarkan Judul")
            st.dataframe(rekom[['track_name', 'track_artist', 'playlist_genre', 'popularity_pred']])
            genre = df.loc[idx, 'playlist_genre']
            genre_top = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
            st.subheader(f"🎼 Rekomendasi Genre Sama: {genre}")
            st.dataframe(genre_top[['track_name', 'track_artist', 'track_popularity']])
            st.session_state.history.append({'input': input_judul, 'judul_result': rekom[['track_name', 'track_artist']].values.tolist(), 'genre_result': genre_top[['track_name', 'track_artist']].values.tolist()})

elif menu == "Rekomendasi Berdasarkan Genre":
    st.title("🎼 Rekomendasi Musik dari Genre")
    genre_input = st.selectbox("Pilih Genre", df['playlist_genre'].unique())

    if st.button("Cari Rekomendasi"):
        genre_top = df[df['playlist_genre'] == genre_input].sort_values(by='track_popularity', ascending=False).head(10)
        genre_top['popularity_pred'] = model.predict(genre_top[feature_cols])
        st.dataframe(genre_top[['track_name', 'track_artist', 'playlist_genre', 'popularity_pred']])
        st.session_state.history.append({'input': genre_input, 'judul_result': [], 'genre_result': genre_top[['track_name', 'track_artist']].values.tolist()})

elif menu == "Histori":
    st.title("🕒 Histori Pencarian")
    if not st.session_state.history:
        st.info("Belum ada histori.")
    else:
        for h in reversed(st.session_state.history):
            st.markdown(f"### 🎼 Input: {h['input']}")
            if h['judul_result']:
                st.markdown("**🎧 Rekomendasi Judul:**")
                for j in h['judul_result']:
                    st.markdown(f"- {j[0]} oleh {j[1]}")
            if h['genre_result']:
                st.markdown("**🎧 Rekomendasi Genre:**")
                for g in h['genre_result']:
                    st.markdown(f"- {g[0]} oleh {g[1]}")
