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
import streamlit as st

# Coba manual untuk testing
SPOTIPY_CLIENT_ID = cfcf66dfcecd4bc3bff4cd3ad52362f9
SPOTIPY_CLIENT_SECRET = 3a27fbc9b3a3425c9d523ca62c5226c4

st.write("CLIENT ID:", SPOTIPY_CLIENT_ID[:10])  # print sebagian saja


# ==================== Setup Spotify API ====================
SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ==================== Load Dataset ====================
@st.cache_data
def load_data():
    if not os.path.exists("spotify_songs.csv"):
        with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
            zip_ref.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'track_artist', 'playlist_genre', 'lyrics'], inplace=True)
    return df

df = load_data()

# ==================== Preprocessing ====================
pop_threshold = df['track_popularity'].median()
df['popularity_label'] = df['track_popularity'].apply(lambda x: 'High' if x >= pop_threshold else 'Low')

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['track_name'])

feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo']
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(df[feature_cols], df['popularity_label'])

if 'history' not in st.session_state:
    st.session_state.history = []

# ==================== Streamlit App ====================
st.set_page_config(page_title="Spotify Recommender", layout="wide")
menu = st.sidebar.selectbox("Menu", ["Beranda", "Rekomendasi Judul", "Rekomendasi Link", "Rekomendasi Genre", "Histori"])

if menu == "Beranda":
    st.title("🎵 10 Musik Terpopuler")
    st.dataframe(df.sort_values(by='track_popularity', ascending=False).head(10)[['track_name', 'track_artist', 'track_popularity']])

    st.subheader("🎶 5 Lagu Terpopuler per Genre")
    for genre in df['playlist_genre'].unique():
        st.markdown(f"#### {genre}")
        top = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
        st.dataframe(top[['track_name', 'track_artist', 'track_popularity']])

elif menu == "Rekomendasi Judul":
    st.title("🔍 Rekomendasi Berdasarkan Judul Lagu")
    judul_input = st.selectbox("Pilih Judul Lagu", df['track_name'].unique()[:50])
    manual_input = st.text_input("Atau ketik manual")

    input_judul = manual_input if manual_input else judul_input

    if st.button("Cari Rekomendasi"):
        if input_judul not in df['track_name'].values:
            st.warning("Judul tidak ditemukan.")
        else:
            idx = df[df['track_name'] == input_judul].index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            similar_idx = cosine_sim.argsort()[-11:-1][::-1]
            rekom = df.iloc[similar_idx]
            rekom['popularity_pred'] = rf_model.predict(rekom[feature_cols])
            st.subheader("🎧 Rekomendasi Judul Serupa")
            st.dataframe(rekom[['track_name', 'track_artist', 'playlist_genre', 'popularity_pred']])

            genre_sama = df[df['playlist_genre'] == df.loc[idx, 'playlist_genre']].sort_values(by='track_popularity', ascending=False).head(5)
            st.subheader(f"🎼 Genre Sama: {df.loc[idx, 'playlist_genre']}")
            st.dataframe(genre_sama[['track_name', 'track_artist', 'track_popularity']])

            st.session_state.history.append({
                'input': input_judul,
                'judul_result': rekom[['track_name', 'track_artist']].values.tolist(),
                'genre_result': genre_sama[['track_name', 'track_artist']].values.tolist()
            })

elif menu == "Rekomendasi Link":
    st.title("🔗 Rekomendasi Berdasarkan Link Spotify")
    link = st.text_input("Tempel link lagu Spotify")

    if st.button("Cari dari Link"):
        try:
            track_id = link.split("/")[-1].split("?")[0]
            track_info = sp.track(track_id)
            track_name = track_info['name']
            artist_name = track_info['artists'][0]['name']

            st.success(f"Ditemukan: {track_name} oleh {artist_name}")

            if track_name not in df['track_name'].values:
                st.warning("Lagu ini tidak ada di dataset. Menampilkan lagu serupa dari judul.")
                cosine_sim = cosine_similarity(tfidf_vectorizer.transform([track_name]), tfidf_matrix).flatten()
                similar_idx = cosine_sim.argsort()[-10:][::-1]
                rekom = df.iloc[similar_idx]
            else:
                idx = df[df['track_name'] == track_name].index[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                similar_idx = cosine_sim.argsort()[-11:-1][::-1]
                rekom = df.iloc[similar_idx]

            rekom['popularity_pred'] = rf_model.predict(rekom[feature_cols])
            st.subheader("🎧 Rekomendasi dari Link")
            st.dataframe(rekom[['track_name', 'track_artist', 'playlist_genre', 'popularity_pred']])

        except Exception as e:
            st.error(f"Gagal mengambil data dari Spotify: {e}")

elif menu == "Rekomendasi Genre":
    st.title("🎼 Rekomendasi Berdasarkan Genre")
    genre_input = st.selectbox("Pilih Genre", df['playlist_genre'].unique())

    if st.button("Rekomendasikan"):
        genre_df = df[df['playlist_genre'] == genre_input].sort_values(by='track_popularity', ascending=False).head(10)
        genre_df['popularity_pred'] = rf_model.predict(genre_df[feature_cols])
        st.dataframe(genre_df[['track_name', 'track_artist', 'track_album_name', 'popularity_pred']])

        st.session_state.history.append({
            'input': genre_input,
            'judul_result': [],
            'genre_result': genre_df[['track_name', 'track_artist']].values.tolist()
        })

elif menu == "Histori":
    st.title("📜 Histori Pencarian")
    if len(st.session_state.history) == 0:
        st.info("Belum ada histori.")
    else:
        for item in st.session_state.history[::-1]:
            st.markdown(f"### Input: {item['input']}")
            if item['judul_result']:
                st.markdown("**🎧 Rekomendasi Judul:**")
                for track, artist in item['judul_result']:
                    st.markdown(f"- {track} oleh {artist}")
            if item['genre_result']:
                st.markdown("**🎶 Rekomendasi Genre:**")
                for track, artist in item['genre_result']:
                    st.markdown(f"- {track} oleh {artist}")
