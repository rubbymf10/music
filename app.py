import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ==== Spotify API Setup ====
SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

@st.cache_data
def load_data():
    if not os.path.exists("spotify_songs.csv"):
        with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
            zip_ref.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'track_artist', 'playlist_genre', 'lyrics'], inplace=True)
    return df

# Ambil audio features dari Spotify URL
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
    except Exception as e:
        return None

df = load_data()

# Labeling popularitas
pop_threshold = df['track_popularity'].median()
df['popularity_label'] = df['track_popularity'].apply(lambda x: 'High' if x >= pop_threshold else 'Low')

# TF-IDF Vectorizer untuk judul
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['track_name'])

# Fitur dan model
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
x = df[feature_cols]
y = df['popularity_label']
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x, y)

# Simpan histori
if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Spotify Music Recommender", layout="wide")
menu = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Rekomendasi", "Rekomendasi Berdasarkan Genre", "Histori"])

if menu == "Beranda":
    st.title("🎵 10 Musik Terpopuler")
    top10 = df.sort_values(by='track_popularity', ascending=False).head(10)
    st.dataframe(top10[['track_name', 'track_artist', 'track_popularity']])

    st.subheader("🎶 5 Musik Terpopuler dari Setiap Genre")
    genres = df['playlist_genre'].unique()
    for genre in genres:
        st.markdown(f"#### Genre: {genre}")
        top_by_genre = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
        st.dataframe(top_by_genre[['track_name', 'track_artist', 'track_popularity']])

elif menu == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik Berdasarkan Judul atau Link")

    judul_input = st.selectbox("Pilih Judul Lagu", df['track_name'].unique()[:50])
    manual_input = st.text_input("Atau ketik judul lagu secara manual")
    link_input = st.text_input("Atau masukkan link Spotify")

    input_judul = manual_input if manual_input else judul_input

    if st.button("Cari Rekomendasi"):
        # === Jika pakai link Spotify ===
        if link_input:
            st.info("🔗 Mengambil fitur dari link Spotify...")
            track_data = get_audio_features_from_link(link_input)
            if track_data is None:
                st.error("Gagal mengambil data dari link Spotify.")
            else:
                input_vector = np.array([list(track_data[c] for c in feature_cols)])
                cosine_sim = cosine_similarity(input_vector, df[feature_cols]).flatten()
                similar_indices = cosine_sim.argsort()[-10:][::-1]
                result = df.iloc[similar_indices]

                result['popularity_pred'] = rf_model.predict(result[feature_cols])
                st.subheader(f"🎧 Rekomendasi Mirip: {track_data['track_name']} oleh {track_data['track_artist']}")
                st.dataframe(result[['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'popularity_pred']])

                st.session_state.history.append({
                    'input': track_data['track_name'],
                    'genre_result': [],
                    'judul_result': result[['track_name', 'track_artist']].values.tolist()
                })
        # === Jika pakai judul lagu ===
        elif input_judul not in df['track_name'].values:
            st.warning("Judul tidak ditemukan dalam data.")
        else:
            selected_index = df[df['track_name'] == input_judul].index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()
            similar_indices = cosine_sim.argsort()[-11:-1][::-1]
            judul_sama = df.iloc[similar_indices]

            judul_sama['popularity_pred'] = rf_model.predict(judul_sama[feature_cols])
            judul_sama = judul_sama.sort_values(by='popularity_pred', ascending=False)

            st.subheader("🎧 Rekomendasi Berdasarkan Kemiripan Judul")
            st.dataframe(judul_sama[['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'popularity_pred']])

            input_genre = df.loc[selected_index, 'playlist_genre']
            genre_sama = df[df['playlist_genre'] == input_genre].sort_values(by='track_popularity', ascending=False).head(5)

            st.subheader(f"🎼 Rekomendasi Berdasarkan Genre yang Sama: {input_genre}")
            st.dataframe(genre_sama[['track_name', 'track_artist', 'track_popularity']])

            st.session_state.history.append({
                'input': input_judul,
                'genre_result': genre_sama[['track_name', 'track_artist']].values.tolist(),
                'judul_result': judul_sama[['track_name', 'track_artist']].values.tolist()
            })

elif menu == "Rekomendasi Berdasarkan Genre":
    st.title("🎼 Rekomendasi Musik Berdasarkan Genre")
    genre_input = st.selectbox("Pilih Genre Lagu", df['playlist_genre'].unique())

    if st.button("Cari Rekomendasi Genre"):
        genre_sama = df[df['playlist_genre'] == genre_input].sort_values(by='track_popularity', ascending=False).head(10)
        genre_sama['popularity_pred'] = rf_model.predict(genre_sama[feature_cols])
        genre_sama = genre_sama.sort_values(by='popularity_pred', ascending=False)

        st.subheader("🎧 Rekomendasi Berdasarkan Genre")
        st.dataframe(genre_sama[['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'popularity_pred']])

        st.session_state.history.append({
            'input': genre_input,
            'genre_result': genre_sama[['track_name', 'track_artist']].values.tolist(),
            'judul_result': []
        })

elif menu == "Histori":
    st.title("🕒 Histori Pencarian")
    if len(st.session_state.history) == 0:
        st.info("Belum ada histori pencarian.")
    else:
        for idx, item in enumerate(st.session_state.history[::-1]):
            st.markdown(f"### 🎼 Input: {item['input']}")
            if item['genre_result']:
                st.markdown("**🎶 Hasil Berdasarkan Genre:**")
                for track, artist in item['genre_result']:
                    st.markdown(f"- {track} oleh {artist}")
            if item['judul_result']:
                st.markdown("**🎶 Hasil Berdasarkan Judul:**")
                for track, artist in item['judul_result']:
                    st.markdown(f"- {track} oleh {artist}")
