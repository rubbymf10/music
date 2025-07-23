import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import zipfile
import os

# --- Styling dark mode ---
st.markdown("""
    <style>
    .main, .block-container {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #1DB954;
    }
    button[kind="primary"] {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
    }
    button[kind="primary"]:hover {
        background-color: #1ed760;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px;
    }
    .music-card {
        background-color: #282828;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .music-card:hover {
        background-color: #333333;
    }
    .music-cover {
        width: 50px;
        height: 50px;
        color: #1DB954;
        font-size: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 6px;
        flex-shrink: 0;
    }
    .music-info {
        flex-grow: 1;
    }
    .music-title {
        font-weight: 600;
        font-size: 16px;
        margin: 0;
    }
    .music-artist {
        color: #b3b3b3;
        margin: 0;
        font-size: 14px;
    }
    .popularity {
        color: #1DB954;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# === Load Dataset ===
@st.cache_data
def load_data():
    with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df['track_name'] = df['track_name'].astype(str).str.lower()
    df['lyrics'] = df['lyrics'].fillna("").astype(str)
    df['playlist_genre'] = df['playlist_genre'].fillna("Unknown")
    return df

df = load_data()

# === Encode Popularitas ===
def prepare_model(data):
    data = data.copy()
    data['popularity_label'] = ['Tinggi' if x >= data['track_popularity'].median() else 'Rendah' for x in data['track_popularity']]
    le = LabelEncoder()
    data['genre_encoded'] = le.fit_transform(data['playlist_genre'])
    feature_cols = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    clf = RandomForestClassifier()
    clf.fit(data[feature_cols], data['popularity_label'])
    return clf, le, feature_cols

clf, le, feature_cols = prepare_model(df)

# === TF-IDF Vectorizer untuk lirik dan judul ===
tfidf_title = TfidfVectorizer()
tfidf_matrix_title = tfidf_title.fit_transform(df['track_name'])

tfidf_lyrics = TfidfVectorizer()
tfidf_matrix_lyrics = tfidf_lyrics.fit_transform(df['lyrics'])

# === Sesi State untuk histori ===
if 'history' not in st.session_state:
    st.session_state['history'] = []

# === Halaman Navigasi ===
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Rekomendasi", "Histori", "Genre"])

# === BERANDA ===
if page == "Beranda":
    st.title("🎵 10 Musik Terpopuler")
    top10 = df.sort_values(by='track_popularity', ascending=False).head(10)
    st.dataframe(top10[['track_name', 'track_artist', 'track_popularity', 'playlist_genre']])

    st.markdown("### 🔥 Musik Terpopuler dari Setiap Genre")
    for genre in df['playlist_genre'].unique():
        st.markdown(f"#### 🎧 {genre}")
        top_by_genre = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
        st.dataframe(top_by_genre[['track_name', 'track_artist', 'track_popularity']])

# === REKOMENDASI ===
elif page == "Rekomendasi":
    st.title("🎶 Rekomendasi Musik")

    tab1, tab2 = st.tabs(["🔍 Input Judul Lagu", "✍️ Input Manual"])
    selected_index = None

    with tab1:
        all_titles = df['track_name'].unique()[:50]
        selected_title = st.selectbox("Pilih Judul Lagu", all_titles)

        if selected_title:
            selected_title_lower = selected_title.lower()
            if selected_title_lower in df['track_name'].values:
                selected_index = df[df['track_name'] == selected_title_lower].index[0]
                st.success(f"Judul ditemukan: {selected_title}")
            else:
                st.warning("Judul tidak ditemukan dalam data.")

    with tab2:
        manual_input = st.text_input("Masukkan Judul Lagu Manual")
        if manual_input:
            manual_input = manual_input.lower()
            if manual_input in df['track_name'].values:
                selected_index = df[df['track_name'] == manual_input].index[0]
                st.success(f"Judul ditemukan: {manual_input}")
            else:
                st.warning("Judul tidak ditemukan dalam data.")

    if selected_index is not None:
        st.markdown("### 🎼 Rekomendasi Musik")

        selected_genre = df.loc[selected_index, 'playlist_genre']
        genre_matches = df[df['playlist_genre'] == selected_genre]

        # === Judul Similarity ===
        cosine_title = cosine_similarity(tfidf_matrix_title[selected_index], tfidf_matrix_title).flatten()
        sim_title_indices = cosine_title.argsort()[::-1][1:6]
        title_results = df.iloc[sim_title_indices]

        # === Lirik Similarity ===
        cosine_lyrics = cosine_similarity(tfidf_matrix_lyrics[selected_index], tfidf_matrix_lyrics).flatten()
        sim_lyrics_indices = cosine_lyrics.argsort()[::-1][1:6]
        lyrics_results = df.iloc[sim_lyrics_indices]

        # === Genre Match ===
        genre_top = genre_matches.sort_values(by='track_popularity', ascending=False).head(5)

        def predict_popularity(subset):
            return clf.predict(subset[feature_cols])

        st.subheader("🎧 Rekomendasi Berdasarkan Genre Sama")
        genre_top = genre_top.copy()
        genre_top['Prediksi Popularitas'] = predict_popularity(genre_top)
        st.dataframe(genre_top[['track_name', 'track_artist', 'playlist_genre', 'track_album_name', 'Prediksi Popularitas']])

        st.subheader("🎧 Rekomendasi Berdasarkan Kemiripan Judul")
        title_results = title_results.copy()
        title_results['Prediksi Popularitas'] = predict_popularity(title_results)
        st.dataframe(title_results[['track_name', 'track_artist', 'track_album_name', 'Prediksi Popularitas']])

        st.subheader("🎧 Rekomendasi Berdasarkan Kemiripan Lirik")
        lyrics_results = lyrics_results.copy()
        lyrics_results['Prediksi Popularitas'] = predict_popularity(lyrics_results)
        st.dataframe(lyrics_results[['track_name', 'track_artist', 'track_album_name', 'Prediksi Popularitas']])

        # Simpan ke histori
        st.session_state.history.append({
            "input": df.loc[selected_index, 'track_name'],
            "genre_rekom": genre_top[['track_name', 'track_artist']].to_dict('records'),
            "judul_rekom": title_results[['track_name', 'track_artist']].to_dict('records'),
            "lirik_rekom": lyrics_results[['track_name', 'track_artist']].to_dict('records'),
        })

# === HISTORI ===
elif page == "Histori":
    st.title("📜 Histori Rekomendasi")
    if len(st.session_state.history) == 0:
        st.info("Belum ada histori pencarian.")
    else:
        for idx, item in enumerate(st.session_state.history[::-1]):
            st.markdown(f"### 🔎 Input: {item['input']}")
            st.markdown("**Rekomendasi Genre:**")
            for g in item['genre_rekom']:
                st.markdown(f"- {g['track_name']} - {g['track_artist']}")
            st.markdown("**Rekomendasi Judul:**")
            for g in item['judul_rekom']:
                st.markdown(f"- {g['track_name']} - {g['track_artist']}")
            st.markdown("**Rekomendasi Lirik:**")
            for g in item['lirik_rekom']:
                st.markdown(f"- {g['track_name']} - {g['track_artist']}")
            st.markdown("---")

# === GENRE ===
elif page == "Genre":
    st.title("🎙️ Rekomendasi Berdasarkan Genre")
    genre_input = st.selectbox("Pilih Genre", df['playlist_genre'].unique())

    if genre_input:
        top_genre = df[df['playlist_genre'] == genre_input].sort_values(by='track_popularity', ascending=False).head(10)
        top_genre['Prediksi Popularitas'] = clf.predict(top_genre[feature_cols])
        st.dataframe(top_genre[['track_name', 'track_artist', 'track_album_name', 'track_popularity', 'Prediksi Popularitas']])
