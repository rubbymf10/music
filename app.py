import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import zipfile
import os

# === CSS ===
st.markdown("""
<style>
/* === FONTS & RESET === */
html, body, .stApp {
    background-color: #121212;
    color: #FFFFFF;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    transition: all 0.3s ease-in-out;
}

/* === LOGO / TITLE === */
h1, h2, h3, h4 {
    color: #1DB954;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    animation: fadein 1s ease-in-out;
}
@keyframes fadein {
    0% {opacity: 0; transform: translateY(-10px);}
    100% {opacity: 1; transform: translateY(0);}
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background-color: #181818;
    border-right: 1px solid #282828;
}
section[data-testid="stSidebar"] .css-1v0mbdj {
    color: #1DB954 !important;
    font-weight: bold;
}

/* === BUTTONS === */
.stButton > button {
    background-color: #1DB954;
    color: #000;
    font-weight: bold;
    border-radius: 50px;
    padding: 0.6rem 1.5rem;
    border: none;
    transition: 0.3s ease-in-out;
    box-shadow: 0 0 0 rgba(0,0,0,0);
}
.stButton > button:hover {
    background-color: #1ed760;
    box-shadow: 0 0 10px #1DB954;
    transform: scale(1.05);
}

/* === INPUTS / TEXTAREA === */
input, textarea {
    background-color: #2A2A2A !important;
    color: white !important;
    border: 1px solid #333 !important;
    border-radius: 8px;
    padding: 0.5em;
}
.stTextInput > div > input {
    background-color: #2A2A2A !important;
    color: white !important;
}

/* === SELECTBOX === */
.css-1cpxqw2, .stSelectbox > div {
    background-color: #2A2A2A !important;
    color: white !important;
    border-radius: 10px;
}

/* === TABS === */
.stTabs [role="tablist"] {
    border-bottom: 1px solid #333;
    margin-bottom: 1rem;
}
.stTabs [role="tab"] {
    background-color: #181818;
    color: #bbb;
    padding: 0.6rem 1rem;
    margin-right: 5px;
    border: 1px solid transparent;
    border-radius: 10px 10px 0 0;
    transition: 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #1DB954;
    color: black;
    font-weight: bold;
    border: none;
}

/* === DATAFRAMES / TABLE === */
.css-1r6slb0, .dataframe {
    background-color: #1c1c1c !important;
    color: white !important;
    border-radius: 10px;
    font-size: 0.92em;
    border: 1px solid #333;
    animation: fadein 0.8s ease-in-out;
}
thead {
    background-color: #1DB954 !important;
    color: black !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #121212;
}
::-webkit-scrollbar-thumb {
    background: #1DB954;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #1ed760;
}

/* === CUSTOM SPACING === */
.block-container {
    padding: 2rem 3rem;
}

.stRadio > div {
    gap: 10px;
}

/* === SECTION BORDER === */
hr {
    border: none;
    height: 2px;
    background-color: #1DB954;
    margin: 2rem 0;
}

/* === FADE ANIMATION === */
.fade-in {
    animation: fadein 1s ease-in-out;
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
