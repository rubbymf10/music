import streamlit as st
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# Load dan Persiapan Dataset
# ============================
@st.cache_data
def load_data():
    with zipfile.ZipFile("spotify_songs.csv.zip", "r") as zip_ref:
        zip_ref.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'playlist_genre', 'track_popularity'], inplace=True)
    return df

df = load_data()

# Simpan histori rekomendasi
if "history" not in st.session_state:
    st.session_state.history = []

# ============================
# Navigasi Sidebar
# ============================
st.set_page_config(page_title="Sistem Rekomendasi Musik", layout="wide")
menu = st.sidebar.radio("📂 Navigasi", ["Home", "Rekomendasi", "Histori"])

# ============================
# Halaman HOME
# ============================
if menu == "Home":
    st.title("🎶 Sistem Rekomendasi Musik - Home")

    st.subheader("🔝 10 Lagu Terpopuler")
    top_10 = df.sort_values(by="track_popularity", ascending=False).head(10)
    st.dataframe(top_10[['track_name', 'track_artist', 'track_popularity']])

    st.subheader("🎧 Top 5 Lagu Per Genre")
    for genre in df['playlist_genre'].dropna().unique():
        st.markdown(f"##### 🎼 Genre: {genre}")
        top_per_genre = df[df['playlist_genre'] == genre].sort_values(by="track_popularity", ascending=False).head(5)
        st.dataframe(top_per_genre[['track_name', 'track_artist', 'track_popularity']])

# ============================
# Halaman Rekomendasi
# ============================
elif menu == "Rekomendasi":
    st.title("🎯 Rekomendasi Musik Berdasarkan Judul")

    input_method = st.radio("Pilih metode input judul lagu:", ["🔽 Dropdown", "⌨️ Input Manual"])

    if input_method == "🔽 Dropdown":
        selected_title = st.selectbox("Pilih Judul Lagu:", df['track_name'].dropna().unique())
    else:
        selected_title = st.text_input("Ketik Judul Lagu:")

    if selected_title:
        match = df[df['track_name'].str.lower() == selected_title.lower()]
        if not match.empty:
            genre = match.iloc[0]['playlist_genre']
            df_genre = df[df['playlist_genre'] == genre].copy()

            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df_genre['track_name'])

            index_input = df_genre[df_genre['track_name'].str.lower() == selected_title.lower()].index
            if len(index_input) > 0:
                idx = index_input[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                similar_indices = cosine_sim.argsort()[::-1][1:6]

                st.success(f"Rekomendasi lagu dari genre '{genre}' mirip dengan judul '{selected_title}':")
                for i in similar_indices:
                    row = df_genre.iloc[i]
                    st.markdown(f"✅ **{row['track_name']}** oleh *{row['track_artist']}*")
                    st.caption(f"Album: {row['track_album_name']} | Genre: {row['playlist_genre']}")

                # Simpan histori
                st.session_state.history.append({
                    "input": selected_title,
                    "genre": genre,
                    "rekomendasi": [df_genre.iloc[i]['track_name'] for i in similar_indices]
                })
            else:
                st.warning("Tidak ditemukan judul dalam genre yang sama.")
        else:
            st.warning("Judul tidak ditemukan.")

# ============================
# Halaman Histori
# ============================
elif menu == "Histori":
    st.title("🕘 Riwayat Rekomendasi")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"### {i}. Judul Input: {item['input']} (Genre: {item['genre']})")
            for rec in item['rekomendasi']:
                st.markdown(f"- {rec}")
    else:
        st.info("Belum ada histori rekomendasi.")
